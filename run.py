"""
Run script — execute the research agent workflow.

Usage:
    python run.py "your research query"    # start a new research session
    python run.py --provider openai        # switch LLM provider
    python run.py --model gpt-4o           # model for the current provider
    python run.py --max-iterations 15      # loop budget
    python run.py --resume <plan_id>       # resume a saved session (by plan ID)
    python run.py --resume <thread_id>     # ... or by raw session/thread ID

Sessions are checkpointed to data/checkpoints.db after every step, so a
run can be interrupted (Ctrl-C, crash, machine sleep) and resumed later.
API keys come from environment variables / .env — nothing is hardcoded here.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import uuid

from config.settings import config, update_config, workflow_config
from db.engine import get_db, init_db
from db.knowledge import get_plan_findings
from graph.workflow import build_app
from langgraph.types import Command


# ── Session helpers ────────────────────────────────────────────────


def resolve_thread_id(ref: str) -> str:
    """Map a plan ID to its LangGraph thread_id; pass raw thread IDs through."""
    try:
        with get_db() as conn:
            row = conn.execute(
                "SELECT thread_id FROM plans WHERE id = ?", (ref,)
            ).fetchone()
        if row and row["thread_id"]:
            return row["thread_id"]
    except sqlite3.Error:
        pass
    return ref


def _print_cost_summary():
    """Print the session cost summary (per-model, per-task)."""
    from utils.cost import get_active_tracker

    tracker = get_active_tracker()
    if not tracker:
        return
    s = tracker.get_summary()
    models = s["models"]
    if len(models) <= 1:
        for model, m in models.items():
            note = "" if m["priced"] else " (not in pricing table)"
            print(f"  Model:    {model}{note}")
    else:
        print("  Models:   (priced per model)")
        for model, m in models.items():
            print(
                f"            {model}: {m['calls']} call(s), "
                f"{m['input_tokens']:,}/{m['output_tokens']:,} tokens, "
                f"${m['cost']:.4f}"
            )
    print(f"  Calls:    {s['total_calls']}")
    print(f"  In/Out:   {s['input_tokens']:,}/{s['output_tokens']:,} tokens")
    print(f"  Cost:     ${s['estimated_cost']:.4f}")
    if s["per_task"]:
        print("  Per-task:")
        for task, count in s["per_task"].items():
            if count > 0:
                print(f"    {task}: {count}")
    if s["unrecorded_calls"]:
        print(f"  Note: {s['unrecorded_calls']} call(s) had no usage metadata")


def _print_review_screen(payload):
    """Render a review/interrupt payload for the user."""
    if not isinstance(payload, dict):
        payload = {"prompt": str(payload)}
    print("\n" + "=" * 60)
    if payload.get("message"):
        print(f"  {payload['message']}")
    print("=" * 60)
    stats = payload.get("stats") or {}
    if stats:
        print("  Stats:")
        for k, v in stats.items():
            print(f"    {k}: {v}")
    qp = payload.get("question_progress") or {}
    if qp:
        print("  Questions:", ", ".join(f"{k}={v}" for k, v in qp.items()))
    open_q = payload.get("open_questions") or []
    if open_q:
        print("  Open questions:")
        for q in open_q[:10]:
            if q.get("status") == "failed":
                err = (q.get("last_error") or "")[:50]
                print(f"    [FAILED] {q.get('question', '')[:70]}  ({err})")
            else:
                print(f"    [{q.get('priority', '?'):>6}] {q.get('question', '')[:70]}")
    n_contra = len(payload.get("contradictions") or [])
    if n_contra:
        print(f"  Contradictions: {n_contra} pair(s) flagged")
    recs = payload.get("recommendations") or []
    if recs:
        print("  Curation recommendations:")
        for r in recs[:10]:
            reason = f" — {r.get('reason', '')[:80]}" if r.get("reason") else ""
            print(f"    [{r.get('action', '?'):>7}] {r.get('id', '?')}{reason}")
    raws = payload.get("raw_entries") or []
    if raws:
        print("  Raw entries (curate <id> / discard <id>):")
        for e in raws[:10]:
            tags = f" ({', '.join(e.get('tags', []))})" if e.get("tags") else ""
            print(f"    {e.get('id', '?')}: {e.get('title', 'Untitled')[:60]}{tags}")
    if payload.get("prompt"):
        print()
        print(payload["prompt"])
    print()


def _print_curation_output(updates: dict):
    """Render the transient display payloads set by curate_knowledge."""
    if updates.get("_error"):
        print(f"  Error: {updates['_error']}")
    elif updates.get("_stats_display"):
        print(f"\n  Stats: {updates['_stats_display']}")
    elif updates.get("_findings_display"):
        display = updates["_findings_display"]
        print(f"\n  === All Findings ({len(display['all'])} total) ===")
        for f in display["valid"]:
            print(f"\n  [VALID] {f.get('summary', '')[:100]}")
            print(f"         Q: {f.get('source_question', '')[:70]}")
            for fact in f.get("facts", []):
                if isinstance(fact, dict):
                    print(
                        f"         - {fact.get('claim', '')[:80]} ({fact.get('confidence', '?')})"
                    )
        if display["discarded"]:
            print(f"\n  --- Discarded ({len(display['discarded'])}) ---")
            for f in display["discarded"]:
                print(f"\n  [DISCARDED] {f.get('summary', '')[:100]}")
                print(f"             Q: {f.get('source_question', '')[:70]}")
        if display.get("contradictions"):
            print(f"\n  --- Contradictions ({len(display['contradictions'])}) ---")
            for c in display["contradictions"]:
                print(f"\n  [{c.get('finding_a')}] vs [{c.get('finding_b')}]")
                print(f"    {c.get('explanation', '')[:150]}")
        if display.get("contradiction_questions"):
            print("\n  --- Suggested resolution questions ---")
            for q in display["contradiction_questions"]:
                print(f"    - {q}")
    elif updates.get("_contradictions_display"):
        display = updates["_contradictions_display"]
        print(f"\n  === Contradictions ({len(display['pairs'])} pairs) ===")
        for c in display["pairs"]:
            print(f"\n  [{c.get('finding_a')}] vs [{c.get('finding_b')}]")
            print(f"    {c.get('explanation', '')}")
        if display.get("questions"):
            print("\n  --- New questions added to bank ---")
            for q in display["questions"]:
                print(f"    - {q}")


def _print_event(event: dict, ctx: dict):
    """Render one node-update stream event."""
    for node_name, updates in event.items():
        if not isinstance(updates, dict):
            continue
        if node_name == "make_plan":
            ctx["plan_ref"] = updates.get("plan_ref", "")
            print(f"  Plan created: {ctx['plan_ref']}")
            similar = updates.get("_similar_plans", [])
            if similar:
                print(f"\n  WARNING: {len(similar)} similar active plan(s) found:")
                for s in similar:
                    status = f"{s['resolved_count']} resolved, {s['pending_count']} pending ({s['progress_pct']}% done)"
                    print(f"     - {s['plan_id']} (similarity={s['similarity']})")
                    print(f"       Query: {s['query'][:80]}...")
                    print(f"       Status: {status}")
        elif node_name == "manage_backlog":
            print("  Selecting next question...")
        elif node_name == "search_execution":
            print("  Searching...")
        elif node_name == "extract_and_verify":
            if ctx.get("plan_ref"):
                n = len(get_plan_findings(ctx["plan_ref"]))
                print(f"  Extracted findings: {n} total (this plan)")
        elif node_name == "consolidation":
            route = updates.get("_route", "?")
            iteration = updates.get("iteration_count", "?")
            if route != "manage_backlog":
                print(f"  Consolidation: {route} (iteration {iteration})")
        elif node_name == "curate_knowledge":
            _print_curation_output(updates)
        elif node_name == "write_findings":
            report = updates.get("final_report", "")
            if report:
                print(f"\n{report}")


def run_loop(app, input_obj, cfg: dict):
    """Drive the graph to completion, answering interrupts as they come.

    Each stream ends either at END or at a suspended interrupt. On an
    interrupt, the review screen is printed, the user's response is read,
    and the graph is resumed with Command(resume=response).
    """
    ctx = {"plan_ref": ""}
    while True:
        interrupted = False
        for event in app.stream(input_obj, cfg, stream_mode="updates"):
            if "__interrupt__" in event:
                interrupted = True
                for intr in event["__interrupt__"]:
                    _print_review_screen(intr.value)
                response = input("Your command> ").strip() or "continue"
                input_obj = Command(resume=response)
            else:
                _print_event(event, ctx)
        if not interrupted:
            # Stream ended without suspending → graph reached END.
            # (Must key off this pass, not the type of input_obj: after a
            # resume, input_obj is still a Command even once the graph is done.)
            break


def main():
    parser = argparse.ArgumentParser(description="Research Agent Workflow")
    parser.add_argument("query", nargs="?", default=None, help="Research query")
    parser.add_argument(
        "--provider",
        choices=["local", "openai", "anthropic"],
        help="LLM provider (overrides config)",
    )
    parser.add_argument("--model", default=None, help="Model name (provider-specific)")
    parser.add_argument(
        "--max-iterations", type=int, default=None, help="Max research iterations"
    )
    parser.add_argument(
        "--resume", default=None, help="Resume a saved session by plan ID or thread ID"
    )
    parser.add_argument(
        "--plan-id",
        default=None,
        help="Continue an existing plan by ID (skips planning)",
    )
    args = parser.parse_args()

    # ── Apply config overrides ──────────────────────────────────
    if args.provider:
        update_config(provider=args.provider)
    if args.model:
        provider = args.provider or config.provider
        update_config(**{f"{provider}_model": args.model})
    if args.max_iterations:
        update_config(max_iterations=args.max_iterations)

    model_name = getattr(config, f"{config.provider}_model", config.local_model)
    print(f"Research Agent — Provider: {config.provider}, Model: {model_name}")

    # ── Initialize database (also runs column migrations) ───────
    init_db()

    app = build_app()  # SQLite-backed checkpointer → resumable sessions

    # ── Resume or rewind an existing session ──────────────────────
    if args.resume:
        thread_id = resolve_thread_id(args.resume)
        cfg = {"configurable": {"thread_id": thread_id}}
        state = app.get_state(cfg)
        if not state.values and not state.next:
            print(f"No session found for '{args.resume}'. Start a new run instead.")
            sys.exit(1)

        # (Rewinding to an older checkpoint is only available in-chat via
        # the 'rewind' command; this LangGraph version's get_state doesn't
        # accept nested checkpoint configs for a CLI flag.)
        print(f"Resuming session: {thread_id}")
        try:
            if state.next:
                # Suspended at a review interrupt — re-display the payload
                # stored in the checkpoint (no LLM calls needed).
                try:
                    payload = state.tasks[0].interrupts[0].value
                except (IndexError, AttributeError):
                    payload = {"prompt": "Session is suspended. Respond to continue."}
                _print_review_screen(payload)
                response = input("Your command> ").strip() or "continue"
                input_obj = Command(resume=response)
            else:
                # Completed or crashed mid-node: continue from the checkpoint.
                input_obj = None
            run_loop(app, input_obj, cfg)
        except (KeyboardInterrupt, EOFError):
            print(
                f"\n\nInterrupted. Resume later with:\n  python run.py --resume {thread_id}"
            )
            sys.exit(130)
        return

    # ── Continue existing plan (skip planning) ──────────────────────
    if args.plan_id:
        plan_id = args.plan_id
        # Verify plan exists
        with get_db() as conn:
            row = conn.execute(
                "SELECT query FROM plans WHERE id = ?", (plan_id,)
            ).fetchone()
        if not row:
            print(f"Plan '{plan_id}' not found. Exiting.")
            sys.exit(1)
        query = row["query"]

        thread_id = str(uuid.uuid4())
        cfg = {"configurable": {"thread_id": thread_id}}

        print(f"Continuing existing plan: {plan_id}")
        print(f"Query: {query[:80]}...")
        print(f"Max iterations: {workflow_config.max_iterations}")
        print(f"Session: {thread_id}")
        print("-" * 60)

        # Pass plan_ref directly to skip the planning node
        try:
            run_loop(app, {"plan_ref": plan_id, "original_query": query}, cfg)
        except (KeyboardInterrupt, EOFError):
            print(
                f"\n\nInterrupted. Progress is checkpointed — resume with:\n  python run.py --resume {thread_id}"
            )
            sys.exit(130)

        print("\n" + "=" * 60)
        print("Research complete.")
        _print_cost_summary()
        return

    # ── New session ─────────────────────────────────────────────
    query = args.query
    if not query:
        query = input("Research query> ").strip()
        if not query:
            print("No query provided. Exiting.")
            sys.exit(1)

    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}

    print(f'Starting research on: "{query}"')
    print(f"Max iterations: {workflow_config.max_iterations}")
    print(f"Session: {thread_id}")
    print("  (resume later with --resume <plan_id> or the session ID)")
    print("-" * 60)

    try:
        run_loop(app, {"original_query": query}, cfg)
    except (KeyboardInterrupt, EOFError):
        print(
            f"\n\nInterrupted. Progress is checkpointed — resume with:\n  python run.py --resume {thread_id}"
        )
        sys.exit(130)

    _print_cost_summary()
    print("\n" + "=" * 60)
    print("Research complete.")


if __name__ == "__main__":
    main()
