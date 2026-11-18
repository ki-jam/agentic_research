"""
Run script — execute the research agent workflow.

Usage:
    python run.py                          # interactive menu (ask / plans /
                                           # backlog / results / reports / stats)
    python run.py "your research query"    # start a new research session
    python run.py ask "your query"         # subcommand form (--provider, --model,
                                           # --max-iterations, --plan-id supported)
    python run.py resume <plan_id|thread>  # resume a saved session
    python run.py plans                    # browse plans (pick one to inspect)
    python run.py plan <plan_id>           # show one plan's questions & reports
    python run.py backlog [plan_id]        # review & re-prioritize open questions
    python run.py results [plan_id]        # browse findings & knowledge entries
    python run.py report [plan_id]         # read saved reports
    python run.py stats                    # knowledge-base stats
    python run.py models                   # view/change the model per step

Legacy forms still work:
    python run.py "your query" --provider openai --model gpt-4o
    python run.py --resume <plan_id>
    python run.py --plan-id <plan_id>

Sessions are checkpointed to data/checkpoints.db after every step, so a
run can be interrupted (Ctrl-C, crash, machine sleep) and resumed later.
API keys come from environment variables / .env — nothing is hardcoded here.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import uuid

from config.settings import (
    config,
    resolve_step_model,
    search_config,
    update_config,
    workflow_config,
)
from db.engine import get_db, init_db
from db.knowledge import get_in_progress_question, get_plan_findings
from graph.workflow import build_app
from langgraph.types import Command
from ui import review as ui_review
from ui import theme
from ui import views as ui_views

_SUBCOMMANDS = frozenset(
    {
        "ask",
        "resume",
        "plans",
        "plan",
        "backlog",
        "topics",
        "results",
        "report",
        "stats",
        "models",
    }
)


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


def _read(prompt: str) -> str:
    """Read one line via builtins.input, resolved at call time (testable)."""
    import builtins

    return builtins.input(prompt)


def _prompt_query() -> str:
    """Ask for a research query; '' on EOF/Ctrl-C, blank input, or a quit
    token (a 'quit;' typed at this prompt once spawned a whole plan)."""
    try:
        raw = _read(theme.prompt("Research query> ")).strip()
    except (EOFError, KeyboardInterrupt):
        return ""
    if raw.lower().rstrip(";!.") in ("quit", "exit", "q"):
        return ""
    return raw


def run_loop(app, input_obj, cfg: dict):
    """Drive the graph to completion, answering interrupts as they come.

    Each stream ends either at END or at a suspended interrupt. On an
    interrupt, the review screen is rendered and the user's response is
    read (unknown input re-prompts — it never silently continues the
    graph), and the graph is resumed with Command(resume=response).
    """
    ctx = {"plan_ref": ""}
    while True:
        interrupted = False
        for event in app.stream(input_obj, cfg, stream_mode="updates"):
            if "__interrupt__" in event:
                interrupted = True
                for intr in event["__interrupt__"]:
                    response = ui_review.ask_review_response(intr.value)
                    input_obj = Command(resume=response)
            else:
                _print_event(event, ctx)
        if not interrupted:
            # Stream ended without suspending → graph reached END.
            # (Must key off this pass, not the type of input_obj: after a
            # resume, input_obj is still a Command even once the graph is done.)
            break


def _print_cost_summary():
    """Print the session cost summary (per-model, per-task)."""
    from utils.cost import get_active_tracker

    tracker = get_active_tracker()
    if not tracker:
        return
    s = tracker.get_summary()
    models = s["models"]
    print(theme.section("Session cost"))
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


def _print_effective_config() -> None:
    """Show which model each step runs on and which search engines are active.

    The config is env-var-driven and otherwise silent, so a new user cannot
    tell what is about to run. This banner makes the session log
    self-explanatory, and it resolves every step so an invalid
    *_PROVIDER value fails at startup instead of mid-session.
    """
    print(theme.section("Effective config"))
    try:
        default = f"{config.provider}/{getattr(config, f'{config.provider}_model', config.local_model)}"
        overrides = []
        for step in ("planning", "extraction", "curation", "report"):
            provider, model = resolve_step_model(step)
            if f"{provider}/{model}" != default:
                overrides.append(f"    {step + ':':<12} {provider}/{model}")
    except ValueError as e:
        print(theme.red(f"  Config error: {e}"))
        sys.exit(2)
    if overrides:
        print(f"    model:      {default} (all steps, except:)")
        print("\n".join(overrides))
    else:
        print(f"    model:      {default} (all steps)")
    citations = "on" if search_config.enable_citation_discovery else "off"
    cache = (
        "never"
        if search_config.cache_ttl_days == 0
        else f"{search_config.cache_ttl_days}d"
    )
    print(
        f"    search:     primary={search_config.primary}  arxiv={search_config.force_arxiv}  "
        f"github={search_config.force_github}  citations={citations}  cache={cache}"
    )


def _print_curation_output(updates: dict):
    """Render the transient display payloads set by curate_knowledge."""
    if updates.get("_error"):
        print(theme.red(f"  Error: {updates['_error']}"))
    elif updates.get("_stats_display"):
        ui_views.render_stats(updates["_stats_display"])
    elif updates.get("_findings_display"):
        display = updates["_findings_display"]
        print(theme.section(f"All findings ({len(display['all'])} total)"))
        for f in display["valid"]:
            print(theme.green(f"\n  [VALID] {f.get('summary', '')[:100]}"))
            print(f"         Q: {f.get('source_question', '')[:70]}")
            for fact in f.get("facts", []):
                if isinstance(fact, dict):
                    conf = theme.dim(f"({fact.get('confidence', '?')})")
                    print(f"         - {fact.get('claim', '')[:80]} {conf}")
        if display["discarded"]:
            print(theme.dim(f"\n  ── Discarded ({len(display['discarded'])}) ──"))
            for f in display["discarded"]:
                print(theme.dim(f"\n  [DISCARDED] {f.get('summary', '')[:100]}"))
                print(f"             Q: {f.get('source_question', '')[:70]}")
        if display.get("contradictions"):
            print(
                theme.yellow(
                    f"\n  ── Contradictions ({len(display['contradictions'])}) ──"
                )
            )
            for c in display["contradictions"]:
                print(f"\n  [{c.get('finding_a')}] vs [{c.get('finding_b')}]")
                print(f"    {c.get('explanation', '')[:150]}")
        if display.get("contradiction_questions"):
            print("\n  ── Suggested resolution questions ──")
            for q in display["contradiction_questions"]:
                print(f"    - {q}")
    elif updates.get("_contradictions_display"):
        display = updates["_contradictions_display"]
        print(theme.section(f"Contradictions ({len(display['pairs'])} pairs)"))
        for c in display["pairs"]:
            print(f"\n  [{c.get('finding_a')}] vs [{c.get('finding_b')}]")
            print(f"    {c.get('explanation', '')}")
        if display.get("questions"):
            print("\n  ── New questions added to bank ──")
            for q in display["questions"]:
                print(f"    - {q}")


def _print_event(event: dict, ctx: dict):
    """Render one node-update stream event."""
    for node_name, updates in event.items():
        if not isinstance(updates, dict):
            continue
        if node_name == "make_plan":
            ctx["plan_ref"] = updates.get("plan_ref", "")
            print(theme.green(f"  Plan created: {ctx['plan_ref']}"))
            similar = updates.get("_similar_plans", [])
            if similar:
                print(
                    theme.yellow(
                        f"\n  WARNING: {len(similar)} similar active plan(s) found:"
                    )
                )
                for s in similar:
                    status = (
                        f"{s['resolved_count']} resolved, "
                        f"{s['pending_count']} pending ({s['progress_pct']}% done)"
                    )
                    print(f"     - {s['plan_id']} (similarity={s['similarity']})")
                    print(f"       Query: {s['query'][:80]}...")
                    print(f"       Status: {status}")
        elif node_name == "manage_backlog":
            # manage_backlog_node returns {} — the pick is a DB side
            # effect — so look the chosen question up to show what this
            # iteration will work on.
            q = (
                get_in_progress_question(ctx["plan_ref"])
                if ctx.get("plan_ref")
                else None
            )
            if q:
                print(f"  {theme.cyan('Question:')} {q['question'][:100]}")
            else:
                print(theme.dim("  Selecting next question..."))
        elif node_name == "search_execution":
            print(theme.dim("  Searching..."))
        elif node_name == "extract_and_verify":
            if ctx.get("plan_ref"):
                n = len(get_plan_findings(ctx["plan_ref"]))
                delta = n - ctx.get("_findings_total", 0)
                ctx["_findings_total"] = n
                new = f" (+{delta} new)" if delta > 0 else ""
                print(f"  {theme.cyan('Findings:')} {n} total{new} (this plan)")
        elif node_name == "consolidation":
            route = updates.get("_route", "?")
            iteration = updates.get("iteration_count", "?")
            if isinstance(iteration, int):
                # Always show the iteration header — it is the anchor for
                # "where is this session?" when tailing the log.
                print(theme.section(f"Iteration {iteration}"))
            if route != "manage_backlog":
                print(theme.dim(f"  Consolidation: {route}"))
        elif node_name == "prepare_review":
            print(theme.dim("  Review checkpoint — answer the prompt below"))
        elif node_name == "curate_knowledge":
            _print_curation_output(updates)
        elif node_name == "write_findings":
            report = updates.get("final_report", "")
            if report:
                print(f"\n{report}")


# ── Session runners ────────────────────────────────────────────────


def _run_new_session(query: str, plan_id: str = "") -> None:
    """Start (or continue) a research session and drive it to completion."""
    model_name = getattr(config, f"{config.provider}_model", config.local_model)
    print(
        theme.header(
            "Research Agent", f"Provider: {config.provider} · Model: {model_name}"
        )
    )
    _print_effective_config()

    init_db()
    app = build_app()  # SQLite-backed checkpointer → resumable sessions

    if plan_id:
        # Continue an existing plan (skip planning)
        with get_db() as conn:
            row = conn.execute(
                "SELECT query FROM plans WHERE id = ?", (plan_id,)
            ).fetchone()
        if not row:
            print(theme.yellow(f"Plan {plan_id!r} not found. Exiting."))
            sys.exit(1)
        query = row["query"]
        thread_id = str(uuid.uuid4())
        cfg = {"configurable": {"thread_id": thread_id}}
        print(f"  Continuing existing plan: {plan_id}")
        print(f"  Query: {query[:80]}...")
        print(f"  Max iterations: {workflow_config.max_iterations}")
        print(theme.dim(f"  Session: {thread_id}"))
        print("-" * 60)
        input_obj = {"plan_ref": plan_id, "original_query": query}
    else:
        thread_id = str(uuid.uuid4())
        cfg = {"configurable": {"thread_id": thread_id}}
        print(f'  Starting research on: "{query}"')
        print(f"  Max iterations: {workflow_config.max_iterations}")
        print(
            theme.dim(
                f"  Session: {thread_id}  (resume later with: python run.py resume <plan_id>)"
            )
        )
        print("-" * 60)
        input_obj = {"original_query": query}

    try:
        run_loop(app, input_obj, cfg)
    except (KeyboardInterrupt, EOFError):
        print(
            f"\n\nInterrupted. Progress is checkpointed — resume with:\n"
            f"  python run.py resume {thread_id}"
        )
        sys.exit(130)

    _print_cost_summary()
    print("\n" + theme.bold("Research complete."))


def _run_resume(ref: str) -> None:
    """Resume a saved session (by plan ID or raw thread ID)."""
    init_db()
    app = build_app()

    thread_id = resolve_thread_id(ref)
    cfg = {"configurable": {"thread_id": thread_id}}
    state = app.get_state(cfg)
    if not state.values and not state.next:
        print(theme.yellow(f"No session found for '{ref}'. Start a new run instead:"))
        print('  python run.py ask "your query"')
        sys.exit(1)

    print(theme.header("Resuming session", thread_id))
    _print_effective_config()
    try:
        # A mid-node crash also leaves state.next set, but without any
        # interrupt payload. Never show the review screen in that case:
        # the user's reply would be consumed as a resume value for a
        # graph that isn't actually suspended.
        interrupted = next(
            (t for t in state.tasks if getattr(t, "interrupts", None)), None
        )
        if interrupted is not None:
            # Suspended at a review interrupt — re-display the payload
            # stored in the checkpoint (no LLM calls needed).
            payload = interrupted.interrupts[0].value
            response = ui_review.ask_review_response(payload)
            input_obj = Command(resume=response)
        else:
            # Completed or crashed mid-node: continue from the checkpoint.
            if state.next:
                print(
                    theme.dim(
                        "  Session crashed mid-node — continuing from the last checkpoint."
                    )
                )
            input_obj = None
        run_loop(app, input_obj, cfg)
    except (KeyboardInterrupt, EOFError):
        print(
            f"\n\nInterrupted. Resume later with:\n  python run.py resume {thread_id}"
        )
        sys.exit(130)


# ── Interactive menu ───────────────────────────────────────────────

_MENU = """
  1) New research   — ask a question and start a session
  2) Plans          — browse plans and their progress
  3) Backlog        — review & re-prioritize open questions
  4) Results        — browse findings and knowledge entries
  5) Reports        — read saved reports
  6) Stats          — knowledge-base stats
  7) Models         — view/change the model (and provider) per step
  0) Quit
"""


def _interactive_menu() -> None:
    """Bare `python run.py` — loop over the top-level menu."""
    init_db()
    print(theme.header("Research Agent", "what do you want to do?"))
    while True:
        print(_MENU)
        try:
            line = _read(theme.prompt("> ")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if line in ("0", "q", "quit", "exit"):
            return
        if line == "1":
            query = _prompt_query()
            if query:
                _run_new_session(query)
        elif line == "2":
            ui_views.show_plans()
        elif line == "3":
            ui_views.show_backlog()
        elif line == "4":
            ui_views.show_results()
        elif line == "5":
            ui_views.show_reports()
        elif line == "6":
            ui_views.show_stats()
        elif line == "7":
            ui_views.show_models()
        else:
            print(theme.yellow("  Type 1–7 or 0 to quit."))


# ── CLI dispatch ───────────────────────────────────────────────────


def _add_run_options(parser: argparse.ArgumentParser) -> None:
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
        "--plan-id",
        default=None,
        help="Continue an existing plan by ID (skips planning)",
    )


def _apply_config_overrides(args) -> None:
    """Apply --provider / --model / --max-iterations to the global config."""
    if getattr(args, "provider", None):
        update_config(provider=args.provider)
    if getattr(args, "model", None):
        provider = args.provider or config.provider
        update_config(**{f"{provider}_model": args.model})
    if getattr(args, "max_iterations", None):
        update_config(max_iterations=args.max_iterations)


def main(argv=None):
    """CLI entry point.

    Dispatch: a recognized subcommand (ask/resume/plans/plan/backlog/
    topics/results/report/stats) → subcommand handler; no args →
    interactive menu; anything else → the legacy form (query + flags).
    """
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] in _SUBCOMMANDS:
        sub, rest = argv[0], argv[1:]

        if sub == "ask":
            parser = argparse.ArgumentParser(
                prog="run.py ask", description="Start a research session"
            )
            parser.add_argument("query", nargs="*", help="Research query")
            _add_run_options(parser)
            args = parser.parse_args(rest)
            _apply_config_overrides(args)
            if args.plan_id:
                _run_new_session("", plan_id=args.plan_id)
            else:
                query = " ".join(args.query).strip()
                if not query:
                    query = _prompt_query()
                    if not query:
                        print("No query provided. Exiting.")
                        sys.exit(1)
                _run_new_session(query)
            return

        if sub == "resume":
            parser = argparse.ArgumentParser(
                prog="run.py resume", description="Resume a saved session"
            )
            parser.add_argument("ref", help="Plan ID or session/thread ID")
            _add_run_options(parser)
            args = parser.parse_args(rest)
            _apply_config_overrides(args)
            _run_resume(args.ref)
            return

        init_db()
        if sub == "plans":
            ui_views.show_plans()
            return
        if sub == "plan":
            if not rest:
                print("Usage: python run.py plan <plan_id>")
                sys.exit(2)
            ui_views.show_plan_detail(rest[0])
            return
        if sub == "backlog":
            ui_views.show_backlog(rest[0] if rest else "")
            return
        if sub == "topics":
            ui_views.show_topics(rest[0] if rest else "")
            return
        if sub == "results":
            ui_views.show_results(rest[0] if rest else "")
            return
        if sub == "report":
            ui_views.show_reports(rest[0] if rest else "")
            return
        if sub == "stats":
            ui_views.show_stats()
            return
        if sub == "models":
            ui_views.show_models()
            return

    if not argv:
        _interactive_menu()
        return

    # ── Legacy form: run.py "query" [--flags] / run.py --resume <id> ──
    parser = argparse.ArgumentParser(description="Research Agent Workflow")
    parser.add_argument(
        "query", nargs="*", help="Research query (unquoted words are joined)"
    )
    _add_run_options(parser)
    parser.add_argument(
        "--resume", default=None, help="Resume a saved session by plan ID or thread ID"
    )
    args = parser.parse_args(argv)

    _apply_config_overrides(args)

    # (Rewinding to an older checkpoint is only available in-chat via the
    # 'rewind' command; this LangGraph version's get_state doesn't accept
    # nested checkpoint configs for a CLI flag.)
    if args.resume:
        _run_resume(args.resume)
        return

    if args.plan_id:
        _run_new_session("", plan_id=args.plan_id)
        return

    query = " ".join(args.query).strip()
    if not query:
        query = _prompt_query()
        if not query:
            print("No query provided. Exiting.")
            sys.exit(1)
    _run_new_session(query)


if __name__ == "__main__":
    main()
