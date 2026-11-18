"""
Browsers for plans, backlogs, results, reports, and stats.

Everything interactive takes ``input_fn=None``, resolved to builtins.input
at call time (so tests can patch builtins.input or pass a scripted
function). Every browser exits cleanly on EOF/Ctrl-C.
"""

from __future__ import annotations

from . import theme


def _input_fn(input_fn):
    """Resolve the input callable at call time (test patches work)."""
    if input_fn is None:
        import builtins

        input_fn = builtins.input
    return input_fn


def _plan_progress(plan_id: str) -> tuple:
    """(total, open, pct_closed) for a plan's question bank."""
    from db.engine import get_db

    with get_db() as conn:
        row = conn.execute(
            """SELECT COUNT(*),
                      SUM(CASE WHEN status NOT IN ('resolved','discarded') THEN 1 ELSE 0 END)
               FROM questions WHERE plan_id = ?""",
            (plan_id,),
        ).fetchone()
    total = row[0] or 0
    open_count = row[1] or 0
    pct = int((total - open_count) / total * 100) if total else 0
    return total, open_count, pct


def _list_plans() -> list:
    """All plans, newest first, each with its (total, open, pct) progress."""
    from db.engine import get_db

    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, query, thread_id, created_at FROM plans "
            "ORDER BY created_at DESC, id DESC"
        ).fetchall()
    plans = []
    for r in rows:
        d = dict(r)
        d["progress"] = _plan_progress(d["id"])
        plans.append(d)
    return plans


def _plan_state(plan: dict) -> str:
    """'active' or 'done' based on the question bank."""
    return "active" if plan["progress"][1] > 0 else "done"


# ── Plans ──────────────────────────────────────────────────────────


def show_plans(input_fn=None) -> None:
    """List all plans; pick one (number or ID) to inspect, loop back."""
    input_fn = _input_fn(input_fn)
    while True:
        plans = _list_plans()
        print(theme.header("Plans", f"{len(plans)} plan(s)"))
        if not plans:
            print(
                "  No plans yet — start one with "
                + theme.cyan('python run.py ask "your query"')
            )
            return
        for i, p in enumerate(plans, 1):
            total, open_count, pct = p["progress"]
            badge = theme.green("done") if open_count == 0 else theme.cyan("active")
            print(f"  {i:>2}. {p['id']}  {badge}  {theme.dim(str(pct) + '% closed')}")
            print(f"      {p['query'][:78]}")
            thread = (p["thread_id"] or "no thread")[:36]
            print(f"      {theme.dim(p['created_at'][:19] + '  ' + thread)}")
        print(
            theme.dim("  Type a number or plan ID to inspect · r = refresh · q = quit")
        )
        try:
            line = input_fn(theme.prompt("plan> ")).strip()
        except (EOFError, KeyboardInterrupt):
            return
        low = line.lower()
        if low in ("q", "quit"):
            return
        if low == "r":
            continue
        if not low:
            continue
        target = None
        if low.isdigit() and 1 <= int(low) <= len(plans):
            target = plans[int(low) - 1]["id"]
        elif any(p["id"] == low for p in plans):
            target = low
        if target is None:
            print(theme.yellow(f"  No plan {line!r} — type a number or plan ID."))
            continue
        show_plan_detail(target)


def show_plan_detail(plan_id: str) -> None:
    """Print one plan's query, thread, question list, and reports."""
    from db.engine import get_db
    from db.knowledge import get_plan

    plan = get_plan(plan_id)
    if not plan:
        print(theme.yellow(f"  Plan {plan_id!r} not found."))
        return
    total, open_count, pct = _plan_progress(plan_id)
    print(theme.header("Plan", plan_id))
    print(f"  Query:    {plan.get('query', '')}")
    print(f"  Thread:   {plan.get('thread_id') or '(none)'}")
    print(f"  Created:  {plan.get('created_at', '')}")
    print(f"  Progress: {theme.progress_bar(total - open_count, total)}")

    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, question, status, priority, last_error "
            "FROM questions WHERE plan_id = ? ORDER BY created_at, id",
            (plan_id,),
        ).fetchall()
    if rows:
        print(theme.section("Questions"))
        for r in rows:
            err = (
                f"  {theme.dim('(' + (r['last_error'] or '')[:50] + ')')}"
                if r["status"] == "failed"
                else ""
            )
            print(
                f"  {theme.status_badge(r['status'])} "
                f"{theme.priority_badge(r['priority'])} {r['id']}"
            )
            print(f"      {r['question'][:90]}{err}")

    with get_db() as conn:
        reps = conn.execute(
            "SELECT id, created_at FROM reports WHERE plan_id = ? ORDER BY created_at",
            (plan_id,),
        ).fetchall()
    if reps:
        print(theme.section("Reports"))
        for r in reps:
            print(
                f"  {r['id']}  {theme.dim(r['created_at'][:19])}  "
                f"{theme.dim('— read with: python run.py report ' + plan_id)}"
            )


def _pick_plan(input_fn) -> str | None:
    """Interactive plan picker: active plans first, then done ones."""
    plans = _list_plans()
    active = [p for p in plans if p["progress"][1] > 0]
    done = [p for p in plans if p["progress"][1] == 0]
    ordered = active + done  # both keep created_at DESC order
    print(theme.header("Pick a plan", f"{len(ordered)} plan(s) — active first"))
    if not ordered:
        print(theme.dim("  No plans yet."))
        return None
    for i, p in enumerate(ordered, 1):
        badge = theme.cyan("active") if p["progress"][1] > 0 else theme.green("done")
        print(f"  {i:>2}. {p['id']}  {badge}  {p['query'][:60]}")
    while True:
        try:
            line = input_fn(theme.prompt("plan> ")).strip()
        except (EOFError, KeyboardInterrupt):
            return None
        low = line.lower()
        if low in ("q", "quit", ""):
            return None
        if low.isdigit() and 1 <= int(low) <= len(ordered):
            return ordered[int(low) - 1]["id"]
        if any(p["id"] == low for p in ordered):
            return low
        print(theme.yellow("  Type a number or plan ID (q to cancel)."))


# ── Backlog ────────────────────────────────────────────────────────


def _actionable_questions(plan_id: str) -> list:
    """Pending (execution order) first, then failed — what can be acted on."""
    from db.engine import get_db

    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM questions
               WHERE plan_id = ? AND status IN ('pending', 'failed')
               ORDER BY (status != 'pending'),
                        CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1
                                      WHEN 'low' THEN 2 ELSE 3 END,
                        created_at, id""",
            (plan_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def _position_after(plan_id: str, question_id: str) -> int | None:
    """1-based position of a question in the actionable view, else None."""
    for i, q in enumerate(_actionable_questions(plan_id), 1):
        if q["id"] == question_id:
            return i
    return None


def _backlist_by_number(plan_id: str, idx: int) -> dict | None:
    qs = _actionable_questions(plan_id)
    if 1 <= idx <= len(qs):
        return qs[idx - 1]
    return None


def _backlog_command(plan_id: str, line: str) -> str:
    """Apply one backlog command. Returns 'quit', 'ok', or 'unknown'."""
    from db.knowledge import (
        add_question,
        set_question_discarded,
        set_question_pending,
        update_question_priority,
    )

    low = line.lower()
    if low in ("q", "quit"):
        return "quit"
    if low in ("", "r", "refresh"):
        return "ok"

    parts = line.split(None, 2)
    cmd = parts[0].lower()

    if cmd in ("up", "down"):
        if len(parts) < 2 or not parts[1].isdigit():
            print(theme.yellow(f"  Usage: {cmd} <n>"))
            return "unknown"
        q = _backlist_by_number(plan_id, int(parts[1]))
        if q is None:
            print(theme.yellow(f"  No question #{parts[1]} in the backlog."))
            return "unknown"
        order = ["low", "medium", "high"]
        cur = order.index(q["priority"]) if q["priority"] in order else 1
        new = order[min(2, cur + 1)] if cmd == "up" else order[max(0, cur - 1)]
        update_question_priority(q["id"], new)
        pos = _position_after(plan_id, q["id"])
        extra = f" · now #{pos} in line" if pos else ""
        print(theme.green(f"  {q['id']} → {new}{extra}"))
        return "ok"

    if cmd == "set":
        if len(parts) < 3 or parts[2].lower() not in ("high", "medium", "low"):
            print(theme.yellow("  Usage: set <n> <high|medium|low>"))
            return "unknown"
        q = _backlist_by_number(plan_id, int(parts[1]))
        if q is None:
            print(theme.yellow(f"  No question #{parts[1]} in the backlog."))
            return "unknown"
        update_question_priority(q["id"], parts[2].lower())
        pos = _position_after(plan_id, q["id"])
        extra = f" · now #{pos} in line" if pos else ""
        print(theme.green(f"  {q['id']} → {parts[2].lower()}{extra}"))
        return "ok"

    if cmd == "add":
        text = " ".join(parts[2:]).strip() if len(parts) > 2 else ""
        if not text:
            print(theme.yellow("  Usage: add <question text>"))
            return "unknown"
        qid = add_question(plan_id, text, discovered_by="user")
        pos = _position_after(plan_id, qid)
        extra = f" · now #{pos} in line" if pos else ""
        print(theme.green(f"  Added {qid}{extra}"))
        return "ok"

    if cmd == "drop":
        if len(parts) < 2 or not parts[1].isdigit():
            print(theme.yellow("  Usage: drop <n>"))
            return "unknown"
        q = _backlist_by_number(plan_id, int(parts[1]))
        if q is None:
            print(theme.yellow(f"  No question #{parts[1]} in the backlog."))
            return "unknown"
        set_question_discarded(q["id"])
        print(theme.green(f"  Dropped {q['id']} (status → discarded)"))
        return "ok"

    if cmd == "retry":
        if len(parts) < 2 or not parts[1].isdigit():
            print(theme.yellow("  Usage: retry <n>"))
            return "unknown"
        q = _backlist_by_number(plan_id, int(parts[1]))
        if q is None:
            print(theme.yellow(f"  No question #{parts[1]} in the backlog."))
            return "unknown"
        if q["status"] != "failed":
            print(theme.yellow(f"  {q['id']} is not failed (status: {q['status']})."))
            return "unknown"
        set_question_pending(q["id"])
        pos = _position_after(plan_id, q["id"])
        extra = f" · now #{pos} in line" if pos else ""
        print(theme.green(f"  Retried {q['id']} (failed → pending){extra}"))
        return "ok"

    return "unknown"


def show_backlog(plan_id: str = "", input_fn=None) -> None:
    """Review the question backlog: re-prioritize, add, drop, retry."""
    input_fn = _input_fn(input_fn)
    if not plan_id:
        plan_id = _pick_plan(input_fn) or ""
        if not plan_id:
            return

    while True:
        qs = _actionable_questions(plan_id)
        total, open_count, pct = _plan_progress(plan_id)
        print(theme.header("Backlog", plan_id))
        print(f"  {theme.progress_bar(total - open_count, total)}")
        pending = [q for q in qs if q["status"] == "pending"]
        failed = [q for q in qs if q["status"] == "failed"]
        if not qs:
            print(theme.dim("  (nothing to do — all questions resolved or discarded)"))
        for i, q in enumerate(pending, 1):
            print(
                f"  {i:>2}. {theme.priority_badge(q['priority'])} "
                f"{q['question'][:68]} {theme.dim(q['id'])}"
            )
        if failed:
            print(theme.section("Failed (retry <n> to re-process)"))
            for q in failed:
                n = len(pending) + failed.index(q) + 1
                err = theme.dim(f"({(q.get('last_error') or '')[:50]})")
                print(f"  {n:>2}. {theme.red('[FAILED]')} {q['question'][:58]} {err}")
        print(
            theme.dim(
                "  up <n> · down <n> · set <n> <high|medium|low> · add <text> · "
                "drop <n> · retry <n> · r = refresh · q = quit"
            )
        )
        try:
            line = input_fn(theme.prompt("backlog> ")).strip()
        except (EOFError, KeyboardInterrupt):
            return
        result = _backlog_command(plan_id, line)
        if result == "quit":
            return
        if result == "unknown":
            print(
                theme.yellow(
                    "  Commands: up <n> · down <n> · set <n> <high|medium|low> · "
                    "add <text> · drop <n> · retry <n> · r · q"
                )
            )


def show_topics(plan_id: str = "", input_fn=None) -> None:
    """Print a topic map of a plan's questions (clustered view).

    One-shot, read-only: clusters the non-discarded questions
    (cluster_plan_questions) and shows each topic with its questions,
    status, and the findings attached to them.
    """
    from db.knowledge import cluster_plan_questions, get_plan_findings

    if not plan_id:
        plan_id = _pick_plan(input_fn) or ""
        if not plan_id:
            return

    clusters = cluster_plan_questions(plan_id)
    findings_by_q: dict = {}
    for f in get_plan_findings(plan_id):
        findings_by_q[f["question_id"]] = findings_by_q.get(f["question_id"], 0) + 1

    total = sum(len(c["members"]) for c in clusters)
    print(theme.header("Topic Map", plan_id))
    if not clusters:
        print(theme.dim("  (no questions yet)"))
        return
    print(theme.dim(f"  {len(clusters)} topic(s) across {total} question(s)"))
    for i, c in enumerate(clusters, 1):
        counts = ", ".join(f"{v} {k}" for k, v in sorted(c["counts"].items()))
        n_findings = sum(findings_by_q.get(m["id"], 0) for m in c["members"])
        print()
        print(theme.section(f"Topic {i}: {c['label'][:80]}"))
        print(theme.dim(f"  {counts} · {n_findings} finding(s)"))
        for m in c["members"]:
            mark = theme.green("[x]") if m["status"] == "resolved" else theme.dim("[ ]")
            print(
                f"  {mark} {theme.priority_badge(m['priority'])} {m['question'][:76]}"
            )


# ── Results ────────────────────────────────────────────────────────


def _plan_results(plan_id: str) -> tuple:
    """(findings, valid_ids, knowledge_entries) for a plan."""
    from db.knowledge import get_plan_entries, get_plan_findings
    from utils.quality import filter_quality

    findings = get_plan_findings(plan_id)
    valid, _discarded = filter_quality(findings)
    valid_ids = {f["id"] for f in valid}
    entries = get_plan_entries(plan_id)
    return findings, valid_ids, entries


def _print_finding_detail(f: dict, valid_ids: set) -> None:
    """Print one finding: question, summary, facts, tags, source."""
    ok = f["id"] in valid_ids
    if ok:
        print(theme.green("  ── Finding (valid)"))
    else:
        print(theme.dim("  ── Finding (discarded as generic filler)"))
    print(f"  ID:       {f['id']}")
    print(f"  Question: {(f.get('source_question') or '')[:100]}")
    print(f"  Summary:  {f.get('summary', '')}")
    facts = f.get("facts") or []
    if isinstance(facts, list) and facts:
        print("  Facts:")
        for fact in facts:
            if isinstance(fact, dict):
                conf = theme.dim(f"({fact.get('confidence', '?')})")
                print(f"    - {fact.get('claim', '')} {conf}")
            else:
                print(f"    - {fact}")
    tags = f.get("tags") or []
    if isinstance(tags, list) and tags:
        print(f"  Tags:     {', '.join(tags)}")
    if f.get("source_title"):
        print(f"  Source:   {f['source_title']}")
    print()


def show_results(plan_id: str = "", input_fn=None) -> None:
    """Browse a plan's findings; type a number for the full detail."""
    input_fn = _input_fn(input_fn)
    if not plan_id:
        plan_id = _pick_plan(input_fn) or ""
        if not plan_id:
            return

    while True:
        findings, valid_ids, entries = _plan_results(plan_id)
        print(theme.header("Results", plan_id))
        if not findings and not entries:
            print(theme.dim("  No findings or knowledge entries for this plan yet."))
        else:
            n_valid = len(valid_ids)
            print(
                f"  {n_valid} of {len(findings)} finding(s) valid · "
                f"{len(entries)} knowledge entr{'y' if len(entries) == 1 else 'ies'}"
            )
            for i, f in enumerate(findings, 1):
                mark = (
                    theme.green("OK")
                    if f["id"] in valid_ids
                    else theme.dim("discarded")
                )
                print(f"  {i:>2}. {mark:<12} {theme.dim(f['id'])}")
                print(f"      {f.get('summary', '')[:72]}")
        print(theme.dim("  Type a finding number for details · r = refresh · q = quit"))
        try:
            line = input_fn(theme.prompt("results> ")).strip()
        except (EOFError, KeyboardInterrupt):
            return
        low = line.lower()
        if low in ("q", "quit"):
            return
        if low in ("", "r", "refresh"):
            continue
        if low.isdigit() and 1 <= int(low) <= len(findings):
            _print_finding_detail(findings[int(low) - 1], valid_ids)
            continue
        print(theme.yellow("  Type a finding number, r, or q."))


# ── Reports ────────────────────────────────────────────────────────


def _list_reports(plan_id: str) -> list:
    from db.engine import get_db

    with get_db() as conn:
        if plan_id:
            rows = conn.execute(
                "SELECT id, plan_id, query, report, created_at FROM reports "
                "WHERE plan_id = ? ORDER BY created_at DESC, id DESC",
                (plan_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, plan_id, query, report, created_at FROM reports "
                "ORDER BY created_at DESC, id DESC"
            ).fetchall()
    return [dict(r) for r in rows]


def show_reports(plan_id: str = "", input_fn=None) -> None:
    """List a plan's saved reports; type a number to read one in full."""
    input_fn = _input_fn(input_fn)
    if not plan_id:
        plan_id = _pick_plan(input_fn) or ""
        if not plan_id:
            return

    while True:
        reports = _list_reports(plan_id)
        print(theme.header("Reports", plan_id))
        if not reports:
            print(theme.dim("  No reports saved for this plan yet."))
        for i, r in enumerate(reports, 1):
            print(
                f"  {i:>2}. {r['id']}  {theme.dim(r['created_at'][:19])}  "
                f"{r['query'][:50]}"
            )
        print(theme.dim("  Type a report number to read it · r = refresh · q = quit"))
        try:
            line = input_fn(theme.prompt("report> ")).strip()
        except (EOFError, KeyboardInterrupt):
            return
        low = line.lower()
        if low in ("q", "quit"):
            return
        if low in ("", "r", "refresh"):
            continue
        if low.isdigit() and 1 <= int(low) <= len(reports):
            print()
            print(reports[int(low) - 1]["report"])
            print()
            continue
        print(theme.yellow("  Type a report number, r, or q."))


# ── Stats ──────────────────────────────────────────────────────────


def render_stats(stats: dict) -> None:
    """Pretty-print a get_stats() dict (used by the review 'stats' command)."""
    print(theme.section("Knowledge base"))
    by_status = stats.get("by_status") or {}
    print(f"  Entries: {stats.get('total', 0)} total")
    for st in ("curated", "raw", "discarded"):
        if by_status.get(st):
            print(f"    {st}: {by_status[st]}")
    top = stats.get("top_tags") or []
    if top:
        print("  Top tags: " + ", ".join(f"{t}×{n}" for t, n in top[:10]))


def show_stats() -> None:
    """Knowledge-base stats plus a per-plan summary (non-interactive)."""
    from db.knowledge import get_stats

    stats = get_stats()
    plans = _list_plans()
    print(theme.header("Knowledge base", "stats"))
    render_stats(stats)
    print(theme.section("Plans"))
    if not plans:
        print(theme.dim("  (no plans yet)"))
    for p in plans:
        total, open_count, pct = p["progress"]
        state = "done" if open_count == 0 else f"{open_count} open"
        print(
            f"  {p['id']}  {theme.dim(state + ' · ' + str(pct) + '% closed')}  "
            f"{p['query'][:50]}"
        )


# ── Models ─────────────────────────────────────────────────────────


# (step, what it does, call profile / model guidance)
_MODEL_STEPS: tuple[tuple[str, str, str], ...] = (
    (
        "planning",
        "turns the query into a question bank (and re-prioritizes it)",
        "a few calls; a strong model pays off here",
    ),
    (
        "extraction",
        "parses each search result into structured findings",
        "one call per question — highest volume; a fast model is usually fine",
    ),
    (
        "curation",
        "ready-check, dedup and contradiction analysis at reviews",
        "moderate volume; a fast model is usually fine",
    ),
    (
        "report",
        "writes the final report (two passes, the longest output)",
        "1–2 calls; this is where a strong model shows most",
    ),
)


_STEP_NAMES: frozenset[str] = frozenset(step for step, _d, _p in _MODEL_STEPS)


def _models_table() -> list[str]:
    """One line per step: resolved provider/model, overrides marked."""
    from config.settings import config, resolve_step_model

    default = f"{config.provider}/{getattr(config, f'{config.provider}_model', config.local_model)}"
    lines = [
        f"    {theme.dim('default:')}   {default}  {theme.dim('(PROVIDER + <provider>_MODEL)')}"
    ]
    for step, _desc, _profile in _MODEL_STEPS:
        try:
            provider, model = resolve_step_model(step)
        except ValueError as e:
            lines.append(f"    {step + ':':<12} {theme.red('config error: ' + str(e))}")
            continue
        resolved = f"{provider}/{model}"
        mark = theme.cyan("  ← override") if resolved != default else ""
        lines.append(f"    {step + ':':<12} {resolved}{mark}")
    return lines


def _models_help() -> None:
    print(
        "  "
        + theme.cyan("set <step> [provider/model]")
        + "  change a step, e.g. "
        + theme.dim("set report anthropic/claude-sonnet-5-5")
    )
    print(
        "  "
        + theme.cyan("unset <step>")
        + "  remove a step's overrides (back to the default)"
    )
    print("  " + theme.cyan("env") + "  show the .env lines for the current setup")
    print("  " + theme.cyan("q") + "  quit")


def _models_set(input_fn, line: str, env_path: str) -> bool:
    """Handle 'set <step> [provider/model]' — True on success.

    A provider equal to the global PROVIDER is not written as an override:
    the ``*_PROVIDER`` line stays empty so the step keeps following the
    global provider if it changes later.
    """
    from config.settings import (
        _VALID_PROVIDERS,
        config,
        resolve_step_model,
        update_config,
        workflow_config,
    )
    from utils.env_file import upsert_env_file

    parts = line.split()
    if len(parts) < 2 or parts[1] not in _STEP_NAMES:
        print(
            theme.yellow(
                "  Usage: set <step> [provider/model] — one of: "
                + ", ".join(s for s, _, _ in _MODEL_STEPS)
            )
        )
        return False
    step = parts[1]

    # Current effective values are the prompt defaults.
    try:
        cur_provider, cur_model = resolve_step_model(step)
    except ValueError:
        cur_provider, cur_model = "", ""

    if len(parts) >= 3:
        arg = " ".join(parts[2:])
        if "/" in arg:
            provider, model = arg.split("/", 1)
        else:  # bare name = model only, provider override untouched
            provider = (
                getattr(workflow_config, f"{step}_provider", "") or config.provider
            )
            model = arg
    else:
        provider = (
            input_fn(theme.prompt(f"  Provider [{cur_provider}]: ")).strip()
            or cur_provider
        )
        model = input_fn(theme.prompt(f"  Model [{cur_model}]: ")).strip() or cur_model

    provider = provider.strip()
    model = model.strip()
    if provider not in _VALID_PROVIDERS:
        print(
            theme.yellow(
                f"  Unknown provider {provider!r} — one of: "
                + ", ".join(_VALID_PROVIDERS)
            )
        )
        return False
    if not model:
        print(
            theme.yellow(
                "  A model name is required (empty = the provider's default — use unset to go back to the default)."
            )
        )
        return False

    provider_override = "" if provider == config.provider else provider
    update_config(**{f"{step}_provider": provider_override, f"{step}_model": model})
    upsert_env_file(env_path, f"{step.upper()}_PROVIDER", provider_override)
    upsert_env_file(env_path, f"{step.upper()}_MODEL", model)
    _, effective_model = resolve_step_model(step)
    print(theme.green(f"  ✓ {step} → {provider}/{effective_model}  (saved to .env)"))
    return True


def _models_unset(line: str, env_path: str) -> bool:
    """Handle 'unset <step>' — clear that step's overrides."""
    from config.settings import update_config
    from utils.env_file import upsert_env_file

    parts = line.split()
    if len(parts) < 2 or parts[1] not in _STEP_NAMES:
        print(
            theme.yellow(
                "  Usage: unset <step> — one of: "
                + ", ".join(s for s, _, _ in _MODEL_STEPS)
            )
        )
        return False
    step = parts[1]
    update_config(**{f"{step}_provider": "", f"{step}_model": ""})
    for key in (f"{step.upper()}_PROVIDER", f"{step.upper()}_MODEL"):
        upsert_env_file(env_path, key, "")
    print(theme.green(f"  ✓ {step} → default  (overrides cleared in .env)"))
    return True


def show_models(input_fn=None, env_path: str | None = None) -> None:
    """Browse and change the model (and provider) used by each pipeline step.

    Changes are applied to the running config and saved to the .env file,
    so the next session — and the session-start banner — pick them up.
    """
    from config.settings import workflow_config
    from utils.env_file import default_env_path

    input_fn = _input_fn(input_fn)
    if env_path is None:
        env_path = default_env_path()

    while True:
        print(theme.header("Models per step", "what runs where — and how to change it"))
        for line in _models_table():
            print(line)
        print()
        print("  " + theme.dim("What each step does (and what to pick):"))
        for step, desc, profile in _MODEL_STEPS:
            print(f"    {step + ':':<12} {desc}")
            print(f"    {'':<12} {theme.dim(profile)}")
        print()
        _models_help()
        try:
            line = input_fn(theme.prompt("models> ")).strip()
        except (EOFError, KeyboardInterrupt):
            return
        low = line.lower()
        if low in ("q", "quit", ""):
            return
        if low in ("?", "help"):
            continue
        if low == "env":
            for step, _d, _p in _MODEL_STEPS:
                for suffix in ("PROVIDER", "MODEL"):
                    val = getattr(workflow_config, f"{step}_{suffix.lower()}")
                    print(f"  {step.upper()}_{suffix}={val}")
            print("  " + theme.dim(f"(in {env_path})"))
            continue
        if low.startswith("set "):
            if _models_set(input_fn, line, env_path):
                continue
        elif low.startswith("unset "):
            if _models_unset(line, env_path):
                continue
        else:
            print(theme.yellow(f"  Unknown command {line!r} — type ? for help."))
