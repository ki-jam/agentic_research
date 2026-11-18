"""
The human-review screen.

Renders the checkpointed review payload (built by prepare_review_node) and
reads the curator's response. The payload carries the analysis text (from
nodes/curation._build_interim_prompt); this module renders it plus the
screen chrome: open questions, recommendations, raw entries, and the
numbered command menu.

Input contract (parse_review_input):
    1..6        → continue / stop / stats / findings / contradictions / rewind
    <blank>     → continue
    ? / help    → show the command help (then re-prompt)
    <full cmd>  → passed through to curate_knowledge_node
                  (curate <id>, discard <id>, tag <id> <t>, retry <id>,
                   accept <id>, gap <n>, continue, stop, ...)
    anything else → None (re-prompt — a typo never silently continues)
"""

from __future__ import annotations

from . import theme

# ── Screen rendering ───────────────────────────────────────────────


def render_review(payload: dict) -> None:
    """Render a review/interrupt payload for the user."""
    if not isinstance(payload, dict):
        payload = {"prompt": str(payload)}

    message = payload.get("message") or "Review"
    print(theme.header("Review", message))

    stats = payload.get("stats") or {}
    qp = payload.get("question_progress") or {}
    by_status = stats.get("by_status") or {}
    if stats.get("total") is not None:
        parts = [f"library {stats['total']} entries"]
        for st in ("curated", "raw", "discarded"):
            if by_status.get(st):
                parts.append(f"{by_status[st]} {st}")
        print("  " + theme.cyan("Library: ") + ", ".join(parts))
    if qp:
        print(
            "  "
            + theme.cyan("Questions: ")
            + ", ".join(f"{v} {k}" for k, v in qp.items())
        )

    open_q = payload.get("open_questions") or []
    if open_q:
        print(theme.section("Open questions (pending first, then failed)"))
        for i, q in enumerate(open_q, 1):
            if q.get("status") == "failed":
                err = (q.get("last_error") or "")[:50]
                print(
                    f"  {i:>2}. {theme.red('[FAILED]')} "
                    f"{q.get('question', '')[:60]}  {theme.dim('(' + err + ')')}"
                )
            else:
                print(
                    f"  {i:>2}. {theme.priority_badge(q.get('priority'))} "
                    f"{q.get('question', '')[:66]}"
                )

    n_contra = len(payload.get("contradictions") or [])
    if n_contra:
        print("  " + theme.yellow(f"Contradictions: {n_contra} pair(s) flagged"))

    recs = payload.get("recommendations") or []
    if recs:
        print(theme.section("Curation recommendations"))
        for r in recs[:10]:
            reason = (
                f"  {theme.dim(r.get('reason', '')[:80])}" if r.get("reason") else ""
            )
            print(
                f"  {theme.status_badge(r.get('action', 'keep'), width=6)} "
                f"{r.get('id', '?')}{reason}"
            )

    raws = payload.get("raw_entries") or []
    if raws:
        print(theme.section("Raw entries (curate <id> / discard <id> / tag <id> <t>)"))
        for e in raws[:10]:
            tags = (
                f" {theme.dim('(' + ', '.join(e.get('tags', [])) + ')')}"
                if e.get("tags")
                else ""
            )
            print(f"  {e.get('id', '?')}: {e.get('title', 'Untitled')[:60]}{tags}")

    if payload.get("prompt"):
        print()
        print(payload["prompt"])

    print()
    print_review_help()


def print_review_help() -> None:
    """Print the numbered command menu (1–6) and the argument commands."""
    print(theme.section("Actions (type a number, a full command, or ? for help)"))
    rows = [
        ("1", "continue", "skip curation, keep researching (blank line does this too)"),
        ("2", "stop", "end research, write the final report"),
        ("3", "stats", "full knowledge-base stats"),
        ("4", "findings", "show all extracted findings (incl. discarded)"),
        ("5", "contradictions", "show flagged contradiction pairs"),
        ("6", "rewind", "force one more backlog pass"),
        ("", "curate <id>", "promote a raw entry to the library"),
        ("", "discard <id>", "remove a raw entry"),
        ("", "tag <id> <t>", "add topic tags to an entry"),
        ("", "retry <id>", "reset a failed question to pending"),
        ("", "accept <id>", "resolve a failed question with degraded evidence"),
        ("", "gap <n>", "add gap #n from the interim analysis as a question"),
    ]
    for num, cmd, desc in rows:
        label = f"{num}) " if num else "    "
        print(f"  {theme.bold(label)} {theme.cyan(f'{cmd:<16}')} {theme.dim(desc)}")


# ── Input parsing ──────────────────────────────────────────────────

_NUMBER_SHORTCUTS = {
    "1": "continue",
    "2": "stop",
    "3": "stats",
    "4": "findings",
    "5": "contradictions",
    "6": "rewind",
}

# Commands that need an argument — a bare word is an incomplete command,
# not a silent continue.
_ARG_COMMANDS = frozenset({"curate", "discard", "tag", "retry", "accept", "gap"})

_NO_ARG_COMMANDS = frozenset(set(_NUMBER_SHORTCUTS.values()) | {"help"})


def parse_review_input(line: str) -> str | None:
    """Map a raw input line to a graph command.

    Returns the command string, or None when the line is unrecognized
    (caller should re-prompt).
    """
    s = (line or "").strip()
    if not s:
        return "continue"
    if s in _NUMBER_SHORTCUTS:
        return _NUMBER_SHORTCUTS[s]
    first, _, rest = s.partition(" ")
    first = first.lower()
    if first in ("?", "help"):
        return "help"
    if first in _ARG_COMMANDS:
        return s if rest.strip() else None
    if first in _NO_ARG_COMMANDS:
        return s
    return None


def ask_review_response(payload: dict, input_fn=None) -> str:
    """Render the review screen and read a valid command.

    Re-prompts on unknown input (a typo must never silently continue the
    graph); "help" prints the menu and re-prompts. ``input_fn`` defaults
    to builtins.input, resolved at call time so test patches are seen.
    EOFError/KeyboardInterrupt propagate to the caller.
    """
    if input_fn is None:
        import builtins

        input_fn = builtins.input
    render_review(payload)
    while True:
        line = input_fn(theme.prompt("Your command> "))
        command = parse_review_input(line)
        if command is None:
            print(
                theme.yellow(f"  Unknown command: {line.strip()!r} — type ? for help.")
            )
            continue
        if command == "help":
            print_review_help()
            continue
        return command
