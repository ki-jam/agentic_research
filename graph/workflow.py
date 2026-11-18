"""
Workflow assembly — wire all nodes into a LangGraph StateGraph.

Flow:
  START → planning → backlog → search → extraction → consolidation
  consolidation → {backlog (loop), prepare_review, write_findings}
  prepare_review → human_review → curation → consolidation (loop)
  write_findings → END

Persistence:
  build_app() compiles with a SQLite checkpointer by default, so sessions
  survive process restarts and can be resumed (run.py --resume).
"""

from __future__ import annotations

from typing import Optional

from langgraph.graph import StateGraph, START, END

from state.research_state import ResearchState
from nodes.planning import make_plan_node
from nodes.backlog import manage_backlog_node
from nodes.search import search_execution_node
from nodes.extraction import extract_and_verify_node
from nodes.consolidation import consolidation_node
from nodes.curation import (
    prepare_review_node,
    human_review_interrupt,
    curate_knowledge_node,
)
from utils.cost import CostTracker, get_active_tracker, set_active_tracker
from utils.llm import invoke_with_cost


# ── Report helpers ────────────────────────────────────────────────

# Pass-1 findings cap for the two-pass report. The local 7B model cannot
# finish a report synthesized from a large findings dump (both attempts
# failed live on 87 findings while 25-76 synthesized), so pass 1 works
# from at most this many findings and pass 2 merges in the rest.
REPORT_PASS1_FINDINGS = 25


def build_question_section(plan_id: str, query: str) -> str:
    """Deterministic 'Research Question' section for the final report.

    Inserted by code rather than left to the LLM: the query verbatim plus
    every sub-question with its status. This guarantees the question is in
    the report and makes partial coverage (unanswered sub-questions) visible
    up front instead of buried in a 'needs more research' section.
    """
    from db.engine import get_db

    with get_db() as conn:
        qrows = conn.execute(
            "SELECT question, status FROM questions WHERE plan_id = ? "
            "ORDER BY created_at, id",
            (plan_id,),
        ).fetchall()
    if not qrows:
        return ""
    resolved = sum(1 for r in qrows if r[1] == "resolved")
    lines = ["## Research Question", ""]
    lines.append(f"**Original query:** {query}")
    lines.append("")
    lines.append(f"**Sub-questions ({resolved} of {len(qrows)} resolved):**")
    lines.append("")
    for row in qrows:
        mark = "x" if row[1] == "resolved" else " "
        lines.append(f"- [{mark}] {row[0]} — {row[1]}")
    return "\n".join(lines) + "\n"


def _insert_after_first_heading(text: str, section: str) -> str:
    """Insert a markdown section right after the first heading line."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.strip():
            if line.lstrip().startswith("#"):
                lines.insert(i + 1, "\n" + section)
                return "\n".join(lines)
            break  # first non-empty line is not a heading
    return section + "\n" + text  # no heading at all: prepend


def conn_qcounts(plan_id: str) -> tuple:
    """(total, open) question counts for a plan.

    "Open" = non-terminal (pending/in_progress/failed) — same semantics
    as the consolidation coverage floor, so the banner, the stats line,
    and the gate agree on what counts as unresolved.
    """
    from db.engine import get_db

    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*), "
            "SUM(CASE WHEN status NOT IN ('resolved', 'discarded') THEN 1 ELSE 0 END) "
            "FROM questions WHERE plan_id = ?",
            (plan_id,),
        ).fetchone()
    return (row[0] or 0, row[1] or 0)


def build_followup_section(plan_id: str) -> str:
    """Deterministic 'Suggested Follow-Up Questions' backstop for the report.

    Lists the plan's open (non-terminal) questions as suggested follow-ups,
    mirroring build_question_section. Returns "" when nothing is open — the
    LLM's own section (required by REPORT_SYSTEM) then stands alone, and
    code must not fabricate questions.
    """
    from db.engine import get_db

    with get_db() as conn:
        rows = conn.execute(
            "SELECT question FROM questions WHERE plan_id = ? "
            "AND status NOT IN ('resolved', 'discarded') "
            "ORDER BY created_at, id",
            (plan_id,),
        ).fetchall()
    if not rows:
        return ""
    lines = ["## Suggested Follow-Up Questions", ""]
    lines.append("Open questions from this session that still need research:")
    lines.append("")
    for row in rows:
        lines.append(f"- {row[0]}")
    return "\n".join(lines) + "\n"


# ── Node: Write Final Report ──────────────────────────────────────


def _write_findings_node(state) -> dict:
    """Compile final report from THIS PLAN's DB findings and curated knowledge."""
    from config.prompts import REPORT_PASS2, REPORT_SYSTEM, REPORT_USER
    from config.settings import config
    from db.engine import get_db
    from db.knowledge import (
        get_plan,
        get_plan_entries,
        get_plan_findings,
        get_plan_stats,
        get_source_coverage,
        save_report,
    )
    from utils.quality import filter_quality

    plan_id = state.get("plan_ref", "")
    query = state.get("original_query", "")

    # Plan-scoped data (the global versions leak findings from other plans)
    findings = get_plan_findings(plan_id) if plan_id else []
    curated = get_plan_entries(plan_id, status="curated") if plan_id else []
    stats = get_plan_stats(plan_id) if plan_id else {}
    plan = get_plan(plan_id) if plan_id else None

    # ── Quality filter for stats ────────────────────────────────
    valid_findings, discarded = filter_quality(findings)

    # ── Collect unique sources for references (plan-scoped) ─────
    # Also batch-fetch engines for citation vs. search classification.
    # Replaces the old N+1 loop: one SELECT engine WHERE id=? per finding.
    sources = set()
    engine_map: dict[int, str] = {}
    raw_id_to_url: dict[int, str] = {}
    with get_db() as conn:
        rows = conn.execute(
            "SELECT DISTINCT source_url, title FROM raw_results WHERE plan_id = ?",
            (plan_id,),
        ).fetchall()
        for r in rows:
            d = dict(r)
            url = d.get("source_url", "")
            title = d.get("title", "")
            if url:
                sources.add((url, title))

        # Batch engine + URL lookup (N+1 → 1 query). The URL per raw result
        # lets each finding show its canonical reference number below.
        raw_ids = [f["raw_result_id"] for f in findings if f.get("raw_result_id")]
        if raw_ids:
            placeholders = ",".join("?" for _ in raw_ids)
            for row in conn.execute(
                f"SELECT id, engine, source_url FROM raw_results WHERE id IN ({placeholders})",
                raw_ids,
            ):
                engine_map[row[0]] = row[1]
                if row[2]:
                    raw_id_to_url[row[0]] = row[2]

    # Also add sources from curated knowledge entries
    for entry in curated:
        url = entry.get("source_url", "")
        title = entry.get("title", "")
        if url and (url, title) not in sources:
            sources.add((url, title))

    # Sort sources for consistent ordering
    sources_list = sorted(sources, key=lambda x: x[0])

    # raw_result_id → canonical reference number, so every finding block
    # shows the exact [n] it came from. Without this the report LLM has to
    # guess which reference belongs to which finding — the observed cause
    # of misattributed citations (claims cited to an unrelated source).
    url_to_refnum = {url: i for i, (url, _t) in enumerate(sources_list, 1)}
    raw_id_to_refnum = {rid: url_to_refnum[url] for rid, url in raw_id_to_url.items()}

    # Build references section
    references_text = ""
    if sources_list:
        references_lines = []
        for i, (url, title) in enumerate(sources_list, 1):
            references_lines.append(f"[{i}] {title} — {url}")
        references_text = "\n\n## References\n\n" + "\n".join(references_lines)

    # Build curated entries text
    curated_text = ""
    for entry in curated:
        curated_text += f"### {entry.get('title', 'Untitled')}\n"
        curated_text += f"Tags: {', '.join(entry.get('topic_tags', []))}\n"
        curated_text += f"Summary: {entry.get('summary', '')}\n"
        entry_ref = url_to_refnum.get(entry.get("source_url", ""))
        ref_note = f"[{entry_ref}] " if entry_ref else ""
        curated_text += f"Source: {ref_note}{entry.get('source_url', 'N/A')}\n\n"

    # ── Findings text builder (shared by both report passes) ───────
    def _findings_block(subset: list) -> str:
        """Render findings in the report format: search findings first,
        then the prominently-labeled citation-discovery section."""

        def _is_citation(f) -> bool:
            rid = f.get("raw_result_id")
            return bool(rid and engine_map.get(rid, "").startswith("exa-answer"))

        def _finding_lines(f) -> str:
            out = f"### Finding: {f.get('summary', 'Untitled')}\n"
            out += f"Q: {f.get('source_question', 'N/A')}\n"
            facts = f.get("facts", [])
            if isinstance(facts, list):
                for fact in facts:
                    if isinstance(fact, dict):
                        out += f"- {fact.get('claim', '')} ({fact.get('confidence', '?')})\n"
                    else:
                        out += f"- {fact}\n"
            tags = f.get("tags", [])
            if isinstance(tags, list):
                out += f"Tags: {', '.join(tags)}\n"
            refnum = raw_id_to_refnum.get(f.get("raw_result_id"))
            if refnum:
                out += f"Source: [{refnum}] — cite this number for claims from this finding\n"
            return out + "\n"

        text = "".join(_finding_lines(f) for f in subset if not _is_citation(f))
        cites = [f for f in subset if _is_citation(f)]
        if cites:
            text += "### 📚 FOUNDATIONAL PAPERS (Citation Discovery)\n"
            text += "These findings come from the Exa answer API, which identified foundational/authoritative papers on the topic.\n\n"
            text += "".join(_finding_lines(f) for f in cites)
        return text

    # Full text — the deterministic fallback embeds EVERYTHING, uncut.
    findings_text = _findings_block(findings) + curated_text

    # ── Two-pass split ─────────────────────────────────────────────
    # Pass 1 synthesizes from a capped, prioritized set; pass 2 merges
    # the remaining findings into the pass-1 report. Selection order:
    # citation-discovery findings first (the report prompt treats them
    # as high-priority sources), then question priority (high → medium
    # → low); ties keep DB order (newest first).
    priority_of: dict[str, int] = {}
    with get_db() as conn:
        for row in conn.execute(
            "SELECT id, priority FROM questions WHERE plan_id = ?", (plan_id,)
        ):
            priority_of[row[0]] = {"high": 0, "medium": 1, "low": 2}.get(row[1], 1)

    def _rank(f):
        rid = f.get("raw_result_id")
        is_cite = bool(rid and engine_map.get(rid, "").startswith("exa-answer"))
        return (0 if is_cite else 1, priority_of.get(f.get("question_id"), 1))

    ranked = sorted(findings, key=_rank)
    pass1_findings = ranked[:REPORT_PASS1_FINDINGS]
    pass2_findings = ranked[REPORT_PASS1_FINDINGS:]
    pass1_findings_text = _findings_block(pass1_findings) + curated_text
    pass2_findings_text = _findings_block(pass2_findings)

    questions_text = ""
    if plan:
        try:
            import json

            raw_questions = plan.get("questions", [])
            if isinstance(raw_questions, str):
                raw_questions = json.loads(raw_questions)
            questions_text = "\n".join(
                f"- {q.get('question', '')}" for q in raw_questions
            )
        except (json.JSONDecodeError, AttributeError):
            pass

    # ── LLM report synthesis (two passes) ───────────────────────────
    # The local 7B model fails on the full findings text at large plans
    # (3 of 6 live sessions: OpenAIAPIError / too-short body, correlated
    # with prompt size — see RESEARCH_LOOP_IMPROVEMENTS.md §6). So:
    #   pass 1 synthesizes from the capped, prioritized set
    #   pass 2 merges the remaining findings into the pass-1 report
    # Each pass retries once (one retry recovers stochastic failures).
    # Pass-1 double failure → deterministic fallback below (unchanged).
    # Pass-2 double failure → keep the pass-1 report + a note: a report
    # covering the top-N findings beats no report at all.
    report_text = ""
    report_passes = 0
    pass2_note = ""
    last_exc: Exception | None = None

    def _call_report_llm(system: str, user: str) -> str:
        # The report is the longest LLM call in the pipeline. Thinking
        # models can spend the entire default token budget on reasoning
        # and return an empty answer (observed live on plan_d3df337a),
        # so it gets its own larger output budget.
        model = config.get_chat_model(task="report", max_tokens=32768)
        response = invoke_with_cost(model, f"{system}\n\n{user}", task="report")
        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        # Empty or whitespace-only content does not raise — treat it as a
        # failure so the fallback engages instead of persisting a bodyless
        # report silently.
        if len(content.strip()) < 200:
            raise ValueError(
                f"LLM returned an empty or too-short report body "
                f"({len(content.strip())} chars after stripping)"
            )
        return content

    for attempt in range(1, 3):
        try:
            print(
                f"  [REPORT] Pass 1: synthesizing from {len(pass1_findings)} of "
                f"{len(findings)} findings…"
            )
            report_text = _call_report_llm(
                REPORT_SYSTEM,
                REPORT_USER.format(
                    query=query,
                    plan=questions_text,
                    findings=pass1_findings_text,
                    references=references_text,
                ),
            )
            report_passes = 1
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            # Log the actual error (one line, capped): session 6's
            # OpenAIAPIError left no trace of what failed — class name
            # alone is not diagnosable.
            detail = " ".join(str(exc).split())[:200]
            if attempt == 1:
                print(
                    f"  [REPORT] Pass 1 attempt 1 failed "
                    f"({type(exc).__name__}: {detail}); retrying once"
                )

    if last_exc is None and pass2_findings:
        # Pass 2: merge the remaining findings into the pass-1 report.
        pass2_exc: Exception | None = None
        for attempt in range(1, 3):
            try:
                print(
                    f"  [REPORT] Pass 2: merging in {len(pass2_findings)} "
                    "remaining findings…"
                )
                report_text = _call_report_llm(
                    REPORT_SYSTEM,
                    REPORT_PASS2.format(
                        report=report_text,
                        findings=pass2_findings_text,
                        references=references_text,
                    ),
                )
                report_passes = 2
                pass2_exc = None
                break
            except Exception as exc:
                pass2_exc = exc
                detail = " ".join(str(exc).split())[:200]
                if attempt == 1:
                    print(
                        f"  [REPORT] Pass 2 attempt 1 failed "
                        f"({type(exc).__name__}: {detail}); retrying once"
                    )
        if pass2_exc is not None:
            # Keep the pass-1 report; flag what it is missing.
            pass2_note = (
                f"> **Note:** Report synthesized in two passes; the second pass "
                f"(merging the remaining {len(pass2_findings)} findings) failed "
                f"after 2 attempts ({type(pass2_exc).__name__}), so this report "
                f"covers the {len(pass1_findings)} highest-priority findings only.\n"
            )

    if last_exc is not None:
        # Fallback: simple text report — deterministic, always complete
        report_text = f"# Research Report: {query}\n\n"
        report_text += (
            f"> **Note:** LLM report generation failed or returned no usable "
            f"content ({type(last_exc).__name__}) after 2 attempts. This is "
            f"the deterministic fallback: all extracted findings, "
            f"unsynthesized.\n\n"
        )
        report_text += "## Summary\n"
        report_text += f"- Total questions: {stats.get('total', 0)}\n"
        report_text += f"- Findings extracted: {len(findings)}\n"
        report_text += f"- Curated entries: {len(curated)}\n\n"
        if findings_text:
            report_text += "## Extracted Findings\n\n"
            report_text += findings_text
        if curated_text:
            report_text += "## Curated Knowledge Base\n\n"
            report_text += curated_text
        if references_text:
            report_text += references_text

    # ── Evidence Matrix (per-source coverage) ────────────────────
    # Inspired by deep-researcher-node: a machine-readable control surface
    # that shows which sources are well-covered vs under-represented.
    evidence_matrix_text = ""
    if plan_id:
        coverage = get_source_coverage(plan_id)
        if coverage:
            # Determine min sources threshold for highlighting
            min_sources = 2
            matrix_lines = ["## Evidence Matrix\n\n"]
            matrix_lines.append("| Source | Facts | Status |")
            matrix_lines.append("|--------|-------|--------|")
            for entry in coverage:
                status = (
                    "✅"
                    if entry["fact_count"] >= min_sources
                    else "⚠️ under-represented"
                )
                # Truncate URL for display
                short_url = (
                    entry["source_url"][:60] + "..."
                    if len(entry["source_url"]) > 60
                    else entry["source_url"]
                )
                title = (
                    entry["source_title"][:40]
                    if entry["source_title"]
                    else "(no title)"
                )
                matrix_lines.append(
                    f"| [{title}]({short_url}) | {entry['fact_count']} | {status} |"
                )
            matrix_lines.append("")
            evidence_matrix_text = "\n".join(matrix_lines)

    # ── Guarantee the research question appears in the report ────
    # The LLM rephrases the query into a title and may never restate it,
    # so insert it deterministically (verbatim query + sub-question
    # status) — this is also what makes partial coverage visible.
    total_questions, open_questions = (0, 0)
    if plan_id:
        # Question-bank counts for the interim banner + stats line.
        total_questions, open_questions = conn_qcounts(plan_id)

        question_section = build_question_section(plan_id, query)
        if question_section:
            report_text = _insert_after_first_heading(report_text, question_section)

        # Pass-2 failure note (lands directly under the title once the
        # banner inserts above it: title → banner → note → question section).
        if pass2_note:
            report_text = _insert_after_first_heading(report_text, pass2_note)

        # Interim banner: when questions are still open, the report is a
        # snapshot, not a settled answer — say so at the top. Inserted after
        # the question section so it lands directly under the title.
        if open_questions:
            banner = (
                f"> **Interim report** — {open_questions} of {total_questions} "
                "sub-questions are still open. This report reflects what is "
                "known so far; the open questions are listed below.\n"
            )
            report_text = _insert_after_first_heading(report_text, banner)

        # ── Suggested Follow-Up Questions (deterministic backstop) ────
        # REPORT_SYSTEM requires the LLM to write this section. If it
        # omitted one and the bank still has open questions, append them so
        # structured follow-ups are never silently lost. The code only lists
        # real bank rows — it never invents questions.
        if open_questions and "follow-up questions" not in report_text.lower():
            followup = build_followup_section(plan_id)
            if followup:
                report_text = report_text.rstrip() + "\n\n" + followup

    # ── Citation validation & repair ─────────────────────────────
    # The LLM may write its own References section with numbers that don't
    # match the canonical list it was given. Rebuild that section from the
    # canonical source list (deterministic, idempotent) and report citation
    # health in the Research Stats block below.
    from utils.citations import validate_citations, citation_stats_line

    report_text, citation_stats = validate_citations(report_text, sources_list)

    # ── Cache & knowledge stats ──────────────────────────────────
    with get_db() as conn:
        cache_hits = conn.execute("SELECT COUNT(*) FROM search_cache").fetchone()[0]

    # ── Cost summary (if tracker is available) ────────────────────
    # Label the footer with the plan so the numbers can be attributed to a
    # session — the tracker is per-process, and resumed/continued plans run
    # in fresh processes (a bare "Model:" line used to be ambiguous).
    cost_tracker = get_active_tracker()
    cost_footer = (
        cost_tracker.format_footer(label=f"  Plan: {plan_id}")
        if cost_tracker and plan_id
        else (cost_tracker.format_footer() if cost_tracker else "")
    )

    # Append the Evidence Matrix after the report body (and after the
    # rebuilt References section), before the Research Stats block.
    if evidence_matrix_text:
        report_text = report_text.rstrip() + "\n\n" + evidence_matrix_text

    # Pass count: 0 = deterministic fallback, 1 = capped/short plan,
    # 2 = full two-pass synthesis (or pass-2 failed on a large plan).
    if last_exc is not None:
        passes_line = "  Report passes: 0 (deterministic fallback)\n"
    else:
        passes_line = f"  Report passes: {report_passes}\n"

    stats_text = (
        "\n\n--- Research Stats ---\n"
        f"  Cache hits: {cache_hits}\n"
        f"  Valid findings: {len(valid_findings)}\n"
        f"  Discarded (generic): {len(discarded)}\n"
        f"  Curated knowledge entries: {len(curated)}\n"
        f"  Open questions: {open_questions} of {total_questions}\n"
        + passes_line
        + citation_stats_line(citation_stats)
    )
    report_text += stats_text

    # Append cost footer (after research stats)
    if cost_footer:
        report_text += cost_footer

    # Save report to DB and file
    if plan_id:
        report_id = save_report(plan_id, query, report_text)
        print(f"  [REPORT] Saved {report_id}")

    return {
        "final_report": report_text,
        "is_done": True,
    }


# ── Node: Write "No Evidence" Terminal ──────────────────────


def _write_findings_no_evidence_node(state) -> dict:
    """Compile a report when there's insufficient evidence to answer the question.

    This is the Consensus pattern: explicitly state when the literature
    doesn't settle the question, rather than generating an overconfident report.
    """
    plan_id = state.get("plan_ref", "")
    query = state.get("original_query", "")

    from db.knowledge import report_quality_gate

    gate = (
        report_quality_gate(plan_id) if plan_id else {"reason": "unknown", "stats": {}}
    )
    stats = gate.get("stats", {})

    reason_map = {
        "insufficient_evidence": (
            "Insufficient evidence was found to answer this question.\n"
            "The agent searched multiple sources and found no reliable, specific findings.\n"
            "This may mean the question is too narrow, the topic is not well-documented,\n"
            "or the search terms need refinement."
        ),
        "no_sources": (
            "No sources were found for this question.\n"
            "The agent searched multiple engines (Exa, Brave, arXiv) and the knowledge base\n"
            "but did not locate any relevant content. Consider refining your query."
        ),
    }
    reason_text = reason_map.get(
        gate.get("reason", "unknown"), reason_map["insufficient_evidence"]
    )

    report_text = f"# Research Report: {query}\n\n"
    report_text += f"> **Note:** {reason_text}\n\n"
    report_text += "## Summary\n"
    report_text += f"- Total questions: {stats.get('resolved_questions', 0) + stats.get('discarded', 0)}\n"
    report_text += f"- Valid findings: {stats.get('valid_findings', 0)}\n"
    report_text += f"- Discarded (generic): {stats.get('discarded', 0)}\n"
    report_text += f"- Sources searched: {stats.get('sources', 0)}\n"

    # Add the research question section
    if plan_id:
        question_section = build_question_section(plan_id, query)
        if question_section:
            report_text = _insert_after_first_heading(report_text, question_section)

    report_text += "\n\n--- Research Stats ---\n"
    report_text += f"  Valid findings: {stats.get('valid_findings', 0)}\n"
    report_text += f"  Discarded (generic): {stats.get('discarded', 0)}\n"
    report_text += f"  Sources: {stats.get('sources', 0)}\n"

    # Save report
    if plan_id:
        from db.knowledge import save_report

        save_report(plan_id, query, report_text)

    return {
        "final_report": report_text,
        "is_done": True,
    }


# ── Routing ───────────────────────────────────────────────────────


def _routing_logic(state) -> str:
    """Conditional router — reads _route set by consolidation node.

    If _rewind_from is set (from the review screen's "rewind" command), the
    graph loops back to the backlog node so the user can choose a different
    path.
    """
    if state.get("_rewind_from") is not None:
        return "manage_backlog"
    return state.get("_route", "manage_backlog")


# ── Build the graph ───────────────────────────────────────────────

workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("make_plan", make_plan_node)
workflow.add_node("manage_backlog", manage_backlog_node)
workflow.add_node("search_execution", search_execution_node)
workflow.add_node("extract_and_verify", extract_and_verify_node)
workflow.add_node("consolidation", consolidation_node)
workflow.add_node("prepare_review", prepare_review_node)
workflow.add_node("human_review_interrupt", human_review_interrupt)
workflow.add_node("curate_knowledge", curate_knowledge_node)
workflow.add_node("write_findings", _write_findings_node)
workflow.add_node("write_findings_no_evidence", _write_findings_no_evidence_node)

# Wire the flow
workflow.add_edge(START, "make_plan")
workflow.add_edge("make_plan", "manage_backlog")
workflow.add_edge("manage_backlog", "search_execution")
workflow.add_edge("search_execution", "extract_and_verify")
workflow.add_edge("extract_and_verify", "consolidation")

# Conditional routing from consolidation
workflow.add_conditional_edges(
    "consolidation",
    _routing_logic,
    {
        "manage_backlog": "manage_backlog",
        "human_review": "prepare_review",
        "write_findings": "write_findings",
        "write_findings_no_evidence": "write_findings_no_evidence",
    },
)

# Curation flow — analysis is built once (prepare_review), then the graph
# suspends at the thin interrupt node and waits for the curator
workflow.add_edge("prepare_review", "human_review_interrupt")
workflow.add_edge("human_review_interrupt", "curate_knowledge")
workflow.add_conditional_edges(
    "curate_knowledge",
    lambda s: "consolidation" if not s.get("is_done") else "write_findings",
    {
        "consolidation": "consolidation",
        "write_findings": "write_findings",
    },
)

workflow.add_edge("write_findings", END)


def build_app(checkpointer: Optional[object] = None):
    """Compile the research graph with a checkpointer.

    Args:
        checkpointer: LangGraph checkpointer instance. None (default) →
            SQLite-backed persistence in data/checkpoints.db, which enables
            resuming sessions after a process restart (run.py --resume).
            Tests should pass MemorySaver().

    Each call also registers a fresh cost tracker (utils.cost) as the
    active one, so LLM cost accounting is per-app, not process-global.
    """
    if checkpointer is None:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        from config.settings import paths_config

        conn = sqlite3.connect(paths_config.checkpoint_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)

    # A fresh cost tracker per app: separate apps (e.g. tests) never share
    # cost state. Registered as the active tracker for invoke_with_cost().
    tracker = CostTracker()
    set_active_tracker(tracker)
    return workflow.compile(checkpointer=checkpointer)
