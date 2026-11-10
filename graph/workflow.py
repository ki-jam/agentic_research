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


# ── Node: Write Final Report ──────────────────────────────────────


def _write_findings_node(state) -> dict:
    """Compile final report from THIS PLAN's DB findings and curated knowledge."""
    from config.prompts import REPORT_SYSTEM, REPORT_USER
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

        # Batch engine lookup (N+1 → 1 query)
        raw_ids = [f["raw_result_id"] for f in findings if f.get("raw_result_id")]
        if raw_ids:
            placeholders = ",".join("?" for _ in raw_ids)
            for row in conn.execute(
                f"SELECT id, engine FROM raw_results WHERE id IN ({placeholders})",
                raw_ids,
            ):
                engine_map[row[0]] = row[1]

    # Also add sources from curated knowledge entries
    for entry in curated:
        url = entry.get("source_url", "")
        title = entry.get("title", "")
        if url and (url, title) not in sources:
            sources.add((url, title))

    # Sort sources for consistent ordering
    sources_list = sorted(sources, key=lambda x: x[0])

    # ── Separate citation discovery findings from search findings ──
    citation_findings = []
    search_findings = []
    for f in findings:
        raw_id = f.get("raw_result_id")
        if raw_id and engine_map.get(raw_id, "").startswith("exa-answer"):
            citation_findings.append(f)
        else:
            search_findings.append(f)

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
        curated_text += f"Source: {entry.get('source_url', 'N/A')}\n\n"

    # Build search findings text
    search_findings_text = ""
    for f in search_findings:
        search_findings_text += f"### Finding: {f.get('summary', 'Untitled')}\n"
        search_findings_text += f"Q: {f.get('source_question', 'N/A')}\n"
        facts = f.get("facts", [])
        if isinstance(facts, list):
            for fact in facts:
                if isinstance(fact, dict):
                    search_findings_text += (
                        f"- {fact.get('claim', '')} ({fact.get('confidence', '?')})\n"
                    )
                else:
                    search_findings_text += f"- {fact}\n"
        tags = f.get("tags", [])
        if isinstance(tags, list):
            search_findings_text += f"Tags: {', '.join(tags)}\n"
        search_findings_text += "\n"

    # Build citation discovery findings text (prominently labeled)
    citation_findings_text = ""
    if citation_findings:
        citation_findings_text += "### 📚 FOUNDATIONAL PAPERS (Citation Discovery)\n"
        citation_findings_text += "These findings come from the Exa answer API, which identified foundational/authoritative papers on the topic.\n\n"
        for f in citation_findings:
            citation_findings_text += f"### Finding: {f.get('summary', 'Untitled')}\n"
            citation_findings_text += f"Q: {f.get('source_question', 'N/A')}\n"
            facts = f.get("facts", [])
            if isinstance(facts, list):
                for fact in facts:
                    if isinstance(fact, dict):
                        citation_findings_text += f"- {fact.get('claim', '')} ({fact.get('confidence', '?')})\n"
                    else:
                        citation_findings_text += f"- {fact}\n"
            tags = f.get("tags", [])
            if isinstance(tags, list):
                citation_findings_text += f"Tags: {', '.join(tags)}\n"
            citation_findings_text += "\n"

    findings_text = search_findings_text + citation_findings_text + curated_text

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

    # Try LLM-generated report
    try:
        # The report is the longest LLM call in the pipeline. Thinking models
        # can spend the entire default token budget on reasoning and return an
        # empty answer (observed live on plan_d3df337a), so it gets its own
        # larger output budget.
        model = config.get_chat_model(task="report", max_tokens=32768)
        prompt = f"{REPORT_SYSTEM}\n\n{
            REPORT_USER.format(
                query=query,
                plan=questions_text,
                findings=findings_text,
                references=references_text,
            )
        }"
        response = invoke_with_cost(model, prompt, task="report")
        report_text = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        # Empty or whitespace-only content does not raise — treat it as a
        # failure so the deterministic fallback engages instead of persisting
        # a bodyless report silently.
        if len(report_text.strip()) < 200:
            raise ValueError(
                f"LLM returned an empty or too-short report body "
                f"({len(report_text.strip())} chars after stripping)"
            )
    except Exception as exc:
        # Fallback: simple text report — deterministic, always complete
        report_text = f"# Research Report: {query}\n\n"
        report_text += (
            f"> **Note:** LLM report generation failed or returned no usable "
            f"content ({type(exc).__name__}). This is the deterministic "
            f"fallback: all extracted findings, unsynthesized.\n\n"
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
    if plan_id:
        question_section = build_question_section(plan_id, query)
        if question_section:
            report_text = _insert_after_first_heading(report_text, question_section)

    # ── Cache & knowledge stats ──────────────────────────────────
    with get_db() as conn:
        cache_hits = conn.execute("SELECT COUNT(*) FROM search_cache").fetchone()[0]

    # ── Cost summary (if tracker is available) ────────────────────
    cost_tracker = get_active_tracker()
    cost_footer = cost_tracker.format_footer() if cost_tracker else ""

    # Insert Evidence Matrix before research stats (if LLM report exists, insert after it)
    if evidence_matrix_text:
        if report_text and not report_text.strip().startswith("# Research Report:"):
            # LLM-generated report — insert Evidence Matrix before the Research Question section
            report_text = report_text.rstrip() + "\n\n" + evidence_matrix_text
        else:
            # Fallback or empty report — add Evidence Matrix
            report_text += evidence_matrix_text

    stats_text = (
        "\n\n--- Research Stats ---\n"
        f"  Cache hits: {cache_hits}\n"
        f"  Valid findings: {len(valid_findings)}\n"
        f"  Discarded (generic): {len(discarded)}\n"
        f"  Curated knowledge entries: {len(curated)}\n"
    )
    report_text += stats_text

    # Append cost footer (after research stats)
    if cost_footer:
        report_text += cost_footer

    # Save report to DB and file
    if plan_id:
        save_report(plan_id, query, report_text)

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
