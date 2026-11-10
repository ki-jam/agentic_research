"""
Extraction node — reads raw search results from DB, calls LLM to extract
facts/findings, writes curated findings to the curated_findings table,
and optionally discovers new questions to add to the bank.
"""

from __future__ import annotations

from config.prompts import EXTRACTION_SYSTEM, EXTRACTION_USER
from config.settings import config
from db.knowledge import (
    get_in_progress_question,
    get_raw_results_for_question,
    set_question_resolved,
    set_question_failed,
    add_question,
    create_finding,
    create_knowledge_entry,
)
from utils.llm import parse_json, invoke_with_cost


def extract_and_verify_node(state) -> dict:
    """Extract findings from raw search results stored in DB."""

    plan_id = state.get("plan_ref")
    if not plan_id:
        return {}

    # Get the in-progress question and its raw results
    question = get_in_progress_question(plan_id)
    if not question:
        return {}

    raw_results = get_raw_results_for_question(question["id"])
    if not raw_results:
        # No results found — mark question resolved anyway
        set_question_resolved(question["id"])
        return {}

    # Build context from raw results.
    # Only results with non-empty content are included — these are the blocks
    # the LLM sees and indexes into via the "source" field in its output.
    # Two caps: MAX_CONTEXT per source, and MAX_TOTAL_CONTEXT for the whole
    # prompt — without the total cap, 20+ sources x 4k chars can overflow the
    # model's context window and fail the entire extraction.
    context_parts = []
    raw_result_ids = []
    raw_result_urls = []
    MAX_CONTEXT = 4000
    MAX_TOTAL_CONTEXT = 24000

    # Citation-discovery blocks first: they're small and the point of that
    # pass. sorted() is stable, so all other sources keep their current order.
    ordered = sorted(
        raw_results,
        key=lambda rr: 0 if rr.get("engine", "").startswith("exa-answer") else 1,
    )

    used = 0
    omitted = 0
    for rr in ordered:
        content = rr.get("content", "")
        engine = rr.get("engine", "")
        if not content:
            continue
        # Label citation discovery results so LLM treats them with special attention
        if engine.startswith("exa-answer"):
            label = "[FOUNDATIONAL PAPERS / CITATION DISCOVERY]"
        else:
            label = f"[SOURCE: {engine}]"
        block = f"{label}\n{content[:MAX_CONTEXT]}"
        if used + len(block) > MAX_TOTAL_CONTEXT:
            omitted += 1
            continue
        context_parts.append(block)
        used += len(block)
        raw_result_ids.append(rr["id"])
        raw_result_urls.append(rr.get("source_url", ""))

    context = "\n\n---\n\n".join(context_parts)
    if omitted:
        context += (
            f"\n\n[Note: {omitted} additional source(s) were omitted to fit the "
            "context limit — base findings only on the sources shown.]"
        )

    # Call LLM to extract findings
    findings, error = _extract_findings(state, question, context)
    if error:
        # LLM failure is NOT the same as "no findings": mark the question
        # failed so it blocks readiness and shows up in the interim analysis.
        set_question_failed(question["id"], error)
        print(f"  [EXTRACTION FAILED] {question['question'][:60]}: {error}")
        return {}

    # Store each finding in DB and link to the relevant raw results
    for finding in findings:
        # Build full text from facts for knowledge entry
        facts_text = "\n".join(
            f"- {f.get('claim', '')} (confidence: {f.get('confidence', 'unknown')})"
            if isinstance(f, dict)
            else str(f)
            for f in finding.get("facts", [])
        )
        full_content = f"Summary: {finding.get('summary', '')}\n\nFacts:\n{facts_text}"

        # Determine the primary source for this finding based on the most common
        # source index in its facts. If no source index is provided, fall back to
        # the first raw result (the old behavior).
        source_indices = [
            f.get("source")
            for f in finding.get("facts", [])
            if isinstance(f, dict) and "source" in f
        ]
        primary_source_idx = (
            max(set(source_indices), key=source_indices.count) if source_indices else 0
        )
        source_raw_id = raw_result_ids[primary_source_idx] if raw_result_ids else None

        # Create knowledge entry (status="raw", can be curated later)
        entry_id = create_knowledge_entry(
            title=finding.get("summary", "")[:120] or "Untitled finding",
            summary=finding.get("summary", ""),
            topic_tags=finding.get("tags", []),
            full_text_path="",  # No file — stored inline
            source_url=raw_result_urls[primary_source_idx]
            if raw_result_urls and primary_source_idx < len(raw_result_urls)
            else "",
            discovered_by="extraction",
            full_text=full_content,
        )

        # Create single finding linked to the primary source raw result
        create_finding(
            question_id=question["id"],
            raw_result_id=source_raw_id,
            knowledge_entry_id=entry_id,
            facts=finding.get("facts", []),
            summary=finding.get("summary", ""),
            tags=finding.get("tags", []),
        )

    # Check for new questions discovered during extraction
    for new_q in findings:
        discovered = new_q.get("discovered_questions", [])
        for dq in discovered:
            add_question(plan_id, dq, discovered_by="extraction")

    # Mark question as resolved
    set_question_resolved(question["id"])

    return {}


def _extract_findings(state, question: dict, context: str) -> tuple[list, str | None]:
    """Call LLM to extract structured findings from search results.

    Returns (findings, error). On LLM/parse failure findings is [] and
    error describes what went wrong — the caller must NOT treat that as
    "no findings found".
    """
    try:
        prompt = EXTRACTION_USER.format(
            question=question["question"],
            context=context,
        )

        model = config.get_chat_model(task="extraction")
        response = invoke_with_cost(
            model, f"{EXTRACTION_SYSTEM}\n\n{prompt}", task="extraction"
        )
        data = parse_json(response.content)

        if not isinstance(data, list):
            data = [data] if data else []

        return data, None

    except Exception as e:
        return [], str(e)
