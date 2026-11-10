"""
Consolidation node — checks if research is complete or needs another cycle.

Uses multiple signals:
- LLM evaluation of research quality (thoroughness, evidence, coverage, usefulness)
- Minimum findings threshold
- Max iterations
- Stagnation (no new findings for N iterations)
- Human review cadence

The consolidation node sets _route which the router reads.
"""

from __future__ import annotations

from config.prompts import REPORT_SYSTEM
from config.settings import config, workflow_config

from db.engine import get_db
from db.knowledge import (
    add_question,
    all_questions_are_resolved,
    get_plan,
    get_plan_findings,
    get_pending_questions,
    report_quality_gate,
)
from utils.llm import parse_json, invoke_with_cost
from utils.quality import filter_quality, is_science_topic


def _evaluate_readiness(
    state, *, all_findings=None, valid_findings=None, discarded=None, all_resolved=None
) -> dict:
    """Ask the LLM to evaluate whether the research is ready to stop.

    Checks four criteria:
    1. Thoroughness — have we reviewed the question from multiple angles?
    2. Evidence — do findings have concrete facts with sources?
    3. Coverage — are key themes and open questions addressed?
    4. Usefulness — would a newcomer to this topic get value from the report?

    Args:
        all_findings, valid_findings, discarded, all_resolved:
            Pre-fetched by the caller (consolidation_node) to avoid
            duplicate DB queries.  If None, fall back to DB lookups.

    Returns: {ready: bool, gaps: list[str], confidence: str}
    On LLM failure returns {"failed": True} so the caller can fall back to
    a heuristic instead of treating "not ready" as a real assessment.
    """
    plan_id = state.get("plan_ref", "")
    original_query = state.get("original_query", "")

    if not plan_id:
        return {"ready": False, "gaps": ["No plan"], "confidence": "low"}

    try:
        # Use pre-fetched data or fall back to DB (for tests that pass None)
        if all_resolved is None:
            all_resolved = all_questions_are_resolved(plan_id)
        if all_findings is None:
            all_findings = get_plan_findings(plan_id)
        if valid_findings is None or discarded is None:
            valid_findings, discarded = filter_quality(all_findings)

        plan = get_plan(plan_id)

        # Build a compact representation of all findings
        findings_text = (
            "\n".join(
                f"F[{i + 1}] (from: {f.get('source_question', '?')[:60]})\n"
                f"  {f.get('summary', '')[:200]}\n"
                f"  Facts: {
                    ', '.join(
                        fact.get('claim', '') if isinstance(fact, dict) else str(fact)
                        for fact in f.get('facts', [])[:3]
                    )
                }"
                for i, f in enumerate(valid_findings[:10])
            )
            or "(none)"
        )

        questions_text = (
            "\n".join(
                f"- {q.get('question', '')}"
                for q in (
                    plan.get("questions", [])
                    if isinstance(plan.get("questions", []), list)
                    else []
                )
            )
            or "(none)"
        )

        prompt = f"""You are evaluating whether a research session is ready to produce a final report.

Research topic: {original_query}

Questions investigated:
{questions_text}

Valid findings extracted ({len(valid_findings)}):
{findings_text}

Discarded findings ({len(discarded)}): {len(discarded)} (generic/irrelevant)

Evaluate using these four criteria:

1. THOROUGHNESS — Have we examined the question from multiple angles?
   (definition, mechanism, comparison, trade-offs, context)

2. EVIDENCE — Do our findings have concrete, traceable facts?
   (not vague summaries — actual claims with confidence levels)

3. COVERAGE — Are the key themes and open questions addressed?
   (No major gaps in understanding the topic)

4. USEFULNESS — Would someone new to this topic get real value?
   (A newcomer should understand the basics and the nuances)

Return ONLY valid JSON:
{{
  "ready": true/false,
  "gaps": ["list of specific gaps or missing angles"],
  "confidence": "high|medium|low"
}}

Be honest. "ready" should be false if the report would feel thin, incomplete, or superficial.
A good report should help a newcomer understand the topic and its nuances."""

        model = config.get_chat_model(task="curation")
        response = invoke_with_cost(
            model, f"System: {REPORT_SYSTEM}\n\n{prompt}", task="consolidation"
        )
        data = parse_json(response.content)
        if isinstance(data, dict):
            return {
                "ready": bool(data.get("ready", False)),
                "gaps": data.get("gaps", []),
                "confidence": data.get("confidence", "low"),
            }
    except Exception:
        pass

    return {"failed": True}


def consolidation_node(state) -> dict:
    """Decide whether to continue, review, or stop."""

    plan_id = state.get("plan_ref")
    iteration = state.get("iteration_count", 0) + 1

    # ── DB-level checks ────────────────────────────────────────
    # Compute once; _evaluate_readiness re-uses these instead of
    # hitting the DB again (P2-3 duplicate work).
    all_findings = get_plan_findings(plan_id) if plan_id else []
    valid_findings, discarded = filter_quality(all_findings)
    num_findings = len(valid_findings)
    all_resolved = all_questions_are_resolved(plan_id) if plan_id else False

    # Stagnation: no new findings for the last 3 iterations
    prev_streak = state.get("_findings_streak", 0)
    prev_finding_count = state.get("_prev_finding_count", 0)
    if num_findings > prev_finding_count:
        findings_streak = 0
    elif num_findings == prev_finding_count and iteration > 1:
        findings_streak = prev_streak + 1
    else:
        findings_streak = 0

    # ── Enforcement: at least one actual search result ─────────
    # (citation discovery = exa-answer is NOT an actual search)
    has_actual_search = _has_actual_search_results(plan_id) if plan_id else False

    # ── Enforcement: arXiv required for science topics ────────
    # Checked at the question level — the same level at which search.py
    # decides to trigger an arXiv search. (The old check used the original
    # query, so a science-flagged query whose sub-questions were not
    # science-flagged made the gate unsatisfiable: the loop spun until
    # max iterations.)
    needs_arxiv = _plan_has_science_questions(plan_id) if plan_id else False
    has_arxiv_search = _has_arxiv_results(plan_id) if needs_arxiv else True

    # ── Citation coverage check ────────────────────────────────
    # If citation discovery ran but produced findings, ensure they're represented
    citation_coverage = _check_citation_coverage(plan_id) if plan_id else {}

    # ── Quality gate — "no evidence" check ─────────────────────
    # Before asking the LLM, check if there's ANY evidence worth reporting.
    # If not, route to the no-evidence terminal so the user gets a
    # transparent "insufficient evidence" message (Consensus pattern).
    quality_gate = {"ok": True, "reason": "sufficient"}
    if plan_id:
        quality_gate = report_quality_gate(plan_id)

    # ── LLM quality evaluation ──────────────────────────────────
    # Pass pre-fetched data via state so _evaluate_readiness avoids
    # duplicate DB calls (get_plan_findings + _filter_quality).
    quality = _evaluate_readiness(
        state,
        all_findings=all_findings,
        valid_findings=valid_findings,
        discarded=discarded,
        all_resolved=all_resolved,
    )
    if quality.get("failed"):
        # LLM evaluation unavailable — heuristic fallback: ready iff every
        # question is terminal (resolved/discarded) and we have enough findings.
        quality = {
            "ready": bool(
                all_resolved and num_findings >= workflow_config.min_findings
            ),
            "gaps": [],
            "confidence": "low",
            "heuristic": True,
        }

    # ── Decision logic ──────────────────────────────────────────
    review_every = workflow_config.review_every_n_iterations
    min_findings = workflow_config.min_findings
    max_iterations = workflow_config.max_iterations
    hit_max_iterations = iteration >= max_iterations

    # At max iterations → always show interim analysis (even if data is thin).
    # The interrupt now actually suspends the graph, and "stop" exits cleanly.
    if hit_max_iterations:
        route = "human_review"
    # Hard blocks: don't allow auto-stop without actual data
    elif not has_actual_search:
        route = "manage_backlog"
    elif needs_arxiv and not has_arxiv_search:
        route = "manage_backlog"
    elif citation_coverage.get("missing_findings", False):
        # Citation discovery produced raw results but no findings extracted
        # Route to backlog so extraction re-processes citation content
        route = "manage_backlog"
    elif not quality_gate["ok"]:
        # No sufficient evidence — route to no-evidence terminal
        route = "write_findings_no_evidence"
    elif quality["ready"] and num_findings >= min_findings:
        route = "write_findings"
    elif findings_streak >= 3:
        route = "write_findings"
    elif iteration % review_every == 0 and iteration >= review_every:
        route = "human_review"
    else:
        route = "manage_backlog"

    # ── Self-correction: turn DoD gaps into questions ──────────
    # If the LLM identifies specific gaps and the queue isn't already
    # drowning in gaps, add them as questions so the loop can actually
    # address the missing angles. We limit to 5 per pass to prevent
    # question spam, and skip gaps that already exist as questions.
    #
    # This runs in two scenarios:
    # 1. During backlog routing (manage_backlog) — gaps become pending questions
    # 2. At max iterations (human_review) — gaps are added so the user can
    #    investigate them via "gap <n>" or "continue"
    #
    # At max iterations we deduplicate against ALL existing questions (not
    # just pending) to avoid re-adding gaps that were already created during
    # backlog routing in earlier iterations.
    add_to_backlog = route == "manage_backlog" or hit_max_iterations
    if add_to_backlog and not quality["ready"] and quality.get("gaps") and plan_id:
        # Build set of existing question texts (fuzzy match by keyword overlap)
        all_qs = get_pending_questions(plan_id) if plan_id else []
        existing_kw = set()
        for q in all_qs:
            for w in q["question"].lower().split():
                if len(w) > 3:
                    existing_kw.add(w)

        added = 0
        for gap in quality["gaps"]:
            if added >= 5:
                break
            if (
                isinstance(gap, str)
                and gap.strip()
                and gap != "Questions still pending"
            ):
                # Skip if gap text overlaps significantly with an existing question
                gap_kw = {w for w in gap.lower().split() if len(w) > 3}
                if gap_kw & existing_kw:
                    continue  # similar question already exists

                add_question(plan_id, gap.strip(), discovered_by="dod_gaps")
                added += 1
                # Track new keywords to avoid adding similar gaps in same pass
                for w in gap.strip().lower().split():
                    if len(w) > 3:
                        existing_kw.add(w)
        if added:
            print(f"  [DOD GAPS → QUESTIONS] Added {added} gap(s) to the question bank")

    # ── Consume _rewind_from flag after one backlog pass ────────
    # The flag tells the router to jump to manage_backlog once. After
    # that pass, clear it so normal _route-based routing resumes.
    rewind_from = state.get("_rewind_from")
    clear_rewind = {"_rewind_from": None} if rewind_from is not None else {}

    return {
        "iteration_count": iteration,
        "_route": route,
        "_findings_streak": findings_streak,
        "_prev_finding_count": num_findings,
        "_quality": quality,
        "_hit_max_iterations": hit_max_iterations if route == "human_review" else False,
        **clear_rewind,
    }


# ── Enforcement helpers ────────────────────────────────────────────


def _has_actual_search_results(plan_id: str) -> bool:
    """Check if there's at least one actual search result (not just citation discovery).

    Citation discovery results have engine starting with 'exa-answer'.
    Actual search results have engines like 'exa', 'brave', 'exa+arxiv', etc.
    """
    try:
        with get_db() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM raw_results WHERE plan_id = ? AND engine NOT LIKE 'exa-answer%'",
                (plan_id,),
            ).fetchone()[0]
        return count > 0
    except Exception:
        return False


def _has_arxiv_results(plan_id: str) -> bool:
    """Check if there's at least one arXiv search result for this plan."""
    try:
        with get_db() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM raw_results WHERE engine LIKE '%arxiv%' AND plan_id = ?",
                (plan_id,),
            ).fetchone()[0]
        return count > 0
    except Exception:
        return False


def _plan_has_science_questions(plan_id: str) -> bool:
    """True if any of the plan's questions (any status) is science-flagged.

    This is the level at which search.py decides to trigger an arXiv search,
    so the gate and the search stay consistent: the loop only requires an
    arXiv result when a question that would actually trigger one exists.
    """
    try:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT question FROM questions WHERE plan_id = ?",
                (plan_id,),
            ).fetchall()
        return any(is_science_topic(r[0]) for r in rows)
    except Exception:
        return False


def _check_citation_coverage(plan_id: str) -> dict:
    """Check whether citation-discovery content has been through extraction.

    The citation raw result is attached to the first question, so its
    content is extracted when that question is processed. Findings only link
    to ONE raw result per extraction batch, so counting linked findings is an
    unreliable proxy: a batch that contains citation content can link its
    findings to the batch's other raw result, leaving missing_findings=True
    forever — which routed every consolidation to manage_backlog and blocked
    the periodic review (observed live on plan_d3df337a). Use the attached
    question's status instead: unresolved → content not (re)processed yet.
    """
    result = {"has_citation_raw": False, "missing_findings": False}
    try:
        with get_db() as conn:
            rows = conn.execute(
                """SELECT q.status FROM raw_results rr
                   JOIN questions q ON q.id = rr.question_id
                   WHERE rr.engine LIKE 'exa-answer%' AND rr.plan_id = ?""",
                (plan_id,),
            ).fetchall()
        result["has_citation_raw"] = len(rows) > 0
        result["missing_findings"] = any(
            r[0] in ("pending", "in_progress", "failed") for r in rows
        )
    except Exception:
        pass
    return result
