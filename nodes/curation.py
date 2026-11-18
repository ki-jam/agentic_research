"""
Curation nodes — human-in-the-loop review and library curation.

The curator reviews progress updates, curates resources into the library,
and decides when research is complete.
"""

from __future__ import annotations

from config.prompts import CURATE_SYSTEM, CURATE_USER
from config.settings import config
from langgraph.types import interrupt

from db.knowledge import (
    create_degraded_evidence,
    get_plan_entries,
    get_plan_findings,
    get_question,
    get_raw_results_for_question,
    get_stats,
    set_question_pending,
    set_question_resolved,
    update_knowledge_status,
    update_tags,
    deduplicate_and_prioritize,
)
from db.engine import get_db
from nodes.consolidation import _coverage_floor_ok, _question_coverage
from utils.llm import parse_json, invoke_with_cost
from utils.quality import filter_quality
from utils.scope import scope_for_plan


# ── Contradiction detector ────────────────────────────────────────


def _detect_contradictions(
    findings: list[dict], plan_id: str | None = None
) -> tuple[list[dict], list[str]]:
    """Flag findings that contradict each other and suggest resolution questions."""
    if len(findings) < 2:
        return [], []

    try:
        finding_list = []
        for f in findings:
            finding_list.append(
                {
                    "id": f["id"],
                    "question": f.get("source_question", ""),
                    "summary": f.get("summary", ""),
                    "facts": [
                        fact.get("claim", "") if isinstance(fact, dict) else str(fact)
                        for fact in f.get("facts", [])
                    ],
                }
            )

        findings_json = "\n".join(
            f"Finding {i + 1} (id={f['id']}, Q: {f['question'][:60]})\n"
            f"  Summary: {f['summary'][:200]}\n"
            f"  Facts: {', '.join(f['facts'][:3])}"
            for i, f in enumerate(finding_list)
        )

        prompt = f"""You are reviewing research findings for consistency and identifying gaps.

Findings:
{findings_json}

Step 1: Identify any pairs of findings that make contradictory claims about the same topic.
Step 2: For each contradiction, suggest 1-2 specific research questions that would help resolve it.

IMPORTANT: Only flag contradictions about LLM serving/scheduling frameworks.
Skip contradictions about infrastructure tools (Kafka, Kubernetes), quantization libraries,
or model training — these are out of scope.

Return ONLY valid JSON — an array:
[
  {{
    "finding_a": "id_of_first_finding",
    "finding_b": "id_of_second_finding",
    "explanation": "What specifically contradicts and why",
    "resolution_questions": ["Question to resolve this contradiction?"]
  }}
]

If no contradictions, return []."""

        model = config.get_chat_model(task="curation")
        response = invoke_with_cost(
            model,
            f"System: You are a consistency checker for research findings.\n\n{prompt}",
            task="consolidation",
        )
        data = parse_json(response.content)
        if not isinstance(data, list):
            data = []

        pairs = []
        suggested_questions = []
        for item in data:
            if isinstance(item, dict):
                pairs.append(
                    {
                        "finding_a": item.get("finding_a", ""),
                        "finding_b": item.get("finding_b", ""),
                        "explanation": item.get("explanation", ""),
                    }
                )
                for q in item.get("resolution_questions", []):
                    suggested_questions.append(q)

        if plan_id and suggested_questions:
            from db.knowledge import add_question

            for q in suggested_questions:
                add_question(plan_id, q, discovered_by="contradiction_resolution")

        return pairs, suggested_questions
    except Exception:
        pass

    return [], []


# ── LLM-assisted question dedup ───────────────────────────────────


def _llm_dedup_open_questions(plan_id: str) -> list[dict]:
    """LLM-assisted duplicate detection for open (pending) questions.

    Complements the keyword-based deduplicate_and_prioritize(): Jaccard
    word overlap (>= 0.4) only catches near-verbatim paraphrases, and it
    only compares pending-to-pending. This pass batches the open
    questions plus the recently resolved ones (context) into one LLM
    call and asks for open questions that ask for the same information
    as another open question or as a question the plan already answered
    — the observed backlog-growth mode (re-asking a just-resolved
    question under different words).

    Discards follow the same logic as the keyword pass: the duplicate
    row becomes status='discarded', priority='low', discovered_by=
    'llm-dedup' (traceable; revivable with 'retry <id>' on the review
    screen).

    Safety rails (the local model is flaky):
    - only question IDs present in the prompt can ever be discarded
    - a group where the LLM flags every member against each other keeps
      its best question (union-find over the discard graph)
    - any failure (LLM down, bad JSON, no duplicates) → no changes, []

    Returns [{"id", "question", "duplicate_of", "reason"}, ...] for the
    rows actually discarded, for the review screen.
    """
    try:
        with get_db() as conn:
            open_rows = conn.execute(
                """SELECT id, question, priority, created_at FROM questions
                   WHERE plan_id = ? AND status = 'pending'
                   ORDER BY created_at""",
                (plan_id,),
            ).fetchall()
            resolved_rows = conn.execute(
                """SELECT id, question FROM questions
                   WHERE plan_id = ? AND status = 'resolved'
                   ORDER BY created_at DESC LIMIT 40""",
                (plan_id,),
            ).fetchall()

        if len(open_rows) < 2:
            return []

        open_by_id = {r["id"]: dict(r) for r in open_rows}
        resolved_ids = {r["id"] for r in resolved_rows}

        open_text = "\n".join(f"- id: {r['id']} — {r['question']}" for r in open_rows)
        resolved_text = (
            "\n".join(f"- id: {r['id']} — {r['question']}" for r in resolved_rows)
            or "(none)"
        )

        prompt = f"""You are deduplicating a list of open research questions.

OPEN questions:
{open_text}

ALREADY ANSWERED (resolved) questions, for reference:
{resolved_text}

Two questions are duplicates if they ask for the same information, even if worded differently.

Return ONLY valid JSON:
{{"discard": [
  {{"id": "<open question id>", "duplicate_of": "<id of the question it duplicates, open or answered>", "reason": "<one short clause>"}}
]}}

Rules:
- "id" MUST be an OPEN question id from the list above.
- "duplicate_of" MUST be an id from either list.
- Only discard an open question if it is genuinely a duplicate.
- If there are no duplicates, return {{"discard": []}}."""

        model = config.get_chat_model(task="curation")
        response = invoke_with_cost(model, prompt, task="curation")
        data = parse_json(response.content)
        entries = data.get("discard", []) if isinstance(data, dict) else []
        if not isinstance(entries, list):
            entries = []

        # Validate against the real bank rows — the LLM may hallucinate ids.
        parent = {qid: qid for qid in open_by_id}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        anchored_resolved: set[str] = set()
        for e in entries:
            if not isinstance(e, dict):
                continue
            qid = e.get("id")
            dup_of = e.get("duplicate_of")
            if qid not in open_by_id:
                continue
            if dup_of in open_by_id:
                parent[find(qid)] = find(dup_of)
            elif dup_of in resolved_ids:
                anchored_resolved.add(find(qid))

        # Decide discards per component: anchored to a resolved question →
        # discard every member; otherwise keep the best (priority, then age).
        components: dict[str, list[str]] = {}
        for qid in open_by_id:
            components.setdefault(find(qid), []).append(qid)

        to_discard: set[str] = set()
        for root, members in components.items():
            if root in anchored_resolved or any(
                m in anchored_resolved for m in members
            ):
                to_discard.update(members)
                continue
            if len(members) > 1:
                priority_order = {"high": 0, "medium": 1, "low": 2}
                members.sort(
                    key=lambda q: (
                        priority_order.get(open_by_id[q]["priority"], 1),
                        open_by_id[q]["created_at"],
                        q,
                    )
                )
                to_discard.update(members[1:])

        if not to_discard:
            return []

        # Keep the LLM's duplicate_of/reason for each discarded row.
        meta = {}
        for e in entries:
            if isinstance(e, dict) and e.get("id") in to_discard:
                meta[e["id"]] = {
                    "duplicate_of": e.get("duplicate_of", ""),
                    "reason": str(e.get("reason", ""))[:120],
                }

        with get_db() as conn:
            for qid in sorted(to_discard):
                conn.execute(
                    "UPDATE questions SET status = 'discarded', priority = 'low', "
                    "discovered_by = 'llm-dedup' WHERE id = ?",
                    (qid,),
                )

        return [
            {
                "id": qid,
                "question": open_by_id[qid]["question"],
                **meta.get(qid, {"duplicate_of": "", "reason": ""}),
            }
            for qid in sorted(to_discard)
        ]
    except Exception:
        # Dedup must never block the review — fall back to the keyword pass only.
        return []


# ── Curation recommendations ──────────────────────────────────────


def _generate_curation_recommendations(state, raw_entries, findings) -> list:
    """Ask LLM to recommend which raw entries to curate, discard, or keep."""
    try:
        entries_text = (
            "\n".join(
                f"- [{e['id']}] {e.get('title', 'Untitled')[:60]} (tags: {', '.join(e.get('topic_tags', []))})\n  {e.get('summary', '')[:200]}"
                for e in raw_entries
            )
            or "  (none)"
        )

        findings_text = (
            "\n".join(
                f"- Q: {f.get('source_question', 'N/A')[:80]}\n  {f.get('summary', '')[:200]}"
                for f in findings
            )
            or "  (none)"
        )

        prompt = CURATE_USER.format(
            query=state.get("original_query", ""),
            scope=scope_for_plan(state.get("plan_ref", "")),
            entries_text=entries_text,
            findings_text=findings_text,
        )

        model = config.get_chat_model(task="curation")
        response = invoke_with_cost(
            model, f"{CURATE_SYSTEM}\n\n{prompt}", task="curation"
        )
        data = parse_json(response.content)
        if isinstance(data, list):
            return data

    except Exception:
        pass

    return []


def _build_interim_prompt(
    state,
    valid_findings,
    discarded_findings,
    contradictions,
    contradiction_questions,
    open_questions,
    quality,
    hit_max,
    llm_dedup=None,
) -> str:
    """Build the analysis text shown on the review screen.

    Only the context-dependent analysis lives here (coverage, interim
    stats, gaps, contradiction notes). The open-question list and the
    command reference are rendered by the review screen itself
    (ui/review.py) from the rest of the payload, so the prompt stays
    short.
    """
    lines = ["How would you like to proceed?"]

    # Question-bank coverage (set by consolidation). When the coverage
    # floor blocks auto-stop, say so explicitly — "stop" still works and
    # writes an interim report.
    coverage = state.get("_coverage") or {}
    if coverage.get("total"):
        lines.append(
            f"  Question coverage: {coverage.get('closed', 0)}/"
            f"{coverage.get('total', 0)} closed "
            f"({coverage.get('open', 0)} open, "
            f"{coverage.get('high_open', 0)} high-priority)"
        )
        if coverage.get("open", 0) > 0 and not _coverage_floor_ok(coverage):
            lines.append(
                '    Auto-stop is blocked — "stop" writes an interim report, '
                '"continue" keeps researching.'
            )

    if llm_dedup:
        lines.append(
            f"  LLM dedup: {len(llm_dedup)} duplicate question(s) discarded "
            "this review (revive one with 'retry <id>'):"
        )
        for d in llm_dedup:
            lines.append(f"    - {d['question'][:80]} (dup of {d['duplicate_of']})")

    if hit_max:
        lines.append("")
        lines.append("  === INTERIM ANALYSIS (max iterations reached) ===")
        lines.append(f"  Iterations: {state.get('iteration_count', 0)}")
        lines.append(
            f"  Valid findings: {len(valid_findings)} | Discarded: {len(discarded_findings)}"
        )
        if quality.get("gaps"):
            lines.append("")
            lines.append("  Identified gaps:")
            for gap in quality["gaps"]:
                lines.append(f"    - {gap}")
    else:
        lines.append("")
        lines.append(
            f"  Quality: {len(valid_findings) + len(discarded_findings)} extracted, "
            f"{len(valid_findings)} passed filter, "
            f"{len(discarded_findings)} discarded (generic filler)."
        )
        lines.append(f"  Contradictions: {len(contradictions)} pair(s) flagged.")
        if contradiction_questions:
            lines.append("")
            lines.append("  Suggested questions from contradictions:")
            for q in contradiction_questions:
                lines.append(f"    - {q}")

    return "\n".join(lines)


def prepare_review_node(state) -> dict:
    """Build the progress analysis shown at the review screen.

    Runs once per review; the payload is checkpointed in state
    (_review_payload). The interrupt lives in a separate node
    (human_review_interrupt), so on resume LangGraph re-runs only the thin
    interrupt node — this analysis (dedup + up to three LLM calls) is not
    repeated.

    At max iterations, presents an interim analysis with prioritized next steps.
    """

    plan_id = state.get("plan_ref")
    hit_max = state.get("_hit_max_iterations", False)

    # ── Dedup + prioritize ──
    # Two layers: keyword Jaccard (cheap, near-verbatim), then one LLM call
    # for semantic duplicates — including re-asks of already-resolved
    # questions, which the keyword pass never sees.
    llm_dedup = []
    if plan_id:
        deduplicate_and_prioritize(plan_id)
        llm_dedup = _llm_dedup_open_questions(plan_id)
        if llm_dedup:
            # The LLM dedup discarded questions after consolidation
            # computed the coverage snapshot — refresh it so the review
            # screen's "Question coverage" line matches the question
            # stats line (both post-dedup).
            state["_coverage"] = _question_coverage(plan_id)

    # Plan-scoped data (the global versions leak other plans' findings)
    raw_entries = get_plan_entries(plan_id, status="raw") if plan_id else []
    all_findings = get_plan_findings(plan_id) if plan_id else []

    # ── Quality filter ──
    valid_findings, discarded_findings = filter_quality(all_findings)

    # ── Contradiction detection ──
    contradictions, contradiction_questions = _detect_contradictions(
        valid_findings, plan_id
    )

    # ── Stats ──
    stats = get_stats()
    question_stats = {}
    if plan_id:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as count FROM questions WHERE plan_id = ? GROUP BY status",
                (plan_id,),
            ).fetchall()
        question_stats = {row["status"]: row["count"] for row in rows}

    # ── Open questions: pending + failed (for interim analysis) ──
    open_questions = []
    if plan_id:
        with get_db() as conn:
            rows = conn.execute(
                """SELECT * FROM questions
                   WHERE plan_id = ? AND status IN ('pending', 'failed')
                   ORDER BY CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 WHEN 'low' THEN 2 ELSE 3 END,
                            created_at""",
                (plan_id,),
            ).fetchall()
        open_questions = [dict(r) for r in rows]

    # ── LLM curation recommendations ──
    recommendations = _generate_curation_recommendations(
        state, raw_entries, valid_findings
    )

    quality = state.get("_quality", {})

    # ── Build prompt ──
    prompt_text = _build_interim_prompt(
        state,
        valid_findings,
        discarded_findings,
        contradictions,
        contradiction_questions,
        open_questions,
        quality,
        hit_max,
        llm_dedup,
    )

    review_data = {
        "message": f"Research iteration {state.get('iteration_count', 0)} complete.",
        "stats": stats,
        "question_progress": question_stats,
        "findings_extracted": len(all_findings),
        "findings_valid": len(valid_findings),
        "findings_discarded": len(discarded_findings),
        "contradictions": contradictions,
        "contradiction_questions": contradiction_questions,
        "llm_dedup": llm_dedup,
        "open_questions": open_questions[:10],
        "quality": quality,
        "recommendations": [
            {
                "id": r.get("id", ""),
                "action": r.get("action", "keep"),
                "reason": r.get("reason", ""),
            }
            for r in recommendations
        ],
        "raw_entries": [
            {
                "id": e["id"],
                "title": e.get("title", "Untitled")[:60],
                "tags": e.get("topic_tags", []),
                "summary": e.get("summary", "")[:120],
            }
            for e in raw_entries[:10]
        ],
        "findings": [
            {
                "id": f["id"],
                "question": f.get("source_question", "Unknown")[:80],
                "summary": f.get("summary", "")[:120],
            }
            for f in valid_findings[:10]
        ],
        "prompt": prompt_text,
    }

    return {"_review_payload": review_data}


def human_review_interrupt(state) -> dict:
    """Suspend the graph and wait for the curator's response.

    Reads the analysis payload built by prepare_review_node.

    NOTE: interrupt() is NOT wrapped in try/except — GraphInterrupt must
    propagate so the graph actually suspends. On resume, LangGraph re-runs
    only this node from the top; interrupt() then returns the user's
    response.
    """
    payload = state.get("_review_payload") or {}
    human_response = interrupt(payload)

    return {"_human_response": human_response, "_review_payload": None}


def curate_knowledge_node(state) -> dict:
    """Process human curation decisions and update the library.

    Does NOT increment iteration_count — consolidation is the only place
    the loop counter advances, so a review cycle advances exactly once.
    """

    response = state.get("_human_response", "continue")
    if not response or response == "continue":
        return {"_human_response": None}

    if response == "stop":
        return {"is_done": True, "_human_response": None}

    if response == "stats":
        return {
            "_human_response": None,
            "_stats_display": get_stats(),
        }

    if response == "findings":
        plan_id = state.get("plan_ref")
        all_findings = get_plan_findings(plan_id) if plan_id else []
        valid, discarded = filter_quality(all_findings)
        contradictions, cq = _detect_contradictions(valid, plan_id)
        return {
            "_human_response": None,
            "_findings_display": {
                "all": all_findings,
                "valid": valid,
                "discarded": discarded,
                "contradictions": contradictions,
                "contradiction_questions": cq,
            },
        }

    if response == "contradictions":
        plan_id = state.get("plan_ref")
        valid, _ = filter_quality(get_plan_findings(plan_id) if plan_id else [])
        contradictions, cq = _detect_contradictions(valid, plan_id)
        return {
            "_human_response": None,
            "_contradictions_display": {
                "pairs": contradictions,
                "questions": cq,
            },
        }

    # Parse commands
    parts = response.strip().split(maxsplit=1)
    if len(parts) < 2:
        return {"_human_response": None}

    command, arg = parts[0].lower(), parts[1].strip()

    if command == "curate":
        entry_id = arg.split()[0]
        try:
            update_knowledge_status(entry_id, "curated")
            return {"_human_response": None}
        except Exception as e:
            return {"_human_response": None, "_error": str(e)}

    elif command == "discard":
        entry_id = arg.split()[0]
        try:
            update_knowledge_status(entry_id, "discarded")
            return {"_human_response": None}
        except Exception as e:
            return {"_human_response": None, "_error": str(e)}

    elif command == "tag":
        parts2 = arg.split(maxsplit=1)
        if len(parts2) < 2:
            return {"_human_response": None}
        entry_id, tags_str = parts2
        new_tags = [t.strip() for t in tags_str.split(",") if t.strip()]
        try:
            update_tags(entry_id, new_tags)
            return {"_human_response": None}
        except Exception as e:
            return {"_human_response": None, "_error": str(e)}

    elif command == "rewind":
        # Rewind sets a flag (presence matters, not value); consolidation
        # consumes it after one backlog pass.
        return {"_rewind_from": 0, "_human_response": None}

    elif command == "retry":
        question_id = arg.strip()
        try:
            set_question_pending(question_id)
            return {"_human_response": None}
        except Exception as e:
            return {"_human_response": None, "_error": f"Retry failed: {e}"}

    elif command == "accept":
        # Accept a failed question with degraded evidence: a placeholder
        # finding records that sources exist but extraction failed, then the
        # question resolves so the plan can complete. (Use 'retry <id>'
        # instead when the failure looks transient.)
        question_id = arg.strip()
        try:
            q = get_question(question_id)
            if not q:
                return {
                    "_human_response": None,
                    "_error": f"No question with id {question_id}",
                }
            if q["status"] != "failed":
                return {
                    "_human_response": None,
                    "_error": f"Question {question_id} is not failed (status: {q['status']})",
                }
            create_degraded_evidence(
                question_id,
                q["question"],
                get_raw_results_for_question(question_id),
                q.get("last_error") or "unknown error",
            )
            set_question_resolved(question_id)
            return {"_human_response": None}
        except Exception as e:
            return {"_human_response": None, "_error": f"Accept failed: {e}"}

    elif command == "gap":
        # Add a specific gap as a research question. The user types "gap 2"
        # to investigate the 2nd gap shown in the interim analysis.
        try:
            plan_id = state.get("plan_ref")
            quality = state.get("_quality", {})
            gaps = quality.get("gaps", [])
            if not plan_id or not gaps:
                return {
                    "_human_response": None,
                    "_error": "No gaps available for this plan",
                }
            gap_idx = int(arg) - 1
            if gap_idx < 0 or gap_idx >= len(gaps):
                return {
                    "_human_response": None,
                    "_error": f"Gap index out of range (1–{len(gaps)})",
                }
            gap_text = gaps[gap_idx]
            if not isinstance(gap_text, str) or not gap_text.strip():
                return {"_human_response": None, "_error": "Gap is empty"}
            from db.knowledge import add_question

            add_question(plan_id, gap_text.strip(), discovered_by="gap_command")
            return {"_human_response": None}
        except ValueError:
            return {
                "_human_response": None,
                "_error": f"Invalid gap index: {arg!r} (use a number, e.g. 'gap 2')",
            }
        except Exception as e:
            return {"_human_response": None, "_error": f"Gap command failed: {e}"}

    return {"_human_response": None}
