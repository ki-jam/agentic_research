"""
Backlog management — dynamically select and potentially re-prioritize questions.

Always reads from the DB (source of truth). LLM only provides suggestions
that are merged with the DB state.
"""

from __future__ import annotations

import json

from config.settings import config
from db.knowledge import (
    get_pending_questions,
    reset_stale_in_progress,
    set_question_in_progress,
    update_question_priority,
)
from utils.llm import parse_json, invoke_with_cost


def manage_backlog_node(state) -> dict:
    """Get all pending questions, optionally re-prioritize, pick the best one."""

    plan_id = state.get("plan_ref")
    if not plan_id:
        return {}

    # Self-heal: reset any question left in_progress by a crashed run so it
    # re-enters the pending pool instead of being reprocessed forever
    # (see reset_stale_in_progress).
    reset_stale_in_progress(plan_id)

    # DB is the source of truth — always get the full list
    pending = get_pending_questions(plan_id)
    if not pending:
        return {}

    # Only re-prioritize if we have enough questions to make it worthwhile
    # (skip on first pass to avoid unnecessary LLM calls)
    iteration = state.get("iteration_count", 0)
    if iteration >= 2:
        pending = _reprioritize(state, pending)

    # Mark the top question as in_progress
    top = pending[0]
    if top.get("id"):
        set_question_in_progress(top["id"])

    return {}


def _reprioritize(state, pending_questions: list) -> list:
    """Use LLM to suggest re-ordering, but always keep all questions.

    The LLM returns a suggested order. We merge it with the DB state
    so no question is ever lost.
    """
    try:
        prompt = f"""You are managing a research queue for: "{state["original_query"]}"

Here are the pending questions (with their current priority):
{json.dumps(pending_questions, indent=2)}

Which question should be researched next? Consider:
- Does it build on already-answered questions?
- Is it a prerequisite for others?

Return ONLY a JSON array of ALL the same question IDs, reordered by
what should be done NEXT. Include every single question — do not omit any.
Example:
[
  {{"id": "q_...", "question": "...", "priority": "high", "_reason": "prerequisite for others"}},
  ...
]"""

        model = config.get_chat_model(task="planning")
        response = invoke_with_cost(model, prompt, task="planning")
        suggested_order = parse_json(response.content)

        if isinstance(suggested_order, list) and len(suggested_order) > 0:
            # Validate: all DB questions must be represented
            db_ids = {q["id"] for q in pending_questions}

            # Filter out LLM entries with unknown IDs (garbage or from other plans)
            suggested_order = [q for q in suggested_order if q.get("id") in db_ids]

            # If LLM dropped any questions, add them back at the end
            llm_ids_after_filter = {q.get("id") for q in suggested_order}
            missing = db_ids - llm_ids_after_filter
            if missing:
                for q in pending_questions:
                    if q["id"] in missing and q not in suggested_order:
                        suggested_order.append(q)

            # Update priorities based on LLM suggestions
            for item in suggested_order:
                qid = item.get("id")
                new_priority = item.get("priority")
                if qid and new_priority:
                    update_question_priority(qid, new_priority)

            return suggested_order

    except Exception:
        pass

    # Fallback: sort by original priority (no LLM)
    priority_order = {"high": 0, "medium": 1, "low": 2}
    pending_questions.sort(
        key=lambda q: priority_order.get(q.get("priority", "medium"), 1)
    )
    return pending_questions
