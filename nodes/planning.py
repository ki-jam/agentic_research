"""
Planning node — stores the user's initial questions in the DB.

No LLM call for question generation from user input.
But we DO use an LLM to refine long queries into focused sub-questions.
"""

from __future__ import annotations

from config.prompts import REFINE_SYSTEM, REFINE_USER
from config.settings import config, workflow_config
from db.knowledge import create_plan, create_questions
from utils.llm import parse_json, invoke_with_cost


def make_plan_node(state, config=None) -> dict:
    """Store user-provided questions in the DB and return plan_ref.

    Also records the LangGraph thread_id on the plan row so sessions can
    be resumed by plan ID (see run.py --resume).

    Pre-flight: checks for similar active plans and returns them in
    '_similar_plans' so run.py can warn the user before starting.

    If plan_ref is already set in state (e.g. via --plan-id), skip
    planning and return immediately — useful for resuming an existing
    plan whose graph was never checkpointed.
    """
    # Fast path: plan_ref already provided → skip planning entirely
    if state.get("plan_ref"):
        from db.knowledge import find_similar_active_plans

        exclude = state.get("plan_ref", "")
        similar = find_similar_active_plans(
            state.get("original_query", ""), exclude_plan_id=exclude
        )
        return {
            "plan_ref": state["plan_ref"],
            "iteration_count": 0,
            "is_done": False,
            "_similar_plans": similar,
        }

    # Use provided questions, or refine the query with LLM
    questions = state.get("questions")
    if not questions:
        questions = _refine_query_into_questions(state["original_query"])

    # Persist to DB
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "")
    plan_id = create_plan(
        query=state["original_query"],
        questions=questions,
        thread_id=thread_id,
    )
    create_questions(plan_id, questions)

    # Pre-flight: find similar active plans (exclude the one we just created)
    from db.knowledge import find_similar_active_plans

    similar = find_similar_active_plans(
        state["original_query"], exclude_plan_id=plan_id
    )

    return {
        "plan_ref": plan_id,
        "iteration_count": 0,
        "is_done": False,
        "_similar_plans": similar,
    }


def _refine_query_into_questions(query: str) -> list[dict]:
    """Use LLM to split a long/broad query into focused, searchable sub-questions.

    This replaces heuristic chunking with semantic understanding.

    Returns list of dicts: [{"question": "...", "priority": "high|medium|low"}]
    """
    # Short queries don't need refinement
    threshold = getattr(workflow_config, "refine_query_threshold", 20)
    if len(query.split()) < threshold:
        return [{"question": query.strip(), "priority": "high"}]

    try:
        prompt = REFINE_USER.format(query=query)
        model = config.get_chat_model(task="planning")
        response = invoke_with_cost(
            model, f"System: {REFINE_SYSTEM}\n\n{prompt}", task="planning"
        )
        data = parse_json(response.content)

        if isinstance(data, list):
            # Validate: ensure each entry has question and priority
            result = []
            for item in data:
                if isinstance(item, dict) and "question" in item:
                    result.append(
                        {
                            "question": item["question"].strip(),
                            "priority": item.get("priority", "medium"),
                        }
                    )
            if result:
                return result

        # Fallback: single question
        return [{"question": query.strip(), "priority": "high"}]
    except Exception:
        return [{"question": query.strip(), "priority": "high"}]
