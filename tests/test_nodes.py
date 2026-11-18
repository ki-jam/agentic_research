"""
Tests for the research agent — nodes, DB, and cache.

Run with:
    cd research_agent && ./../venv/bin/python -m pytest tests/ -v

No live LLM calls and no network access: all LLM interactions use scripted
mocks, and search calls are patched.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

# Ensure we can import from the package root
sys.path.insert(0, os.path.dirname(__file__) + "/..")

# Fake key — tests never call the real Exa API (search functions are patched)
os.environ["EXA_API_KEY"] = "test-key"

from db.knowledge import (
    add_question,
    all_questions_are_resolved,
    cache_check,
    cache_store,
    create_finding,
    create_knowledge_entry,
    cluster_plan_questions,
    create_plan,
    create_questions,
    deduplicate_and_prioritize,
    get_in_progress_question,
    get_pending_question,
    get_pending_questions,
    get_question,
    get_plan_entries,
    get_plan_findings,
    get_plan_stats,
    get_raw_results_for_question,
    get_source_coverage,
    get_stats,
    search_knowledge,
    set_question_discarded,
    set_question_failed,
    set_question_in_progress,
    set_question_resolved,
    store_raw_result,
    update_question_priority,
    update_knowledge_status,
    update_tags,
)
from db.engine import get_db
from cache.search_cache import check_cache, hash_query, store_cache


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def fresh_test_db():
    """Use a test DB so production data is never touched.

    Sets RESEARCH_DB_PATH to knowledge_test.db, drops/recreates it.
    No longer wipes the main data/ directory.
    """
    import db.engine as engine_mod

    # Determine test DB path
    db_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    test_db = os.path.join(db_dir, "knowledge_test.db")

    # Point all DB operations at the test DB
    os.environ["RESEARCH_DB_PATH"] = test_db

    # Clean only the test DB, not production
    if os.path.exists(test_db):
        os.remove(test_db)

    # Re-init so the engine picks up the new path (runs migrations)
    engine_mod.init_db()

    yield

    # Cleanup test DB
    if os.path.exists(test_db):
        os.remove(test_db)
    os.environ.pop("RESEARCH_DB_PATH", None)


# ── DB: Plan & Questions ────────────────────────────────────────────


def test_create_plan_stores_in_db():
    plan_id = create_plan(
        "What is AI?",
        [
            {"question": "Definition", "priority": "high"},
            {"question": "History", "priority": "low"},
        ],
    )
    assert plan_id.startswith("plan_")
    create_questions(
        plan_id,
        [
            {"question": "Definition", "priority": "high"},
            {"question": "History", "priority": "low"},
        ],
    )
    stats = get_plan_stats(plan_id)
    assert "pending" in stats
    assert stats["pending"] == 2


def test_create_questions_from_user_input():
    plan_id = create_plan("Test", [{"question": "Q1", "priority": "high"}])
    ids = create_questions(
        plan_id,
        [
            {"question": "Q1", "priority": "high"},
            {"question": "Q2", "priority": "medium"},
            {"question": "Q3", "priority": "low"},
        ],
    )
    assert len(ids) == 3
    pending = get_pending_questions(plan_id)
    assert len(pending) == 3
    # Verify ordering: high first
    assert pending[0]["priority"] == "high"
    assert pending[1]["priority"] == "medium"
    assert pending[2]["priority"] == "low"


def test_get_pending_question_returns_highest_priority():
    plan_id = create_plan("Test", [])
    create_questions(
        plan_id,
        [
            {"question": "Low priority", "priority": "low"},
            {"question": "High priority", "priority": "high"},
            {"question": "Medium priority", "priority": "medium"},
        ],
    )
    q = get_pending_question(plan_id)
    assert q["question"] == "High priority"


def test_set_question_in_progress_removes_from_pending():
    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q1", "priority": "high"}])
    q = get_pending_question(plan_id)
    set_question_in_progress(q["id"])
    assert get_pending_question(plan_id) is None
    in_progress = get_in_progress_question(plan_id)
    assert in_progress is not None
    assert in_progress["question"] == "Q1"


def test_set_question_resolved():
    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q1", "priority": "high"}])
    q = get_pending_question(plan_id)
    set_question_in_progress(q["id"])
    set_question_resolved(q["id"])
    assert all_questions_are_resolved(plan_id) is True


def test_add_question_grows_the_bank():
    plan_id = create_plan("Test", [{"question": "Original", "priority": "high"}])
    create_questions(plan_id, [{"question": "Original", "priority": "high"}])
    new_id = add_question(plan_id, "Discovered question", discovered_by="extraction")
    assert new_id.startswith("q_")
    stats = get_plan_stats(plan_id)
    assert stats["pending"] == 2


def test_update_question_priority():
    plan_id = create_plan("Test", [])
    create_questions(
        plan_id,
        [
            {"question": "A (low)", "priority": "low"},
            {"question": "B (high)", "priority": "high"},
        ],
    )
    # Promote A to high
    pending = get_pending_questions(plan_id)
    a_id = pending[1]["id"]  # A is at index 1 (low after B)
    update_question_priority(a_id, "high")
    pending = get_pending_questions(plan_id)
    # Both are now high — A should be first due to stable sort
    assert pending[0]["priority"] == "high"
    assert pending[1]["priority"] == "high"


def test_add_question_defaults_to_medium_priority():
    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q", "priority": "high"}])
    add_question(plan_id, "New discovery")
    stats = get_plan_stats(plan_id)
    assert stats["pending"] == 2


# ── DB: Question statuses (terminal vs non-terminal) ────────────────


def test_discarded_question_is_terminal():
    """'discarded' (deduped duplicate) questions must NOT block readiness.

    Regression: dedup runs at every review, so treating 'discarded' as
    open made plans permanently un-ready.
    """
    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q1", "priority": "high"}])
    q = get_pending_question(plan_id)
    with get_db() as conn:
        conn.execute("UPDATE questions SET status='discarded' WHERE id=?", (q["id"],))
    assert all_questions_are_resolved(plan_id) is True


def test_failed_question_blocks_readiness():
    """A 'failed' question (LLM error) is NON-terminal — it blocks readiness
    and keeps last_error for the interim analysis screen.
    """
    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q1", "priority": "high"}])
    q = get_pending_question(plan_id)
    set_question_failed(q["id"], "LLM unavailable")
    assert all_questions_are_resolved(plan_id) is False
    stats = get_plan_stats(plan_id)
    assert stats.get("failed") == 1
    with get_db() as conn:
        row = conn.execute(
            "SELECT last_error FROM questions WHERE id=?", (q["id"],)
        ).fetchone()
    assert row["last_error"] == "LLM unavailable"


def test_retry_resets_failed_question_to_pending():
    """The 'retry <id>' command resets a failed question to pending so the
    loop can re-process it instead of being permanently blocked.
    """
    from db.knowledge import set_question_pending

    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q1", "priority": "high"}])
    q = get_pending_question(plan_id)
    set_question_failed(q["id"], "LLM error")
    assert all_questions_are_resolved(plan_id) is False

    # Retry resets to pending
    set_question_pending(q["id"])
    with get_db() as conn:
        row = conn.execute(
            "SELECT status, last_error FROM questions WHERE id=?", (q["id"],)
        ).fetchone()
    assert row["status"] == "pending"
    assert row["last_error"] is None
    # Now the question is re-processable (pending, not failed)


def test_add_question_ids_never_collide():
    """uuid4 IDs: adding the same question twice must not raise a
    primary-key collision (the old hash-based IDs could collide).

    Uses a user source: the insert-time similarity guard suppresses
    duplicate AUTO questions (see the guard tests below).
    """
    plan_id = create_plan("Test", [])
    i1 = add_question(plan_id, "Same question text", discovered_by="user")
    i2 = add_question(plan_id, "Same question text", discovered_by="user")
    assert i1 != i2
    assert len(get_pending_questions(plan_id)) == 2


# ── Regression: plan scoping (no cross-plan leakage) ────────────────


def test_findings_do_not_leak_across_plans():
    plan_a = create_plan("Plan A", [])
    plan_b = create_plan("Plan B", [])
    create_questions(plan_a, [{"question": "QA?", "priority": "high"}])
    create_questions(plan_b, [{"question": "QB?", "priority": "high"}])
    qa = get_pending_question(plan_a)
    rid = store_raw_result(qa["id"], plan_a, "QA?", "exa", "content", "Title A")
    create_finding(
        question_id=qa["id"],
        raw_result_id=rid,
        knowledge_entry_id=None,
        facts=[{"claim": "c", "confidence": "high"}],
        summary="Finding about A",
        tags=[],
    )
    assert len(get_plan_findings(plan_a)) == 1
    assert get_plan_findings(plan_b) == []
    assert get_plan_entries(plan_b) == []


def test_search_knowledge_is_plan_scoped():
    plan_a = create_plan("Plan A", [])
    plan_b = create_plan("Plan B", [])
    create_questions(plan_a, [{"question": "QA?", "priority": "high"}])
    entry = create_knowledge_entry(
        title="Zorblatt entry",
        summary="uniquezorblatt summary",
        topic_tags=["z"],
        full_text_path="",
        source_url="",
    )
    qa = get_pending_question(plan_a)
    create_finding(
        question_id=qa["id"],
        raw_result_id=None,
        knowledge_entry_id=entry,
        facts=[],
        summary="Finding about A",
        tags=[],
    )
    assert len(search_knowledge("uniquezorblatt", status="raw", plan_id=plan_a)) == 1
    assert search_knowledge("uniquezorblatt", status="raw", plan_id=plan_b) == []


def test_has_actual_search_results_is_plan_scoped():
    """Citation-discovery results (exa-answer*) do NOT count as actual search,
    and one plan's results must not satisfy another plan's enforcement gate.
    """
    from nodes.consolidation import _has_actual_search_results

    plan_a = create_plan("A", [])
    plan_b = create_plan("B", [])
    plan_c = create_plan("C", [])

    qa_id = create_questions(plan_a, [{"question": "QA?", "priority": "high"}])[0]
    store_raw_result(qa_id, plan_a, "QA?", "exa", "content", "T")
    store_raw_result(qa_id, plan_a, "QA?", "exa-answer", "content", "T")

    qc_id = create_questions(plan_c, [{"question": "QC?", "priority": "high"}])[0]
    store_raw_result(qc_id, plan_c, "QC?", "exa-answer", "content", "T")

    assert _has_actual_search_results(plan_a) is True
    assert _has_actual_search_results(plan_b) is False
    assert _has_actual_search_results(plan_c) is False  # only exa-answer results


# ── DB: Cache ───────────────────────────────────────────────────────


def test_cache_store_and_check():
    results = [{"title": "Article 1", "url": "https://example.com", "content": "Text"}]
    cache_store("abc123", "Test query", "exa", results)
    row = cache_check("abc123")
    assert row is not None
    assert row["query"] == "Test query"


def test_cache_hash_is_deterministic():
    assert hash_query("Hello World") == hash_query("Hello World")
    assert hash_query("Hello World") != hash_query("Goodbye World")


# ── DB: Raw Results ─────────────────────────────────────────────────


def test_store_and_retrieve_raw_results():
    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q?", "priority": "high"}])
    q = get_pending_question(plan_id)
    set_question_in_progress(q["id"])
    rid = store_raw_result(q["id"], plan_id, "Q?", "exa", "Content here", "Title")
    assert rid > 0
    results = get_raw_results_for_question(q["id"])
    assert len(results) == 1
    assert results[0]["content"] == "Content here"


# ── DB: Knowledge Entries ───────────────────────────────────────────


def test_update_knowledge_status_and_tags():
    entry_id = create_knowledge_entry(
        title="Test Entry",
        summary="A test",
        topic_tags=["tag1"],
        full_text_path="/tmp/test.txt",
        source_url="https://example.com",
    )
    entry = update_knowledge_status(entry_id, "curated")
    assert entry["status"] == "curated"
    entry = update_tags(entry_id, ["tag2", "tag3"])
    assert "tag2" in entry["topic_tags"]
    assert "tag3" in entry["topic_tags"]


# ── DB: Findings ────────────────────────────────────────────────────


def test_create_finding():
    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q?", "priority": "high"}])
    q = get_pending_question(plan_id)
    rid = store_raw_result(q["id"], plan_id, "Q?", "exa", "Content", "Title")
    fid = create_finding(
        question_id=q["id"],
        raw_result_id=rid,
        knowledge_entry_id=None,
        facts=[{"claim": "Fact 1", "confidence": "high"}],
        summary="Summary text",
        tags=["tag1"],
    )
    assert fid.startswith("finding_")
    findings = get_plan_findings(plan_id)
    assert len(findings) == 1
    assert findings[0]["summary"] == "Summary text"


# ── DB: Stats ───────────────────────────────────────────────────────


def test_get_stats():
    stats = get_stats()
    assert "by_status" in stats
    assert "top_tags" in stats
    assert "total" in stats


# ── Cache Module ────────────────────────────────────────────────────


def test_cache_check_returns_none_when_empty():
    assert check_cache("nonexistent query") is None


def test_cache_store_and_retrieve():
    results = [{"title": "Cached", "url": "https://x.com", "content": "Data"}]
    store_cache("cached query", results, engine="exa")
    cached = check_cache("cached query")
    assert cached is not None
    assert len(cached) == 1
    assert cached[0]["title"] == "Cached"


def test_cache_miss_on_different_query():
    store_cache("query A", [{"title": "A"}], engine="exa")
    assert check_cache("query B") is None


# ── DB: Cache TTL ───────────────────────────────────────────────────
#
# The TTL used to be hardcoded to 7 days in cache_check(). It is now
# search_config.cache_ttl_days (env SEARCH_CACHE_TTL_DAYS, default 30;
# 0 = never expire).


def test_cache_ttl_expires_entries_older_than_ttl(monkeypatch):
    from config.settings import search_config

    monkeypatch.setattr(search_config, "cache_ttl_days", 30)
    cache_store("ttl-old", "Old query", "exa", [{"title": "T"}])
    with get_db() as conn:
        conn.execute(
            "UPDATE search_cache SET cached_at = datetime('now', '-31 days') "
            "WHERE query_hash = 'ttl-old'"
        )
    assert cache_check("ttl-old") is None


def test_cache_ttl_zero_never_expires(monkeypatch):
    from config.settings import search_config

    monkeypatch.setattr(search_config, "cache_ttl_days", 0)
    cache_store("ttl-never", "Never query", "exa", [{"title": "T"}])
    with get_db() as conn:
        conn.execute(
            "UPDATE search_cache SET cached_at = datetime('now', '-1000 days') "
            "WHERE query_hash = 'ttl-never'"
        )
    row = cache_check("ttl-never")
    assert row is not None
    assert row["query"] == "Never query"


def test_cache_ttl_fresh_entry_hits(monkeypatch):
    from config.settings import search_config

    monkeypatch.setattr(search_config, "cache_ttl_days", 30)
    cache_store("ttl-fresh", "Fresh query", "exa", [{"title": "T"}])
    row = cache_check("ttl-fresh")
    assert row is not None


# ── Planning Node ───────────────────────────────────────────────────


def test_make_plan_node():
    from nodes.planning import make_plan_node

    state = {
        "original_query": "What is quantum computing?",
        "questions": [
            {"question": "Definition", "priority": "high"},
            {"question": "Applications", "priority": "medium"},
        ],
    }
    result = make_plan_node(state)
    assert "plan_ref" in result
    assert result["plan_ref"].startswith("plan_")
    assert "iteration_count" in result
    assert result["iteration_count"] == 0


def test_make_plan_node_records_thread_id():
    """The LangGraph thread_id from config is persisted on the plan row."""
    from nodes.planning import make_plan_node

    state = {
        "original_query": "What is quantum computing?",
        "questions": [{"question": "Definition", "priority": "high"}],
    }
    config = {"configurable": {"thread_id": "thread-abc-123"}}
    result = make_plan_node(state, config)
    with get_db() as conn:
        row = conn.execute(
            "SELECT thread_id FROM plans WHERE id=?", (result["plan_ref"],)
        ).fetchone()
    assert row["thread_id"] == "thread-abc-123"


def test_make_plan_node_fallback_question():
    from nodes.planning import make_plan_node

    state = {"original_query": "Solo query"}
    result = make_plan_node(state)
    assert "plan_ref" in result
    # Should have created a question from the query (short query → no LLM call)
    stats = get_plan_stats(result["plan_ref"])
    assert stats["pending"] == 1


# ── Pre-flight: find_similar_active_plans ───────────────────────────


def test_find_similar_active_plans_no_plans():
    """With no plans, returns an empty list."""
    from db.knowledge import find_similar_active_plans

    assert find_similar_active_plans("What is quantum computing?") == []


def test_find_similar_active_plans_no_match():
    """Completely different topics yield no matches."""
    plan_a = create_plan("Topic A", [{"question": "Q1", "priority": "high"}])
    # Add a second question to keep it active (not all resolved)
    add_question(plan_a, "Follow-up on topic A", discovered_by="extraction")
    from db.knowledge import find_similar_active_plans

    results = find_similar_active_plans("What is the capital of France?")
    assert results == []


def test_find_similar_active_plans_finds_similar():
    """Similar queries are detected and returned with similarity scores."""
    plan_id = create_plan(
        "Deep learning memory optimization",
        [
            {"question": "Memory-efficient frameworks", "priority": "high"},
        ],
    )
    add_question(plan_id, "Follow-up", discovered_by="extraction")
    from db.knowledge import find_similar_active_plans

    results = find_similar_active_plans(
        "What are the most memory-efficient deep learning libraries"
    )
    assert len(results) == 1
    assert results[0]["plan_id"] == plan_id
    assert results[0]["similarity"] > 0


def test_find_similar_active_plans_filters_resolved():
    """Plans where all questions are resolved are excluded."""
    plan_id = create_plan(
        "Deep learning memory optimization",
        [
            {"question": "Memory-efficient frameworks", "priority": "high"},
        ],
    )
    # Resolve the initial question and the follow-up
    set_question_resolved(f"q_{plan_id}_0")
    # The follow-up was added by add_question — need to find it
    pending = get_pending_questions(plan_id)
    if pending:
        set_question_resolved(pending[0]["id"])
    from db.knowledge import find_similar_active_plans

    results = find_similar_active_plans(
        "What are the most memory-efficient deep learning libraries"
    )
    assert results == []


def test_find_similar_active_plans_multiple_matches_sorted():
    """Multiple similar plans are returned sorted by similarity."""
    plan_close = create_plan(
        "Memory-efficient deep learning frameworks",
        [
            {"question": "Memory frameworks?", "priority": "high"},
        ],
    )
    add_question(plan_close, "Follow-up", discovered_by="extraction")
    plan_far = create_plan(
        "Quantum computing hardware architectures",
        [
            {"question": "What is quantum?", "priority": "high"},
        ],
    )
    add_question(plan_far, "Follow-up", discovered_by="extraction")
    from db.knowledge import find_similar_active_plans

    results = find_similar_active_plans(
        "What are the most memory-efficient deep learning libraries"
    )
    # Should find at least the close match
    assert len(results) >= 1
    assert results[0]["plan_id"] == plan_close
    # Close match should have higher similarity than far match
    if len(results) == 2:
        assert results[0]["similarity"] > results[1]["similarity"]


def test_make_plan_node_returns_similar_plans():
    """make_plan_node includes _similar_plans in its return value."""
    from nodes.planning import make_plan_node

    # Create a plan first so there's something to match
    existing_plan = create_plan(
        "Cross-compile ML compilers and frameworks",
        [{"question": "Q1", "priority": "high"}],
    )
    add_question(existing_plan, "Follow-up on compilation", discovered_by="extraction")

    state = {
        "original_query": "Cross-compile ML compilers for machine learning",
        "questions": [{"question": "Cross-compile?", "priority": "high"}],
    }
    result = make_plan_node(state)
    assert "_similar_plans" in result
    # Should find the existing plan as similar
    assert len(result["_similar_plans"]) >= 1


def test_make_plan_node_skip_when_plan_ref_set():
    """When plan_ref is already in state (e.g. --plan-id), skip planning
    and return early — no new plan is created. The plan_ref flows through
    from the input state (LangGraph merges node output with existing state).
    """
    from nodes.planning import make_plan_node

    with get_db() as conn:
        initial_count = conn.execute("SELECT COUNT(*) FROM plans").fetchone()[0]
    result = make_plan_node({"plan_ref": "plan_2249", "original_query": "Test"})
    assert result["plan_ref"] == "plan_2249"
    assert result["iteration_count"] == 0
    assert result["is_done"] is False
    # No new plan created
    with get_db() as conn:
        new_count = conn.execute("SELECT COUNT(*) FROM plans").fetchone()[0]
    assert new_count == initial_count


def test_chunk_query_short():
    from nodes.planning import _refine_query_into_questions

    # Short query should not be refined (no LLM call)
    result = _refine_query_into_questions("What is quantum computing?")
    assert len(result) == 1
    assert "quantum computing" in result[0]["question"].lower()


def test_chunk_query_long(monkeypatch):
    """Long query is refined by the (mocked) LLM into focused sub-questions."""
    from nodes.planning import _refine_query_into_questions
    from config import settings

    class MockModel:
        def invoke(self, prompt, **kwargs):
            return type(
                "R",
                (),
                {
                    "content": json.dumps(
                        [
                            {
                                "question": "Which ML compilers and runtimes exist?",
                                "priority": "high",
                            },
                            {
                                "question": "How do they compare on GPU and TPU?",
                                "priority": "high",
                            },
                            {
                                "question": "What are the benchmark trade-offs?",
                                "priority": "medium",
                            },
                        ]
                    )
                },
            )()

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": MockModel(),
    )
    result = _refine_query_into_questions(
        "What are the major compilers and runtimes for ML? Compare TVM, MLIR, IREE, XLA for GPU and TPU acceleration. Look at benchmarks and performance trade-offs between frameworks."
    )
    assert len(result) == 3
    assert all("priority" in q for q in result)
    assert all(q["priority"] in ("high", "medium", "low") for q in result)


def test_chunk_query_fallback(monkeypatch):
    """A long unstructured query still yields usable (mocked) questions."""
    from nodes.planning import _refine_query_into_questions
    from config import settings

    class MockModel:
        def invoke(self, prompt, **kwargs):
            return type(
                "R",
                (),
                {
                    "content": json.dumps(
                        [
                            {
                                "question": "How do compilers map neural networks to hardware?",
                                "priority": "high",
                            },
                        ]
                    )
                },
            )()

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": MockModel(),
    )
    result = _refine_query_into_questions(
        "I want to know about everything related to compiling neural networks to different hardware types including graphics processing units tensor processing units neural processing units field programmable gate arrays and application specific integrated circuits with a focus on performance benchmarks accuracy trade offs and framework comparisons"
    )
    assert len(result) >= 1
    combined = " ".join(q["question"] for q in result).lower()
    assert any(
        kw in combined for kw in ["compile", "ml", "hardware", "framework", "compiler"]
    )


# ── Backlog Node ────────────────────────────────────────────────────


def test_manage_backlog_node():
    from nodes.backlog import manage_backlog_node

    plan_id = create_plan("Test", [])
    create_questions(
        plan_id,
        [
            {"question": "Q1", "priority": "high"},
            {"question": "Q2", "priority": "low"},
        ],
    )
    manage_backlog_node({"plan_ref": plan_id, "iteration_count": 0})
    # Should pick high priority question
    in_progress = get_in_progress_question(plan_id)
    assert in_progress is not None
    assert in_progress["question"] == "Q1"


def test_manage_backlog_no_plan_returns_empty():
    from nodes.backlog import manage_backlog_node

    result = manage_backlog_node({})
    assert result == {}


def test_manage_backlog_no_pending_returns_empty():
    from nodes.backlog import manage_backlog_node

    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q1", "priority": "high"}])
    set_question_resolved(f"q_{plan_id}_0")
    result = manage_backlog_node({"plan_ref": plan_id})
    assert result == {}


# ── LLM Question Loss Safety Net ────────────────────────────────────


def test_reprioritize_recovers_dropped_questions(monkeypatch):
    """Mock LLM drops Q2 — safety net should add it back."""
    from nodes.backlog import _reprioritize
    from config import settings

    plan_id = create_plan("Test", [])
    create_questions(
        plan_id,
        [
            {"question": "Q1", "priority": "high"},
            {"question": "Q2", "priority": "medium"},
        ],
    )

    pending = get_pending_questions(plan_id)
    q1_id = pending[0]["id"]  # high priority, first in list

    # Mock LLM that only returns Q1 (drops Q2)
    class MockModel:
        def invoke(self, prompt, **kwargs):
            return type(
                "Response",
                (),
                {
                    "content": json.dumps(
                        [
                            {"id": q1_id, "question": "Q1", "priority": "high"},
                            # Q2 intentionally omitted!
                        ]
                    )
                },
            )()

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": MockModel(),
    )

    state = {"original_query": "Test", "iteration_count": 5}
    result = _reprioritize(state, pending)

    # Safety net: all questions must be present
    result_ids = {q["id"] for q in result}
    db_ids = {q["id"] for q in pending}
    assert db_ids.issubset(result_ids), f"Dropped questions: {db_ids - result_ids}"
    assert len(result) == len(pending)


# ── Consolidation Node ──────────────────────────────────────────────


def test_consolidation_route_backlog(monkeypatch):
    """No raw results yet → keep looping, regardless of the LLM verdict."""
    from nodes.consolidation import consolidation_node
    from config import settings

    class MockModel:
        def invoke(self, prompt, **kwargs):
            return type(
                "R",
                (),
                {
                    "content": json.dumps(
                        {
                            "ready": False,
                            "gaps": ["thin coverage"],
                            "confidence": "low",
                        }
                    )
                },
            )()

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": MockModel(),
    )
    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q1", "priority": "high"}])
    result = consolidation_node({"plan_ref": plan_id, "iteration_count": 0})
    assert result["_route"] == "manage_backlog"


def test_consolidation_route_write_findings_all_resolved(monkeypatch):
    """All resolved + LLM says not ready + no search data → keep looping."""
    from nodes.consolidation import consolidation_node
    from config import settings

    class MockModel:
        def invoke(self, prompt, **kwargs):
            return type(
                "R",
                (),
                {
                    "content": json.dumps(
                        {
                            "ready": False,
                            "gaps": ["thin coverage"],
                            "confidence": "low",
                        }
                    )
                },
            )()

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": MockModel(),
    )
    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q1", "priority": "high"}])
    set_question_resolved(f"q_{plan_id}_0")
    result = consolidation_node({"plan_ref": plan_id, "iteration_count": 0})
    assert result["_route"] == "manage_backlog"


def test_consolidation_route_write_findings_when_ready(monkeypatch):
    """All resolved + LLM says ready + enough findings + real search → stop."""
    from nodes.consolidation import consolidation_node
    from config import settings
    from config.settings import workflow_config

    class MockModel:
        def invoke(self, prompt, **kwargs):
            return type(
                "R",
                (),
                {
                    "content": json.dumps(
                        {
                            "ready": True,
                            "gaps": [],
                            "confidence": "high",
                        }
                    )
                },
            )()

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": MockModel(),
    )
    monkeypatch.setattr(workflow_config, "min_findings", 1)
    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q1", "priority": "high"}])
    q = get_pending_question(plan_id)
    set_question_resolved(q["id"])
    rid = store_raw_result(q["id"], plan_id, "Q1", "exa", "content", "Title")
    create_finding(
        question_id=q["id"],
        raw_result_id=rid,
        knowledge_entry_id=None,
        facts=[{"claim": "c", "confidence": "high"}],
        summary="A specific finding about Q1",
        tags=[],
    )
    result = consolidation_node({"plan_ref": plan_id, "iteration_count": 1})
    assert result["_route"] == "write_findings"


def test_consolidation_route_max_iterations(monkeypatch):
    """At the iteration budget → route to the human review (interim analysis)."""
    from nodes.consolidation import consolidation_node
    from config import settings
    from config.settings import workflow_config

    class MockModel:
        def invoke(self, prompt, **kwargs):
            return type(
                "R",
                (),
                {
                    "content": json.dumps(
                        {
                            "ready": False,
                            "gaps": ["thin coverage"],
                            "confidence": "low",
                        }
                    )
                },
            )()

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": MockModel(),
    )
    monkeypatch.setattr(workflow_config, "max_iterations", 1)
    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q1", "priority": "high"}])
    # At max iterations → routes to human_review for interim analysis
    result = consolidation_node({"plan_ref": plan_id, "iteration_count": 0})
    assert result["_route"] == "human_review"
    assert result["_hit_max_iterations"] is True


def test_consolidation_increments_iteration(monkeypatch):
    from nodes.consolidation import consolidation_node
    from config import settings

    class MockModel:
        def invoke(self, prompt, **kwargs):
            return type(
                "R",
                (),
                {
                    "content": json.dumps(
                        {
                            "ready": False,
                            "gaps": ["thin coverage"],
                            "confidence": "low",
                        }
                    )
                },
            )()

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": MockModel(),
    )
    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q1", "priority": "high"}])
    result = consolidation_node({"plan_ref": plan_id, "iteration_count": 2})
    assert result["iteration_count"] == 3


def test_consolidation_gaps_become_questions(monkeypatch):
    """Empty bank + not-ready DoD → gaps are promoted to questions so the
    loop can actually address them instead of spinning (self-correction).
    """
    from nodes.consolidation import consolidation_node
    from config import settings

    class MockModel:
        def invoke(self, prompt, **kwargs):
            return type(
                "R",
                (),
                {
                    "content": json.dumps(
                        {
                            "ready": False,
                            "gaps": ["gap one", "gap two"],
                            "confidence": "low",
                        }
                    )
                },
            )()

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": MockModel(),
    )
    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q1", "priority": "high"}])
    set_question_resolved(f"q_{plan_id}_0")
    result = consolidation_node({"plan_ref": plan_id, "iteration_count": 0})
    assert result["_route"] == "manage_backlog"
    pending = get_pending_questions(plan_id)
    assert {q["question"] for q in pending} == {"gap one", "gap two"}


# ── Gap → question shape fix (_gap_to_question) ────────────────────
#
# The readiness LLM emits gaps as statements ("Missing TGI performance
# details"), and the old code stored those verbatim as "questions" —
# the search node then fed the statement straight into a search query
# (live example: q_plan_f9b12648_6ad4d299).


def test_gap_to_question_wraps_missing_statements():
    from nodes.consolidation import _gap_to_question

    assert (
        _gap_to_question("Missing TGI current performance details")
        == "What do we still need to know about TGI current performance details?"
    )
    assert (
        _gap_to_question("No head-to-head benchmarks for SGLang vs vLLM")
        == "What do we still need to know about head-to-head benchmarks for SGLang vs vLLM?"
    )
    # Acronyms and camelCase must survive the wrap (the old code
    # lowercased the first letter: "TGI" → "tgi").
    assert (
        _gap_to_question("Missing Tensorflow coverage")
        == "What do we still need to know about Tensorflow coverage?"
    )


def test_gap_to_question_passes_through_questions_and_noun_phrases():
    from nodes.consolidation import _gap_to_question

    assert (
        _gap_to_question("How does vLLM schedule requests?")
        == "How does vLLM schedule requests?"
    )
    # Noun phrases already work as search queries — unchanged.
    assert (
        _gap_to_question("API compatibility and ecosystem integrations")
        == "API compatibility and ecosystem integrations"
    )


def test_consolidation_gap_statement_becomes_question_in_bank(monkeypatch):
    from nodes.consolidation import consolidation_node
    from config import settings

    class MockModel:
        def invoke(self, prompt, **kwargs):
            return type(
                "R",
                (),
                {
                    "content": json.dumps(
                        {
                            "ready": False,
                            "gaps": ["Missing TGI current performance details"],
                            "confidence": "low",
                        }
                    )
                },
            )()

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": MockModel(),
    )
    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q1", "priority": "high"}])
    set_question_resolved(f"q_{plan_id}_0")
    result = consolidation_node({"plan_ref": plan_id, "iteration_count": 0})
    assert result["_route"] == "manage_backlog"
    pending = get_pending_questions(plan_id)
    assert [q["question"] for q in pending] == [
        "What do we still need to know about TGI current performance details?"
    ]


def test_consolidation_evaluation_failure_falls_back_to_heuristic(monkeypatch):
    """If the readiness LLM is down, consolidation must not crash — it falls
    back to a heuristic (ready iff all resolved + enough findings).
    """
    from nodes.consolidation import consolidation_node
    from config import settings
    from config.settings import workflow_config

    class BoomModel:
        def invoke(self, prompt, **kwargs):
            raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": BoomModel(),
    )
    monkeypatch.setattr(workflow_config, "min_findings", 1)
    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q1", "priority": "high"}])
    q = get_pending_question(plan_id)
    set_question_resolved(q["id"])
    rid = store_raw_result(q["id"], plan_id, "Q1", "exa", "content", "Title")
    create_finding(
        question_id=q["id"],
        raw_result_id=rid,
        knowledge_entry_id=None,
        facts=[{"claim": "c", "confidence": "high"}],
        summary="A specific finding about Q1",
        tags=[],
    )
    result = consolidation_node({"plan_ref": plan_id, "iteration_count": 1})
    # Heuristic says ready (all resolved + 1 finding >= min 1)
    assert result["_quality"].get("heuristic") is True
    assert result["_route"] == "write_findings"


def _ready_model():
    class MockModel:
        def invoke(self, prompt, **kwargs):
            return type(
                "R",
                (),
                {
                    "content": json.dumps(
                        {"ready": True, "gaps": [], "confidence": "high"}
                    )
                },
            )()

    return MockModel()


def _seed_ready_plan():
    """Plan with one resolved question + one real search result + one finding
    — everything the ready route needs except the coverage floor."""
    from db.knowledge import (
        create_finding,
        create_plan,
        create_questions,
        store_raw_result,
    )

    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q1", "priority": "high"}])
    qid = f"q_{plan_id}_0"
    set_question_resolved(qid)
    rid = store_raw_result(qid, plan_id, "Q1", "exa", "content", "Title")
    create_finding(
        question_id=qid,
        raw_result_id=rid,
        knowledge_entry_id=None,
        facts=[{"claim": "c", "confidence": "high"}],
        summary="A specific finding about Q1",
        tags=[],
    )
    return plan_id


def test_consolidation_coverage_floor_blocks_open_high_priority(monkeypatch):
    """LLM says ready + enough findings, but a high-priority question is
    still open → auto-stop is blocked and the session goes to human_review
    (the user can stop → interim report, or continue)."""
    from nodes.consolidation import consolidation_node
    from config import settings
    from config.settings import workflow_config

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": _ready_model(),
    )
    monkeypatch.setattr(workflow_config, "min_findings", 1)
    plan_id = _seed_ready_plan()
    # Second, high-priority question left open
    qid2 = add_question(plan_id, "Q2")
    update_question_priority(qid2, "high")

    result = consolidation_node({"plan_ref": plan_id, "iteration_count": 1})
    assert result["_route"] == "human_review"
    assert result["_coverage"]["total"] == 2
    assert result["_coverage"]["high_open"] == 1


def test_consolidation_coverage_floor_blocks_low_resolution(monkeypatch):
    """No open high-priority questions, but only 1/3 of the bank is closed
    (below the 0.5 floor) → auto-stop is blocked."""
    from nodes.consolidation import consolidation_node
    from config import settings
    from config.settings import workflow_config

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": _ready_model(),
    )
    monkeypatch.setattr(workflow_config, "min_findings", 1)
    plan_id = _seed_ready_plan()
    add_question(plan_id, "Q2")  # medium by default
    add_question(plan_id, "Q3")  # medium by default
    # 1 closed of 3 = 0.33 < 0.5

    result = consolidation_node({"plan_ref": plan_id, "iteration_count": 1})
    assert result["_route"] == "human_review"
    assert result["_coverage"]["closed"] == 1
    assert result["_coverage"]["high_open"] == 0


def test_consolidation_coverage_floor_passes_at_threshold(monkeypatch):
    """1/2 closed (== the 0.5 default) with no open high-priority → the
    ready route proceeds as before."""
    from nodes.consolidation import consolidation_node
    from config import settings
    from config.settings import workflow_config

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": _ready_model(),
    )
    monkeypatch.setattr(workflow_config, "min_findings", 1)
    plan_id = _seed_ready_plan()
    add_question(plan_id, "Q2")  # medium by default
    # 1 closed of 2 = 0.5 >= 0.5

    result = consolidation_node({"plan_ref": plan_id, "iteration_count": 1})
    assert result["_route"] == "write_findings"


def test_consolidation_coverage_gate_can_be_disabled(monkeypatch):
    """READY_COVERAGE_GATE=false restores the pre-gate behavior: open
    high-priority questions no longer block auto-stop."""
    from nodes.consolidation import consolidation_node
    from config import settings
    from config.settings import workflow_config

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": _ready_model(),
    )
    monkeypatch.setattr(workflow_config, "min_findings", 1)
    monkeypatch.setattr(workflow_config, "ready_coverage_gate", False)
    plan_id = _seed_ready_plan()
    qid2 = add_question(plan_id, "Q2")
    update_question_priority(qid2, "high")

    result = consolidation_node({"plan_ref": plan_id, "iteration_count": 1})
    assert result["_route"] == "write_findings"


def test_readiness_prompt_uses_scope_block(monkeypatch):
    """The readiness prompt embeds the plan's scope block and no longer
    hard-codes topic-specific systems (vLLM/SGLang/TGI/Kafka)."""
    from nodes.consolidation import _evaluate_readiness
    from config import settings
    from utils.scope import build_scope_block

    captured = {}

    class CapModel:
        def invoke(self, prompt, **kwargs):
            captured["prompt"] = prompt
            return type(
                "R",
                (),
                {
                    "content": json.dumps(
                        {"ready": False, "gaps": [], "confidence": "low"}
                    )
                },
            )()

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": CapModel(),
    )

    q = "Compare vLLM, SGLang, and TGI for LLM serving"
    plan_id = create_plan(q, [], scope=build_scope_block(q))
    _evaluate_readiness({"plan_ref": plan_id, "original_query": q})

    prompt = captured["prompt"]
    assert "RESEARCH SCOPE (enforced):" in prompt
    assert "Named systems in scope: vLLM, SGLang, TGI" in prompt
    # The hard-coded topic rules are gone (no system names in the rules)
    assert "Kafka" not in prompt
    assert '"vLLM", "SGLang", "TGI"' not in prompt


def test_consolidation_clears_rewind_flag_after_one_pass(monkeypatch):
    """Regression (P0-1): _rewind_from must be consumed by consolidation.

    Without this fix, _rewind_from is never cleared, so _routing_logic always
    returns 'manage_backlog', pinning the graph in an infinite loop. After one
    backlog pass, consolidation must return {'_rewind_from': None}.
    """
    from nodes.consolidation import consolidation_node
    from config import settings
    from config.settings import workflow_config

    class MockModel:
        def invoke(self, prompt, **kwargs):
            return type(
                "R",
                (),
                {
                    "content": json.dumps(
                        {
                            "ready": True,
                            "gaps": [],
                            "confidence": "high",
                        }
                    )
                },
            )()

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": MockModel(),
    )
    monkeypatch.setattr(workflow_config, "min_findings", 1)
    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q1", "priority": "high"}])
    q = get_pending_question(plan_id)
    set_question_resolved(q["id"])
    rid = store_raw_result(q["id"], plan_id, "Q1", "exa", "content", "Title")
    create_finding(
        question_id=q["id"],
        raw_result_id=rid,
        knowledge_entry_id=None,
        facts=[{"claim": "c", "confidence": "high"}],
        summary="A specific finding about Q1",
        tags=[],
    )
    # Simulate the state after a rewind command: _rewind_from is set
    state = {
        "plan_ref": plan_id,
        "iteration_count": 1,
        "_rewind_from": 5,
    }
    result = consolidation_node(state)
    # The flag must be cleared
    assert result.get("_rewind_from") is None, (
        "_rewind_from was not cleared by consolidation — infinite loop risk"
    )


def test_consolidation_adds_gaps_even_with_pending_questions(monkeypatch):
    """Regression fix: DoD gaps should be added as questions even when the
    pending queue isn't empty. The old code had `not get_pending_questions(plan_id)`
    which meant gaps were never added if there were already pending questions.

    This is critical: the LLM often identifies important gaps (e.g. "no TensorFlow
    coverage") that need to be investigated, but the old code silently ignored them.
    """
    from nodes.consolidation import consolidation_node
    from config import settings

    class MockModel:
        def invoke(self, prompt, **kwargs):
            return type(
                "R",
                (),
                {
                    "content": json.dumps(
                        {
                            "ready": False,
                            "gaps": [
                                "No TensorFlow coverage",
                                "Missing JAX comparison",
                            ],
                            "confidence": "high",
                        }
                    )
                },
            )()

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": MockModel(),
    )
    plan_id = create_plan("Test", [])
    # Add multiple pending questions — old code would skip gap creation
    create_questions(
        plan_id,
        [
            {"question": "Q1", "priority": "high"},
            {"question": "Q2", "priority": "medium"},
            {"question": "Q3", "priority": "low"},
        ],
    )
    assert len(get_pending_questions(plan_id)) == 3

    result = consolidation_node({"plan_ref": plan_id, "iteration_count": 0})
    assert result["_route"] == "manage_backlog"

    # New questions should have been added despite the queue not being empty
    all_questions = get_pending_questions(plan_id)
    added_texts = [
        q["question"] for q in all_questions if q.get("discovered_by") == "dod_gaps"
    ]
    assert len(added_texts) == 2, (
        f"Expected 2 gap-questions, got {len(added_texts)}: {added_texts}"
    )
    # Gap statements arrive as questions now (see _gap_to_question), so
    # assert on the topic, not the raw statement wording.
    assert "TensorFlow coverage" in added_texts[0]
    assert "JAX comparison" in added_texts[1]


def test_consolidation_gaps_limit_to_5_per_pass(monkeypatch):
    """Gaps should be capped at 5 per pass to prevent question spam."""
    from nodes.consolidation import consolidation_node
    from config import settings

    class MockModel:
        def invoke(self, prompt, **kwargs):
            return type(
                "R",
                (),
                {
                    "content": json.dumps(
                        {
                            "ready": False,
                            "gaps": [f"Gap {i}" for i in range(10)],
                            "confidence": "high",
                        }
                    )
                },
            )()

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": MockModel(),
    )
    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q1", "priority": "high"}])

    result = consolidation_node({"plan_ref": plan_id, "iteration_count": 0})
    assert result["_route"] == "manage_backlog"

    all_questions = get_pending_questions(plan_id)
    added_texts = [
        q["question"] for q in all_questions if q.get("discovered_by") == "dod_gaps"
    ]
    assert len(added_texts) == 5, (
        f"Expected 5 gap-questions (capped), got {len(added_texts)}"
    )


def test_citation_coverage_uses_attached_question_status():
    """Regression (live, plan_d3df337a): the old check counted findings
    linked to the exa-answer raw result. Findings only link to ONE raw
    result per extraction batch, so a batch containing citation content
    could link its findings to the batch's other raw result — leaving
    missing_findings=True forever, which routed every consolidation to
    manage_backlog and blocked the periodic review. missing_findings must
    instead track the attached question's status: unresolved → the content
    has not (re)been processed; resolved → it has.
    """
    from nodes.consolidation import _check_citation_coverage

    plan_id = create_plan("Citation coverage", [])
    create_questions(plan_id, [{"question": "First question", "priority": "high"}])
    q = get_pending_question(plan_id)
    store_raw_result(
        q["id"], plan_id, "cite", "exa-answer", "content", "Foundational papers"
    )

    # Unresolved → citation content not processed yet
    assert _check_citation_coverage(plan_id) == {
        "has_citation_raw": True,
        "missing_findings": True,
    }

    set_question_in_progress(q["id"])
    assert _check_citation_coverage(plan_id)["missing_findings"] is True

    # Resolved → extraction has processed the content (whether or not any
    # finding happened to link back to the citation raw result)
    set_question_resolved(q["id"])
    assert _check_citation_coverage(plan_id)["missing_findings"] is False

    # No citation raw results → never missing
    plan_b = create_plan("No citations", [{"question": "Q", "priority": "high"}])
    assert _check_citation_coverage(plan_b) == {
        "has_citation_raw": False,
        "missing_findings": False,
    }


def test_evidence_matrix_returns_coverage_rows():
    """Regression (P0-3): get_source_coverage must join curated_findings
    (not the non-existent 'findings' table). The previous SQL produced an
    OperationalError that was swallowed by the try/except, always returning [].
    """
    plan_id = create_plan("Evidence matrix", [])
    create_questions(plan_id, [{"question": "Q1", "priority": "high"}])
    q = get_pending_question(plan_id)
    # Create raw results with different domains
    rid1 = store_raw_result(
        q["id"], plan_id, "Q1", "exa", "content", "Source A", "https://example.com/a"
    )
    rid2 = store_raw_result(
        q["id"], plan_id, "Q1", "exa", "content", "Source B", "https://example.com/b"
    )
    # Create findings linked to those raw results
    create_finding(
        question_id=q["id"],
        raw_result_id=rid1,
        knowledge_entry_id=None,
        facts=[{"claim": "fact 1", "confidence": "high"}],
        summary="Finding from A",
        tags=[],
    )
    create_finding(
        question_id=q["id"],
        raw_result_id=rid2,
        knowledge_entry_id=None,
        facts=[
            {"claim": "fact 2", "confidence": "medium"},
            {"claim": "fact 3", "confidence": "high"},
        ],
        summary="Finding from B",
        tags=[],
    )

    coverage = get_source_coverage(plan_id)
    assert len(coverage) >= 2, "Evidence matrix should return at least 2 sources"
    urls = {e["source_url"] for e in coverage}
    assert "https://example.com/a" in urls, "Source A should appear in coverage"
    assert "https://example.com/b" in urls, "Source B should appear in coverage"


def test_write_findings_falls_back_on_empty_llm_report(e2e_env, monkeypatch):
    """Regression (live, plan_d3df337a): a thinking model can spend the
    entire token budget on reasoning and return an empty report body
    WITHOUT raising. The node must treat empty/whitespace content as a
    failure and use the deterministic fallback (all findings, unsynthesized)
    instead of persisting a bodyless report."""
    from config import settings
    from graph.workflow import _write_findings_node

    plan_id = create_plan("Empty report test", [])
    create_questions(
        plan_id, [{"question": "What is the core mechanism?", "priority": "high"}]
    )
    qid = f"q_{plan_id}_0"
    set_question_in_progress(qid)
    rid = store_raw_result(qid, plan_id, "q", "exa", "content", "Source A")
    create_finding(
        qid,
        rid,
        None,
        [{"claim": "The mechanism behaves as specified.", "confidence": "high"}],
        "The mechanism behaves as specified and is widely adopted.",
        ["e2e"],
    )

    class EmptyModel:
        def invoke(self, prompt, **kwargs):
            return type("R", (), {"content": ""})()

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": EmptyModel(),
    )
    result = _write_findings_node(
        {
            "plan_ref": plan_id,
            "original_query": "What is the core mechanism?",
        }
    )

    report = result["final_report"]
    # Fallback engaged — the report is not bodyless
    assert "deterministic fallback" in report
    assert "The mechanism behaves as specified" in report
    # Question section is still guaranteed
    assert "## Research Question" in report
    assert "What is the core mechanism?" in report


def test_write_findings_retries_flaky_llm_report(e2e_env, monkeypatch):
    """A flaky model (first call returns a too-short body, second call
    succeeds) must yield the synthesized report via the retry — not the
    deterministic fallback. Regression for the live sessions where 2 of 3
    consecutive report generations failed on the local 7B model."""
    from config import settings
    from graph.workflow import _write_findings_node

    plan_id = create_plan("Retry report test", [])
    create_questions(
        plan_id, [{"question": "What is the core mechanism?", "priority": "high"}]
    )
    qid = f"q_{plan_id}_0"
    set_question_in_progress(qid)
    rid = store_raw_result(qid, plan_id, "q", "exa", "content", "Source A")
    create_finding(
        qid,
        rid,
        None,
        [{"claim": "The mechanism behaves as specified.", "confidence": "high"}],
        "The mechanism behaves as specified and is widely adopted.",
        ["e2e"],
    )

    calls = {"n": 0}

    class FlakyModel:
        def invoke(self, prompt, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return type("R", (), {"content": "too short"})()
            return type(
                "R", (), {"content": "# Report\n\n" + "The synthesized body. " * 20}
            )()

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": FlakyModel(),
    )
    result = _write_findings_node(
        {
            "plan_ref": plan_id,
            "original_query": "What is the core mechanism?",
        }
    )

    report = result["final_report"]
    assert calls["n"] == 2, "retry should make exactly one extra LLM call"
    assert "deterministic fallback" not in report
    assert "The synthesized body." in report


# ── Report: two-pass synthesis ──────────────────────────────────────
#
# The local 7B model fails on the full findings text at large plans
# (prompt-size driven). Pass 1 synthesizes from REPORT_PASS1_FINDINGS
# (capped, prioritized: citation findings first, then question priority);
# pass 2 merges the remaining findings into the pass-1 report.


def _seed_report_plan(n_findings: int, priorities=None) -> str:
    """Plan with n resolved questions, each with one distinct finding."""
    priorities = priorities or ["medium"] * n_findings
    plan_id = create_plan("Two-pass report", [])
    # ONE create_questions call: it assigns index-based ids
    # q_{plan_id}_{i} starting at 0 per call, so a per-question loop
    # collides on the UNIQUE constraint.
    create_questions(
        plan_id,
        [
            {
                "question": f"Question number {i} about topic {i}?",
                "priority": priorities[i],
            }
            for i in range(n_findings)
        ],
    )
    for i in range(n_findings):
        qid = f"q_{plan_id}_{i}"
        set_question_in_progress(qid)
        rid = store_raw_result(
            qid, plan_id, f"q{i}", "exa", f"content {i}", f"Source {i}"
        )
        create_finding(
            qid,
            rid,
            None,
            [{"claim": f"Claim number {i} is specific.", "confidence": "high"}],
            f"Unique finding summary number {i} alpha.",
            ["report-test"],
        )
        set_question_resolved(qid)
    return plan_id


def _patch_report_model(monkeypatch, model):
    from config import settings

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model_name="", max_tokens=None, task="": model(),
    )


def test_report_two_passes_when_over_cap(monkeypatch, tmp_path):
    from config.settings import paths_config
    from graph.workflow import _write_findings_node

    monkeypatch.setattr(paths_config, "base_dir", str(tmp_path))
    monkeypatch.setattr("graph.workflow.REPORT_PASS1_FINDINGS", 2)
    plan_id = _seed_report_plan(3)

    prompts = []

    class TwoPassModel:
        def invoke(self, prompt, **kwargs):
            prompts.append(prompt)
            if "=== FIRST DRAFT ===" in prompt:
                return type(
                    "R", (), {"content": "# Merged\n\n" + "Pass two merged body. " * 20}
                )()
            return type(
                "R", (), {"content": "# Draft\n\n" + "Pass one draft body. " * 20}
            )()

    _patch_report_model(monkeypatch, TwoPassModel)
    result = _write_findings_node(
        {"plan_ref": plan_id, "original_query": "Two-pass report topic"}
    )

    report = result["final_report"]
    assert len(prompts) == 2, "over-cap plan must make exactly two LLM calls"
    assert "Pass two merged body." in report
    assert "Report passes: 2" in report
    # Pass 1's prompt carried only the capped set (2 of 3 findings)
    assert prompts[0].count("### Finding:") == 2
    # Pass 2's prompt carried the pass-1 report plus the remaining finding
    assert "Pass one draft body." in prompts[1]
    assert prompts[1].count("### Finding:") == 1


def test_report_single_pass_under_cap(monkeypatch, tmp_path):
    from config.settings import paths_config
    from graph.workflow import _write_findings_node

    monkeypatch.setattr(paths_config, "base_dir", str(tmp_path))
    plan_id = _seed_report_plan(2)  # default cap is 25

    calls = {"n": 0}

    class OneShotModel:
        def invoke(self, prompt, **kwargs):
            calls["n"] += 1
            return type("R", (), {"content": "# One\n\n" + "Single pass body. " * 20})()

    _patch_report_model(monkeypatch, OneShotModel)
    result = _write_findings_node(
        {"plan_ref": plan_id, "original_query": "One-pass report topic"}
    )

    assert calls["n"] == 1
    assert "Single pass body." in result["final_report"]
    assert "Report passes: 1" in result["final_report"]


def test_report_pass2_failure_keeps_pass1_with_note(monkeypatch, tmp_path):
    """Pass-2 double failure must keep the pass-1 report (with a note),
    not discard it into the deterministic fallback."""
    from config.settings import paths_config
    from graph.workflow import _write_findings_node

    monkeypatch.setattr(paths_config, "base_dir", str(tmp_path))
    monkeypatch.setattr("graph.workflow.REPORT_PASS1_FINDINGS", 1)
    plan_id = _seed_report_plan(3)

    class SplitModel:
        def invoke(self, prompt, **kwargs):
            if "=== FIRST DRAFT ===" in prompt:
                raise RuntimeError("pass 2 exploded")
            return type("R", (), {"content": "# Draft\n\n" + "Pass one body. " * 30})()

    _patch_report_model(monkeypatch, SplitModel)
    result = _write_findings_node(
        {"plan_ref": plan_id, "original_query": "Pass2-failure topic"}
    )

    report = result["final_report"]
    assert "Pass one body." in report
    assert "deterministic fallback" not in report
    assert "second pass" in report  # the failure note
    assert "RuntimeError" in report
    assert "Report passes: 1" in report


def test_report_pass1_failure_uses_full_fallback(monkeypatch, tmp_path):
    """Pass-1 double failure keeps the existing deterministic fallback,
    which embeds EVERY finding (uncapped)."""
    from config.settings import paths_config
    from graph.workflow import _write_findings_node

    monkeypatch.setattr(paths_config, "base_dir", str(tmp_path))
    monkeypatch.setattr("graph.workflow.REPORT_PASS1_FINDINGS", 1)
    plan_id = _seed_report_plan(3)

    class BrokenModel:
        def invoke(self, prompt, **kwargs):
            raise RuntimeError("always broken")

    _patch_report_model(monkeypatch, BrokenModel)
    result = _write_findings_node(
        {"plan_ref": plan_id, "original_query": "Pass1-failure topic"}
    )

    report = result["final_report"]
    assert "deterministic fallback" in report
    assert "Report passes: 0" in report
    # Fallback is uncapped: every finding is present
    for i in range(3):
        assert f"Unique finding summary number {i} alpha." in report


def test_report_pass1_prefers_citations_and_high_priority(monkeypatch, tmp_path):
    """Pass-1 selection order: citation findings first, then question
    priority (high → medium → low)."""
    from config.settings import paths_config
    from graph.workflow import _write_findings_node

    monkeypatch.setattr(paths_config, "base_dir", str(tmp_path))
    monkeypatch.setattr("graph.workflow.REPORT_PASS1_FINDINGS", 2)

    plan_id = create_plan("Priority report", [])
    # q0: low, q1: high, q2: medium + citation engine
    create_questions(
        plan_id,
        [
            {"question": "Low priority question?", "priority": "low"},
            {"question": "High priority question?", "priority": "high"},
            {"question": "Citation question?", "priority": "medium"},
        ],
    )
    engines = ["exa", "exa", "exa-answer"]
    for i in range(3):
        qid = f"q_{plan_id}_{i}"
        set_question_in_progress(qid)
        rid = store_raw_result(
            qid, plan_id, f"q{i}", engines[i], f"content {i}", f"Source {i}"
        )
        create_finding(
            qid,
            rid,
            None,
            [{"claim": f"Claim {i}.", "confidence": "high"}],
            f"Unique finding summary number {i} alpha.",
            ["report-test"],
        )
        set_question_resolved(qid)

    prompts = []

    class TwoPassModel:
        def invoke(self, prompt, **kwargs):
            prompts.append(prompt)
            if "=== FIRST DRAFT ===" in prompt:
                return type(
                    "R", (), {"content": "# Merged\n\n" + "Merged body. " * 20}
                )()
            return type("R", (), {"content": "# Draft\n\n" + "Draft body. " * 20})()

    _patch_report_model(monkeypatch, TwoPassModel)
    result = _write_findings_node(
        {"plan_ref": plan_id, "original_query": "Priority topic"}
    )

    first, second = prompts
    # Citation (f2) and high-priority (f1) win pass 1; low (f0) goes to pass 2
    assert "Unique finding summary number 2" in first
    assert "Unique finding summary number 1" in first
    assert "Unique finding summary number 0" not in first
    assert "Unique finding summary number 0" in second
    assert result["final_report"].count("Merged body.") >= 1


# ── Report: Suggested Follow-Up Questions ───────────────────────────


def _seed_resolved_report_plan():
    """Plan with one resolved question (plus raw result + finding) so the
    report node has real content to work with."""
    plan_id = create_plan("Follow-up report test", [])
    create_questions(
        plan_id, [{"question": "What is the core mechanism?", "priority": "high"}]
    )
    qid = f"q_{plan_id}_0"
    set_question_in_progress(qid)
    rid = store_raw_result(qid, plan_id, "q", "exa", "content", "Source A")
    create_finding(
        qid,
        rid,
        None,
        [{"claim": "The mechanism behaves as specified.", "confidence": "high"}],
        "The mechanism behaves as specified and is widely adopted.",
        ["e2e"],
    )
    set_question_resolved(qid)
    return plan_id


def _run_report_node(plan_id, query, report_body, monkeypatch, tmp_path):
    """Run _write_findings_node with a scripted LLM body and an isolated
    data dir (the node writes a report file under paths_config.base_dir)."""
    from config import settings
    from config.settings import paths_config
    from graph.workflow import _write_findings_node

    monkeypatch.setattr(paths_config, "base_dir", str(tmp_path))

    class ReportModel:
        def invoke(self, prompt, **kwargs):
            return type("R", (), {"content": report_body})()

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": ReportModel(),
    )
    return _write_findings_node({"plan_ref": plan_id, "original_query": query})[
        "final_report"
    ]


def test_report_followup_backstop_appends_open_questions(monkeypatch, tmp_path):
    """The LLM omits the 'Suggested Follow-Up Questions' section and the
    bank has an open question → the node appends the open questions so
    structured follow-ups are never silently lost."""
    body = (
        "# Follow-up report\n\n## Executive Summary\n\n"
        "This report synthesizes the findings and deliberately omits the "
        "follow-up section, which the deterministic backstop must restore "
        "from the open rows in the question bank for this plan."
    )
    plan_id = _seed_resolved_report_plan()
    add_question(plan_id, "How do the top tools handle non-ML users?")

    report = _run_report_node(
        plan_id, "What is the core mechanism?", body, monkeypatch, tmp_path
    )

    assert "## Suggested Follow-Up Questions" in report
    assert "How do the top tools handle non-ML users?" in report


def test_report_followup_section_from_llm_is_preserved(monkeypatch, tmp_path):
    """When the LLM writes its own 'Suggested Follow-Up Questions' section,
    the node must keep it verbatim and not append a duplicate backstop."""
    body = (
        "# Follow-up report\n\n## Executive Summary\n\n"
        "Report body that already contains the follow-up section, which the "
        "node must preserve and must not duplicate with a deterministic "
        "backstop listing the bank rows."
        "\n\n## Suggested Follow-Up Questions\n\n"
        "- How does the mechanism scale to distributed settings?\n"
    )
    plan_id = _seed_resolved_report_plan()
    add_question(plan_id, "An open bank row that must not be duplicated?")

    report = _run_report_node(
        plan_id, "What is the core mechanism?", body, monkeypatch, tmp_path
    )

    assert report.count("## Suggested Follow-Up Questions") == 1
    assert "How does the mechanism scale to distributed settings?" in report


def test_report_followup_backstop_does_not_fabricate(monkeypatch, tmp_path):
    """The LLM omits the section AND the bank has no open questions → the
    node must not invent a section (code never fabricates questions)."""
    body = (
        "# Follow-up report\n\n## Executive Summary\n\n"
        "Report body with no follow-up section, and the question bank for "
        "this plan has no open rows, so the node must not fabricate one."
    )
    plan_id = _seed_resolved_report_plan()

    report = _run_report_node(
        plan_id, "What is the core mechanism?", body, monkeypatch, tmp_path
    )

    assert "Suggested Follow-Up Questions" not in report


# ── Extraction Node ─────────────────────────────────────────────────


def test_extraction_no_raw_results():
    """Extraction handles missing raw results gracefully."""
    from nodes.extraction import extract_and_verify_node

    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q1", "priority": "high"}])
    set_question_in_progress(f"q_{plan_id}_0")
    result = extract_and_verify_node({"plan_ref": plan_id})
    assert result == {}


def test_extraction_with_mock_results(monkeypatch):
    """Extraction with LLM mock produces a finding."""
    from nodes.extraction import extract_and_verify_node
    from config import settings

    class MockModel:
        def invoke(self, prompt, **kwargs):
            return type(
                "Response",
                (),
                {
                    "content": json.dumps(
                        [
                            {
                                "facts": [{"claim": "Test fact", "confidence": "high"}],
                                "summary": "Test summary",
                                "discovered_questions": [],
                                "tags": ["test"],
                            }
                        ]
                    )
                },
            )()

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": MockModel(),
    )

    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q1?", "priority": "high"}])
    q = get_pending_question(plan_id)
    set_question_in_progress(q["id"])
    store_raw_result(q["id"], plan_id, "Q1?", "mock", "Some content", "Title")

    result = extract_and_verify_node({"plan_ref": plan_id})
    assert result == {}

    # Question should be resolved
    assert all_questions_are_resolved(plan_id) is True

    # Finding should exist
    findings = get_plan_findings(plan_id)
    assert len(findings) == 1
    assert findings[0]["summary"] == "Test summary"


def test_extraction_llm_failure_marks_question_failed(monkeypatch):
    """An LLM error must mark the question 'failed' — not silently 'resolved' —
    so the loop cannot declare a plan complete when research actually failed.
    """
    from nodes.extraction import extract_and_verify_node
    from config import settings

    class BoomModel:
        def invoke(self, prompt, **kwargs):
            raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": BoomModel(),
    )
    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q1?", "priority": "high"}])
    q = get_pending_question(plan_id)
    set_question_in_progress(q["id"])
    store_raw_result(q["id"], plan_id, "Q1?", "mock", "Some content", "Title")

    result = extract_and_verify_node({"plan_ref": plan_id})
    assert result == {}

    assert all_questions_are_resolved(plan_id) is False
    stats = get_plan_stats(plan_id)
    assert stats.get("failed") == 1


def test_extraction_attribution_uses_source_index(monkeypatch):
    """Findings should link to the raw result indicated by each fact's
    'source' index, not blindly to raw_result_ids[0].
    """
    from nodes.extraction import extract_and_verify_node
    from config import settings

    class MockModel:
        def invoke(self, prompt, **kwargs):
            # 2 facts from source 1, 1 fact from source 0 → primary = 1
            return type(
                "Response",
                (),
                {
                    "content": json.dumps(
                        [
                            {
                                "facts": [
                                    {
                                        "claim": "Fact A",
                                        "confidence": "high",
                                        "source": 0,
                                    },
                                    {
                                        "claim": "Fact B",
                                        "confidence": "high",
                                        "source": 1,
                                    },
                                    {
                                        "claim": "Fact C",
                                        "confidence": "high",
                                        "source": 1,
                                    },
                                ],
                                "summary": "Multi-source finding",
                                "discovered_questions": [],
                                "tags": ["test"],
                            }
                        ]
                    )
                },
            )()

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": MockModel(),
    )

    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q?", "priority": "high"}])
    q = get_pending_question(plan_id)
    set_question_in_progress(q["id"])

    # Three raw results. Stored in order: A=1, B=2, C=3.
    # raw_results are DESC → raw_result_ids = [3, 2, 1]
    store_raw_result(q["id"], plan_id, "Q?", "exa", "Content A", "Title A")
    store_raw_result(q["id"], plan_id, "exa", "Content B", "Title B")
    store_raw_result(q["id"], plan_id, "exa", "Content C", "Title C")

    result = extract_and_verify_node({"plan_ref": plan_id})
    assert result == {}

    # Primary source = 1 (most common). raw_result_ids[1] = 2 (the middle one)
    findings = get_plan_findings(plan_id)
    assert len(findings) == 1
    assert findings[0]["raw_result_id"] == 2


def test_extraction_attribution_falls_back_to_first_when_no_source(monkeypatch):
    """When facts lack a 'source' index (old model output), fall back to
    raw_result_ids[0] — the previous behaviour.

    Note: raw_results are ordered DESC by ID, so raw_result_ids[0] is the
    LATEST raw result stored for the question.
    """
    from nodes.extraction import extract_and_verify_node
    from config import settings

    class MockModel:
        def invoke(self, prompt, **kwargs):
            return type(
                "Response",
                (),
                {
                    "content": json.dumps(
                        [
                            {
                                "facts": [
                                    {
                                        "claim": "Fact 1",
                                        "confidence": "high",
                                    },  # no source field
                                    {"claim": "Fact 2", "confidence": "medium"},
                                ],
                                "summary": "Old-style finding",
                                "discovered_questions": [],
                                "tags": ["test"],
                            }
                        ]
                    )
                },
            )()

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": MockModel(),
    )

    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q?", "priority": "high"}])
    q = get_pending_question(plan_id)
    set_question_in_progress(q["id"])
    store_raw_result(q["id"], plan_id, "Q?", "exa", "Content A", "Title A")
    store_raw_result(q["id"], plan_id, "Q?", "exa", "Content B", "Title B")

    result = extract_and_verify_node({"plan_ref": plan_id})
    assert result == {}

    findings = get_plan_findings(plan_id)
    assert len(findings) == 1
    # Falls back to raw_result_ids[0], which is the LATEST stored (DESC order)
    assert findings[0]["raw_result_id"] == 2


def test_extraction_attribution_handles_empty_content_result(monkeypatch):
    """Regression (P0-5): when a raw result has empty content, it is excluded
    from the context blocks. Source indices in the LLM output refer to the
    *filtered* block positions. Both raw_result_id AND source_url must come
    from the filtered list, not the full raw_results list.
    """
    from nodes.extraction import extract_and_verify_node
    from config import settings

    class MockModel:
        def invoke(self, prompt, **kwargs):
            return type(
                "Response",
                (),
                {
                    "content": json.dumps(
                        [
                            {
                                "facts": [
                                    {
                                        "claim": "Fact from first real block",
                                        "confidence": "high",
                                        "source": 0,
                                    },
                                ],
                                "summary": "Single-source finding",
                                "discovered_questions": [],
                                "tags": ["test"],
                            }
                        ]
                    )
                },
            )()

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": MockModel(),
    )

    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q?", "priority": "high"}])
    q = get_pending_question(plan_id)
    set_question_in_progress(q["id"])

    # Three raw results: middle one has empty content.
    # Stored order: A(id=1), empty(id=2), B(id=3).
    # get_raw_results returns DESC → [B(id=3), empty(id=2), A(id=1)].
    # Filtered (non-empty): B(id=3), A(id=1) → raw_result_ids = [3, 1].
    # LLM source=0 → raw_result_ids[0] = 3 (Content B).
    store_raw_result(q["id"], plan_id, "Q?", "exa", "Content A", "Title A")
    store_raw_result(q["id"], plan_id, "exa", "", "Empty Title")  # empty content
    store_raw_result(q["id"], plan_id, "exa", "Content B", "Title B")

    result = extract_and_verify_node({"plan_ref": plan_id})
    assert result == {}

    # The LLM's source=0 → first non-empty block (DESC) = Content B (id=3)
    findings = get_plan_findings(plan_id)
    assert len(findings) == 1
    assert findings[0]["raw_result_id"] == 3, (
        f"Expected raw_result_id=3 (Content B, first non-empty DESC), got {findings[0]['raw_result_id']}"
    )


def test_extraction_context_total_cap_omits_excess_sources(monkeypatch):
    """P1-7: the extraction prompt has a TOTAL context budget, not just a
    per-source cap. When sources exceed it, the excess is omitted with an
    explicit note, citation-discovery sources are prioritized, and source
    indices still refer to the included blocks only.
    """
    from nodes.extraction import extract_and_verify_node
    from config import settings

    prompts = []

    class MockModel:
        def invoke(self, prompt, **kwargs):
            prompts.append(prompt)
            return type(
                "Response",
                (),
                {
                    "content": json.dumps(
                        [
                            {
                                "facts": [
                                    {
                                        "claim": "Captured fact",
                                        "confidence": "high",
                                        "source": 0,
                                    },
                                ],
                                "summary": "Context cap test finding",
                                "discovered_questions": [],
                                "tags": ["test"],
                            }
                        ]
                    )
                },
            )()

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": MockModel(),
    )

    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q?", "priority": "high"}])
    q = get_pending_question(plan_id)
    set_question_in_progress(q["id"])

    # 12 exa results x 3000 chars = 36k chars — well over the 24k total budget
    for i in range(12):
        store_raw_result(q["id"], plan_id, "Q?", "exa", "x" * 3000, f"Title {i}")
    # One small citation-discovery result — must survive and be indexed first
    cite_id = store_raw_result(
        q["id"], plan_id, "Q?", "exa-answer", "foundational note", "Paper", "http://x"
    )

    result = extract_and_verify_node({"plan_ref": plan_id})
    assert result == {}

    prompt = prompts[0]
    # Budget: citation block (60 chars) + 7 exa blocks (3014 each) fit in
    # 24k; the remaining 5 exa results are omitted with an explicit note.
    assert "additional source(s) were omitted" in prompt
    assert prompt.count("[SOURCE: exa]") == 7
    # Citation-discovery block is included and ordered first
    assert "[FOUNDATIONAL PAPERS / CITATION DISCOVERY]" in prompt
    assert prompt.index("[FOUNDATIONAL PAPERS") < prompt.index("[SOURCE: exa]")

    # source=0 → first included block = the citation-discovery result
    findings = get_plan_findings(plan_id)
    assert len(findings) == 1
    assert findings[0]["raw_result_id"] == cite_id
    # Omission is prompt-level only — nothing is lost from the DB
    assert len(get_raw_results_for_question(q["id"])) == 13


# ── Search Node ─────────────────────────────────────────────────────


def test_search_node_no_active_question():
    from nodes.search import search_execution_node

    result = search_execution_node({"plan_ref": "nonexistent"})
    assert result == {}


def test_search_node_cached_result():
    """Search node uses cached results when available."""
    from nodes.search import search_execution_node
    from config.settings import search_config

    # Deterministic: disable citation discovery (its flag file lives in the
    # production data dir and must not influence test behavior)
    original_cite = search_config.enable_citation_discovery
    search_config.enable_citation_discovery = False
    try:
        plan_id = create_plan("Test", [])
        create_questions(
            plan_id,
            [
                {
                    "question": "Unique cached query xyz about grpo training",
                    "priority": "high",
                }
            ],
        )
        qid = f"q_{plan_id}_0"
        set_question_in_progress(qid)

        # Pre-populate cache with the composite key
        # (query||primary||arxiv_flag||arxiv_max_results||github_flag||github_max_results).
        # "grpo" is a strong science term, so the routing plan sets arxiv=True
        # for this query; it has no OSS vocabulary, so github=False.
        q = "Unique cached query xyz about grpo training"
        store_cache(
            f"{q}||exa||True||{search_config.arxiv_max_results}||False"
            f"||{search_config.github_max_results}",
            [
                {
                    "title": "Cached Result",
                    "url": "https://example.com",
                    "content": "Cached text",
                },
            ],
            engine="exa+arxiv",
        )

        result = search_execution_node({"plan_ref": plan_id})
        assert result == {}

        # The cached result should be stored as a raw result
        results = get_raw_results_for_question(qid)
        assert len(results) >= 1
        cached = [
            r
            for r in results
            if "Cached Result" in r.get("title", "")
            or r["engine"] == "exa (cached)+arxiv"
        ]
        assert len(cached) >= 1
    finally:
        search_config.enable_citation_discovery = original_cite


# ── Workflow Compilation & State Schema ─────────────────────────────


def test_workflow_compiles():
    """The full workflow graph compiles without errors."""
    from graph.workflow import build_app, workflow
    from langgraph.checkpoint.memory import MemorySaver

    app = build_app(MemorySaver())
    assert app is not None
    assert workflow is not None


def test_run_loop_stops_when_resumed_graph_completes():
    """Regression (live-run hang): after a Command(resume=...) stream ends at
    END with no new interrupt, run_loop must exit. The old loop keyed off
    isinstance(input_obj, Command) and kept re-streaming the same resume
    against the completed graph forever."""
    import builtins
    from langgraph.types import Command  # noqa: F401  (ensures import path works)
    from run import run_loop

    class FakeApp:
        def __init__(self):
            self.calls = 0

        def stream(self, input_obj, cfg, stream_mode="updates"):
            self.calls += 1
            if self.calls > 3:
                raise AssertionError(
                    f"run_loop re-streamed a completed graph ({self.calls} calls)"
                )
            if self.calls == 1:
                # Pass 1: graph suspends with an interrupt
                return [
                    {"__interrupt__": (type("Intr", (), {"value": {"prompt": "p"}}),)}
                ]
            # Pass 2: resumed graph completes — no interrupt
            return [{"write_findings": {"final_report": "done"}}]

    app = FakeApp()
    orig_input = builtins.input
    builtins.input = lambda prompt="": "stop"
    try:
        run_loop(app, {"original_query": "q"}, {"configurable": {"thread_id": "t"}})
    finally:
        builtins.input = orig_input
    assert app.calls == 2, f"expected exactly 2 stream passes, got {app.calls}"


def test_workflow_state_declares_control_keys():
    """Every key a node returns MUST be declared in ResearchState —
    LangGraph silently drops undeclared keys from node output, which
    breaks control flow (human response, quality eval, display payloads).
    """
    from state.research_state import ResearchState

    keys = set(ResearchState.__annotations__.keys())
    expected = {
        "original_query",
        "questions",
        "plan_ref",
        "iteration_count",
        "is_done",
        "_route",
        "_human_response",
        "_quality",
        "_hit_max_iterations",
        "_coverage",
        "_findings_streak",
        "_prev_finding_count",
        "_stats_display",
        "_findings_display",
        "_contradictions_display",
        "_error",
        "final_report",
    }
    assert expected <= keys


# ── LLM progress logging (invoke_with_cost) ─────────────────────────
#
# Every LLM call prints a start line and a done/failed line (duration +
# tokens) so a running session is checkable at a glance — silence used
# to be indistinguishable from a hang on the slow local model.


def test_invoke_with_cost_prints_progress_lines(capsys):
    from utils.llm import invoke_with_cost

    class Model:
        model_name = "test-model"

        def invoke(self, prompt):
            return type("R", (), {"content": "ok"})()

    invoke_with_cost(Model(), "hello", task="extraction")
    out = capsys.readouterr().out
    assert "[LLM] extraction (test-model): generating" in out
    assert "[LLM] extraction done in" in out
    assert "tokens)" in out


def test_invoke_with_cost_prints_failure_line(capsys):
    from utils.llm import invoke_with_cost

    class Model:
        model_name = "test-model"

        def invoke(self, prompt):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        invoke_with_cost(Model(), "hello", task="extraction")
    out = capsys.readouterr().out
    assert "[LLM] extraction FAILED" in out
    assert "RuntimeError" in out


# ── E2E: Full Graph Runs (scripted LLM, mocked search) ──────────────


class MockLLM:
    """Scripted LLM — dispatches on prompt content, never touches the network.

    prompts: class-level record of every prompt received (shared across
    per-call instances) so tests can assert what the graph actually sent.
    """

    prompts: list = []

    def invoke(self, prompt: str):
        MockLLM.prompts.append(prompt)
        if "focused, searchable sub-questions" in prompt:
            content = json.dumps(
                [
                    {"question": "What is the core mechanism?", "priority": "high"},
                    {"question": "What are the trade-offs?", "priority": "medium"},
                ]
            )
        elif "Research question to answer" in prompt:
            content = json.dumps(
                [
                    {
                        "facts": [
                            {
                                "claim": "E2E fact: the mechanism behaves as specified.",
                                "confidence": "high",
                                "source": 0,
                            },
                        ],
                        "summary": "E2E finding: the mechanism behaves as specified and is widely adopted.",
                        "discovered_questions": [],
                        "tags": ["e2e"],
                    }
                ]
            )
        elif "ready to produce a final report" in prompt:
            content = json.dumps(
                {"ready": False, "gaps": ["needs more depth"], "confidence": "high"}
            )
        elif "deduplicating a list of open research questions" in prompt:
            # No duplicates in the E2E script — keep the bank intact.
            content = json.dumps({"discard": []})
        elif "consistency checker" in prompt:
            content = "[]"
        elif "Generate recommendations for each raw entry" in prompt:
            content = "[]"
        elif "managing a research queue" in prompt:
            content = "[]"
        elif "comprehensive research report" in prompt:
            # Must exceed the report node's 200-char minimum (empty-body guard)
            content = (
                "# E2E-REPORT\n\n## Executive Summary\n\n"
                "This is a comprehensive research report generated by the E2E mock "
                "LLM. It synthesizes the curated findings for the test plan and "
                "states coverage per named system, as the report prompt requires "
                "for balanced reporting across all systems mentioned in the query."
            )
        else:
            raise AssertionError(f"Unexpected LLM prompt in E2E test: {prompt[:120]!r}")
        return type("Response", (), {"content": content})()


def _mock_search(query: str) -> list:
    return [
        {
            "title": f"E2E source: {query[:40]}",
            "url": "https://example.com/e2e",
            "content": "E2E search content: the mechanism behaves as specified.",
        }
    ]


@pytest.fixture
def e2e_env(tmp_path, monkeypatch):
    """Isolated environment for E2E graph runs: scripted LLM, mocked search,
    no citation discovery, report files in tmp_path. Everything is restored
    automatically by monkeypatch; tests pin max_iterations/review_every
    the same way."""
    from config import settings
    from config.settings import search_config, workflow_config, paths_config
    import nodes.search as search_mod

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": MockLLM(),
    )
    monkeypatch.setattr(search_mod, "_search_exa", _mock_search)
    monkeypatch.setattr(search_mod, "_search_arxiv", lambda q: [])
    monkeypatch.setattr(search_config, "enable_citation_discovery", False)
    monkeypatch.setattr(workflow_config, "min_findings", 1)
    monkeypatch.setattr(paths_config, "base_dir", str(tmp_path))
    yield


def _stream(app, input_obj, cfg: dict):
    """Stream one phase of the graph. Returns the interrupt payload if the
    graph suspended, else None (graph finished)."""
    payload = None
    for event in app.stream(input_obj, cfg, stream_mode="updates"):
        if "__interrupt__" in event:
            payload = event["__interrupt__"][0].value
    return payload


def test_e2e_review_interrupt_then_stop(e2e_env, monkeypatch):
    """review_every=1 → graph suspends at the iteration-1 review; 'stop'
    produces the final report and reaches END. Verifies the P0 interrupt fix
    (the graph actually suspends) and the clean exit path.
    """
    from graph.workflow import build_app
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command
    from config.settings import workflow_config

    monkeypatch.setattr(workflow_config, "max_iterations", 10)
    monkeypatch.setattr(workflow_config, "review_every_n_iterations", 1)

    app = build_app(MemorySaver())
    thread = "e2e-scenario-1"
    cfg = {"configurable": {"thread_id": thread}}
    MockLLM.prompts.clear()
    initial = {
        "original_query": "E2E research topic alpha",
        "questions": [
            {"question": "What is the core mechanism?", "priority": "high"},
            {"question": "What are the trade-offs?", "priority": "medium"},
        ],
    }

    payload = _stream(app, initial, cfg)
    assert payload is not None, "graph should suspend at the iteration-1 review"
    assert "How would you like to proceed?" in payload["prompt"]
    # Coverage floor: the question bank's state is visible on the review
    # screen (1 resolved, 1 pending at this point)
    assert "Question coverage: 1/2 closed" in payload["prompt"]

    # Scope guard: the extraction, readiness, and curation prompts all
    # received the deterministic scope block for this query
    scope_prompts = [p for p in MockLLM.prompts if "RESEARCH SCOPE (enforced):" in p]
    assert len(scope_prompts) >= 3
    assert all('Topic: "E2E research topic alpha"' in p for p in scope_prompts)

    payload2 = _stream(app, Command(resume="stop"), cfg)
    assert payload2 is None, "graph should finish after 'stop'"

    state = app.get_state(cfg)
    assert state.next == ()
    assert state.values["is_done"] is True
    assert "E2E-REPORT" in state.values["final_report"]

    # The question section is inserted by code (guaranteed, not LLM-dependent):
    # verbatim query + every sub-question with its status
    report = state.values["final_report"]
    assert "## Research Question" in report
    assert "E2E research topic alpha" in report
    assert "- [x] What is the core mechanism? — resolved" in report
    assert "- [ ] What are the trade-offs? — pending" in report

    # Interim labeling: the trade-offs question is still pending, so the
    # report must be explicitly a snapshot (top banner + stats line)
    assert "**Interim report** — 1 of 2 sub-questions are still open" in report
    assert "Open questions: 1 of 2" in report

    plan_ref = state.values["plan_ref"]
    with get_db() as conn:
        n_reports = conn.execute(
            "SELECT COUNT(*) c FROM reports WHERE plan_id=?", (plan_ref,)
        ).fetchone()["c"]
        t = conn.execute(
            "SELECT thread_id FROM plans WHERE id=?", (plan_ref,)
        ).fetchone()["thread_id"]
        n_resolved = conn.execute(
            "SELECT COUNT(*) c FROM questions WHERE plan_id=? AND status='resolved'",
            (plan_ref,),
        ).fetchone()["c"]
    assert n_reports == 1
    assert t == thread
    assert n_resolved == 1  # second question never researched (stopped early)


def test_e2e_max_iterations_interim_then_stop(e2e_env, monkeypatch):
    """max_iterations=2, quality never ready → DoD gap is promoted to a
    question (self-correction), the loop hits max iterations, the interim
    analysis interrupts, 'stop' writes the report and reaches END.
    """
    from graph.workflow import build_app
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command
    from config.settings import workflow_config

    monkeypatch.setattr(workflow_config, "max_iterations", 2)
    monkeypatch.setattr(workflow_config, "review_every_n_iterations", 99)

    app = build_app(MemorySaver())
    thread = "e2e-scenario-2"
    cfg = {"configurable": {"thread_id": thread}}
    initial = {
        "original_query": "E2E research topic beta",
        "questions": [{"question": "What is the mechanism?", "priority": "high"}],
    }

    payload = _stream(app, initial, cfg)
    assert payload is not None, "graph should suspend at the interim review"
    assert "INTERIM ANALYSIS (max iterations reached)" in payload["prompt"]

    payload2 = _stream(app, Command(resume="stop"), cfg)
    assert payload2 is None, "graph should finish after 'stop'"

    state = app.get_state(cfg)
    assert state.next == ()
    assert state.values["is_done"] is True
    assert "E2E-REPORT" in state.values["final_report"]

    plan_ref = state.values["plan_ref"]
    with get_db() as conn:
        n_reports = conn.execute(
            "SELECT COUNT(*) c FROM reports WHERE plan_id=?", (plan_ref,)
        ).fetchone()["c"]
        total_q = conn.execute(
            "SELECT COUNT(*) c FROM questions WHERE plan_id=?", (plan_ref,)
        ).fetchone()["c"]
        n_resolved = conn.execute(
            "SELECT COUNT(*) c FROM questions WHERE plan_id=? AND status='resolved'",
            (plan_ref,),
        ).fetchone()["c"]
    assert n_reports == 1
    # At least the original question + at least one DoD gap.
    # The exact DoD gap count is non-deterministic (LLM generates
    # different gap texts per iteration), but should be >= 2.
    assert total_q >= 2
    assert n_resolved >= 2


# ── Dedup & Prioritize ──────────────────────────────────────────────


def test_deduplicate_removes_exact_duplicates():
    """Near-duplicate questions (normalized) are collapsed."""
    plan_id = create_plan("Test", [])
    create_questions(
        plan_id,
        [
            {"question": "What are micro-frontends?", "priority": "high"},
            {"question": "What is a micro-frontend?", "priority": "medium"},
            {"question": "What about microfrontends?", "priority": "low"},
            {"question": "What are the challenges?", "priority": "high"},
        ],
    )
    deduplicate_and_prioritize(plan_id)

    # Check: only non-deduped questions remain pending
    with get_db() as conn:
        pending = conn.execute(
            "SELECT * FROM questions WHERE plan_id = ? AND status = 'pending'",
            (plan_id,),
        ).fetchall()
    assert (
        len(pending) == 2
    )  # "micro-frontends" group collapsed to 1, "challenges" kept

    # The kept question should be the highest priority one
    kept = [
        dict(r)
        for r in pending
        if "micro" in r["question"].lower() or "frontend" in r["question"].lower()
    ]
    assert len(kept) == 1
    assert kept[0]["priority"] == "high"  # inherited high from the group


def test_deduplicate_noop_with_single_question():
    """Nothing to dedup when only one question."""
    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Just one question", "priority": "high"}])
    deduplicate_and_prioritize(plan_id)
    stats = get_plan_stats(plan_id)
    assert stats["pending"] == 1


def test_deduplicate_preserves_distinct_questions():
    """Truly different questions are not deduped."""
    plan_id = create_plan("Test", [])
    create_questions(
        plan_id,
        [
            {"question": "What is X?", "priority": "high"},
            {"question": "What is Y?", "priority": "medium"},
            {"question": "What is Z?", "priority": "low"},
        ],
    )
    deduplicate_and_prioritize(plan_id)
    stats = get_plan_stats(plan_id)
    assert stats["pending"] == 3  # all distinct, none removed


# ── LLM-assisted question dedup ─────────────────────────────────────
#
# Keyword Jaccard only catches near-verbatim paraphrases pending-to-
# pending. _llm_dedup_open_questions() batches the open questions plus
# the recently resolved ones into one LLM call and catches semantic
# duplicates — including re-asks of just-resolved questions.


def _patch_dedup_llm(monkeypatch, discard: list):
    """Script config.get_chat_model so the dedup call returns `discard`."""
    from config import settings

    class DedupModel:
        def invoke(self, prompt, **kwargs):
            assert "deduplicating a list of open research questions" in prompt
            return type("R", (), {"content": json.dumps({"discard": discard})})()

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": DedupModel(),
    )


def _two_open_questions(plan_id: str) -> tuple:
    """Two open questions that pass the insert-time Jaccard guard."""
    q1 = add_question(
        plan_id,
        "What is the performance of vLLM on A100 GPUs?",
        discovered_by="user",
    )
    q2 = add_question(
        plan_id,
        "How fast is vLLM on A100 hardware?",
        discovered_by="user",
    )
    return q1, q2


def test_llm_dedup_discards_paraphrase_duplicate(monkeypatch):
    from nodes.curation import _llm_dedup_open_questions

    plan_id = create_plan("Test", [])
    q1, q2 = _two_open_questions(plan_id)
    _patch_dedup_llm(
        monkeypatch,
        [{"id": q2, "duplicate_of": q1, "reason": "asks the same thing"}],
    )

    result = _llm_dedup_open_questions(plan_id)

    assert [d["id"] for d in result] == [q2]
    assert result[0]["duplicate_of"] == q1
    row = get_question(q2)
    assert row["status"] == "discarded"
    assert row["priority"] == "low"
    assert row["discovered_by"] == "llm-dedup"
    assert get_question(q1)["status"] == "pending"


def test_llm_dedup_discards_re_ask_of_resolved_question(monkeypatch):
    """The observed growth mode: an open question that re-asks one the
    plan just resolved (anchored to a resolved id → whole component
    discarded)."""
    from nodes.curation import _llm_dedup_open_questions

    plan_id = create_plan("Test", [])
    q1, q2 = _two_open_questions(plan_id)
    # A resolved question the plan already answered.
    resolved_q = add_question(
        plan_id,
        "Which serving framework handles prefix caching best?",
        discovered_by="user",
    )
    set_question_resolved(resolved_q)
    _patch_dedup_llm(
        monkeypatch,
        [{"id": q2, "duplicate_of": resolved_q, "reason": "already answered"}],
    )

    result = _llm_dedup_open_questions(plan_id)

    assert [d["id"] for d in result] == [q2]
    assert get_question(resolved_q)["status"] == "resolved"  # untouched
    assert get_question(q1)["status"] == "pending"


def test_llm_dedup_mutual_flags_keep_best_member(monkeypatch):
    """A group where the LLM flags every member against each other keeps
    its best question (priority, then age) — never discards the group."""
    from nodes.curation import _llm_dedup_open_questions

    plan_id = create_plan("Test", [])
    q1, q2 = _two_open_questions(plan_id)
    update_question_priority(q1, "high")  # q1 beats q2 (medium)
    _patch_dedup_llm(
        monkeypatch,
        [
            {"id": q1, "duplicate_of": q2, "reason": "same"},
            {"id": q2, "duplicate_of": q1, "reason": "same"},
        ],
    )

    result = _llm_dedup_open_questions(plan_id)

    assert [d["id"] for d in result] == [q2]
    assert get_question(q1)["status"] == "pending"


def test_llm_dedup_ignores_hallucinated_ids(monkeypatch):
    from nodes.curation import _llm_dedup_open_questions

    plan_id = create_plan("Test", [])
    q1, q2 = _two_open_questions(plan_id)
    _patch_dedup_llm(
        monkeypatch,
        [
            {"id": "q_plan_fake_0000", "duplicate_of": q1, "reason": "ghost"},
            {"id": q2, "duplicate_of": "q_plan_fake_0001", "reason": "ghost"},
        ],
    )

    result = _llm_dedup_open_questions(plan_id)

    assert result == []
    assert get_plan_stats(plan_id)["pending"] == 2


def test_llm_dedup_llm_failure_returns_empty(monkeypatch):
    """Any failure (LLM down, bad JSON) → no changes, []."""
    from config import settings
    from nodes.curation import _llm_dedup_open_questions

    plan_id = create_plan("Test", [])
    _two_open_questions(plan_id)

    class BrokenModel:
        def invoke(self, prompt, **kwargs):
            raise RuntimeError("llm down")

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": BrokenModel(),
    )

    assert _llm_dedup_open_questions(plan_id) == []
    assert get_plan_stats(plan_id)["pending"] == 2


def test_llm_dedup_skips_when_fewer_than_two_open(monkeypatch):
    """No LLM call at all with <2 open questions (the model must never
    be invoked — it would raise here if it were)."""
    from config import settings
    from nodes.curation import _llm_dedup_open_questions

    plan_id = create_plan("Test", [])
    add_question(plan_id, "A single open question?", discovered_by="user")

    class NeverCallModel:
        def invoke(self, prompt, **kwargs):
            raise AssertionError("LLM must not be called with <2 open questions")

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": NeverCallModel(),
    )

    assert _llm_dedup_open_questions(plan_id) == []
    assert get_plan_stats(plan_id)["pending"] == 1


def test_prepare_review_payload_includes_llm_dedup(monkeypatch):
    """prepare_review_node runs the dedup pass and surfaces its result in
    the review payload (both the structured field and the prompt text)."""
    from config import settings
    from nodes.curation import prepare_review_node

    plan_id = create_plan("Test", [])
    q1, q2 = _two_open_questions(plan_id)

    class DedupModel:
        def invoke(self, prompt, **kwargs):
            if "deduplicating a list of open research questions" in prompt:
                content = json.dumps(
                    {"discard": [{"id": q2, "duplicate_of": q1, "reason": "same"}]}
                )
            elif "consistency checker" in prompt:
                content = "[]"
            elif "Generate recommendations" in prompt:
                content = "[]"
            else:
                raise AssertionError(f"Unexpected prompt: {prompt[:100]!r}")
            return type("R", (), {"content": content})()

    monkeypatch.setattr(
        settings.LLMConfig,
        "get_chat_model",
        lambda self, model="", max_tokens=None, task="": DedupModel(),
    )

    payload = prepare_review_node({"plan_ref": plan_id, "iteration_count": 1})[
        "_review_payload"
    ]

    assert [d["id"] for d in payload["llm_dedup"]] == [q2]
    assert "LLM dedup" in payload["prompt"]
    # Coverage is refreshed after the LLM dedup: q2 discarded -> 1 of 2
    # closed, 1 open (q1, medium priority -> 0 high-priority). Without the
    # refresh the line would be missing entirely (state had no _coverage).
    assert (
        "Question coverage: 1/2 closed (1 open, 0 high-priority)" in payload["prompt"]
    )


# ── Insert-time similarity guard (add_question) ───────────────────
#
# Session 4 of the OSS agent-eval run showed the pipeline re-asking
# questions it had just resolved (7 of 12 new questions were paraphrase
# duplicates). add_question() now suppresses near-duplicates for
# automatic sources (Jaccard >= 0.4 vs any non-discarded plan
# question, same rule as curation dedup). User-typed questions are
# never suppressed.


BASE_Q = "What standard agent evaluation metrics are commonly interpreted by non-technical users?"
# Same keywords reordered + two added words -> Jaccard ~0.64 (suppressed).
PARAPHRASE_Q = "Which standard agent evaluation metrics can non-technical users interpret without ML expertise?"
# Shares only a few keywords with BASE_Q -> Jaccard ~0.28 (inserted).
DISTINCT_Q = "How do agent evaluation libraries score trajectories when the final output is missing due to timeout or failure?"


def test_add_question_guard_suppresses_auto_near_duplicate_of_open():
    """Auto-source near-duplicate of an open question is not inserted."""
    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": BASE_Q, "priority": "high"}])

    result = add_question(plan_id, PARAPHRASE_Q, discovered_by="extraction")

    assert result is None
    assert get_plan_stats(plan_id)["pending"] == 1


def test_add_question_guard_inserts_distinct_auto_question():
    """Auto question below the similarity threshold is inserted."""
    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": BASE_Q, "priority": "high"}])

    result = add_question(plan_id, DISTINCT_Q, discovered_by="extraction")

    assert result is not None
    assert get_plan_stats(plan_id)["pending"] == 2


def test_add_question_guard_counts_resolved_questions():
    """Re-asking a resolved question is suppressed (the observed growth mode)."""
    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": BASE_Q, "priority": "high"}])
    q = get_pending_question(plan_id)
    set_question_resolved(q["id"])

    result = add_question(plan_id, PARAPHRASE_Q, discovered_by="dod_gaps")

    assert result is None
    assert get_plan_stats(plan_id).get("pending", 0) == 0


def test_add_question_guard_never_suppresses_user_questions():
    """User-typed questions bypass the guard even when near-duplicate."""
    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": BASE_Q, "priority": "high"}])

    result = add_question(plan_id, PARAPHRASE_Q, discovered_by="user")

    assert result is not None
    assert get_plan_stats(plan_id)["pending"] == 2


def test_add_question_guard_ignores_discarded_questions():
    """A discarded near-duplicate does not block a new question."""
    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": BASE_Q, "priority": "high"}])
    q = get_pending_question(plan_id)
    set_question_discarded(q["id"])

    result = add_question(plan_id, PARAPHRASE_Q, discovered_by="extraction")

    assert result is not None
    assert get_plan_stats(plan_id)["pending"] == 1


# ── Topic clustering (cluster_plan_questions) ─────────────────────
#
# Gives the question bank a readable topic structure (run.py topics).
# Deterministic keyword clustering — no LLM, no network.


def test_cluster_groups_similar_questions():
    """Two topical pairs cluster separately from each other."""
    plan_id = create_plan("Test", [])
    create_questions(
        plan_id,
        [
            {
                "question": "What are the latency trade-offs of continuous batching in vLLM?",
                "priority": "high",
            },
            {
                "question": "What latency trade-offs does SGLang continuous batching have?",
                "priority": "medium",
            },
            {
                "question": "How does TensorRT-LLM handle KV cache memory limits?",
                "priority": "medium",
            },
            {
                "question": "What KV cache memory limits does TGI enforce?",
                "priority": "low",
            },
        ],
    )
    clusters = cluster_plan_questions(plan_id)
    assert len(clusters) == 2
    texts = [[m["question"] for m in c["members"]] for c in clusters]
    assert any(len(t) == 2 and all("batching" in q for q in t) for t in texts)
    assert any(len(t) == 2 and all("cache" in q for q in t) for t in texts)
    # label is always one of the cluster's members
    for c in clusters:
        assert c["label"] in [m["question"] for m in c["members"]]
        assert sum(c["counts"].values()) == len(c["members"])


def test_cluster_excludes_discarded_questions():
    """Discarded questions do not appear in any cluster."""
    plan_id = create_plan("Test", [])
    create_questions(
        plan_id,
        [
            {
                "question": "What are the latency trade-offs of continuous batching in vLLM?"
            },
            {"question": "What KV cache memory limits does TGI enforce?"},
        ],
    )
    q = get_pending_question(plan_id)
    set_question_discarded(q["id"])

    clusters = cluster_plan_questions(plan_id)

    assert sum(len(c["members"]) for c in clusters) == 1


def test_cluster_is_deterministic_and_empty_plan():
    """Same bank → same labels; an empty plan yields no clusters."""
    plan_id = create_plan("Test", [])
    assert cluster_plan_questions(plan_id) == []
    create_questions(
        plan_id,
        [
            {
                "question": "What are the latency trade-offs of continuous batching in vLLM?"
            },
            {
                "question": "What latency trade-offs does SGLang continuous batching have?"
            },
            {"question": "What KV cache memory limits does TGI enforce?"},
        ],
    )
    first = cluster_plan_questions(plan_id)
    second = cluster_plan_questions(plan_id)
    assert [c["label"] for c in first] == [c["label"] for c in second]


# ── Quality Gate (Phase A: "No Evidence Found") ──────────────────


def test_quality_gate_insufficient_evidence():
    """When all findings are discarded, quality gate should flag insufficient evidence."""
    from db.knowledge import (
        create_finding,
        create_knowledge_entry,
        create_plan,
        create_questions,
        report_quality_gate,
        set_question_resolved,
        store_raw_result,
        get_pending_question,
        set_question_in_progress,
    )

    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q?", "priority": "high"}])
    q = get_pending_question(plan_id)
    set_question_in_progress(q["id"])
    store_raw_result(q["id"], plan_id, "Q?", "exa", "Some content", "Title")

    # Create knowledge entry first (FK requirement)
    entry_id = create_knowledge_entry(
        title="Generic",
        summary="Generic",
        topic_tags=["test"],
        full_text_path="",
        source_url="",
        discovered_by="test",
        full_text="",
    )

    # Create a discarded finding — summary must contain a _GENERIC_PHRASE
    create_finding(
        question_id=q["id"],
        raw_result_id=1,
        knowledge_entry_id=entry_id,
        facts=[{"claim": "The content indicates something", "confidence": "low"}],
        summary="The content indicates a finding",
        tags=["test"],
    )
    set_question_resolved(q["id"])

    result = report_quality_gate(plan_id)
    assert result["ok"] is False
    assert result["reason"] == "insufficient_evidence"
    assert result["stats"]["valid_findings"] == 0
    assert result["stats"]["discarded"] == 1


def test_quality_gate_no_sources():
    """When no raw results exist at all, quality gate should flag no sources."""
    from db.knowledge import (
        create_plan,
        create_questions,
        report_quality_gate,
        set_question_resolved,
        get_pending_question,
    )

    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q?", "priority": "high"}])
    q = get_pending_question(plan_id)
    set_question_resolved(q["id"])

    result = report_quality_gate(plan_id)
    assert result["ok"] is False
    assert result["reason"] == "no_sources"
    assert result["stats"]["sources"] == 0


def test_quality_gate_sufficient_evidence():
    """When valid findings exist, quality gate should pass."""
    from db.knowledge import (
        create_finding,
        create_knowledge_entry,
        create_plan,
        create_questions,
        report_quality_gate,
        set_question_resolved,
        store_raw_result,
        get_pending_question,
        set_question_in_progress,
    )

    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q?", "priority": "high"}])
    q = get_pending_question(plan_id)
    set_question_in_progress(q["id"])
    store_raw_result(q["id"], plan_id, "Q?", "exa", "Good content", "Title")

    entry_id = create_knowledge_entry(
        title="Specific",
        summary="Specific",
        topic_tags=["test"],
        full_text_path="",
        source_url="",
        discovered_by="test",
        full_text="",
    )

    create_finding(
        question_id=q["id"],
        raw_result_id=1,
        knowledge_entry_id=entry_id,
        facts=[{"claim": "A specific fact", "confidence": "high"}],
        summary="Specific finding",
        tags=["test"],
    )
    set_question_resolved(q["id"])

    result = report_quality_gate(plan_id)
    assert result["ok"] is True
    assert result["reason"] == "sufficient"
    assert result["stats"]["valid_findings"] == 1


# ── P1-5: Fallback Placeholder on Extraction Failure ───────────────


def test_accept_command_creates_degraded_evidence():
    """'accept <id>' resolves a failed question with a placeholder finding
    that records sources existed but extraction failed. The placeholder must
    count as valid evidence — the quality gate passes and readiness unblocks.
    """
    from nodes.curation import curate_knowledge_node
    from db.knowledge import get_question, report_quality_gate

    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q?", "priority": "high"}])
    q = get_pending_question(plan_id)
    set_question_in_progress(q["id"])
    rr_id = store_raw_result(
        q["id"], plan_id, "Q?", "exa", "Some content", "Title", "http://src"
    )
    set_question_failed(q["id"], "LLM returned invalid JSON")

    # The failed question blocks readiness...
    assert all_questions_are_resolved(plan_id) is False

    # ...until a human explicitly accepts the degraded evidence
    result = curate_knowledge_node(
        {"plan_ref": plan_id, "_human_response": f"accept {q['id']}"}
    )
    assert result.get("_error") is None
    assert get_question(q["id"])["status"] == "resolved"
    assert all_questions_are_resolved(plan_id) is True

    # The placeholder finding is tagged and linked to the raw result
    findings = get_plan_findings(plan_id)
    assert len(findings) == 1
    f = findings[0]
    assert "extraction-failed" in f["tags"]
    assert f["raw_result_id"] == rr_id
    assert "Extraction failed (LLM returned invalid JSON)" in f["summary"]

    # The knowledge entry records the first source URL
    entries = get_plan_entries(plan_id)
    assert len(entries) == 1
    assert entries[0]["source_url"] == "http://src"
    assert "extraction-failed" in entries[0]["topic_tags"]

    # The placeholder counts as valid evidence — the quality gate passes
    gate = report_quality_gate(plan_id)
    assert gate["ok"] is True
    assert gate["stats"]["valid_findings"] == 1
    assert gate["stats"]["discarded"] == 0


def test_accept_command_rejects_non_failed_questions():
    """'accept' only works on failed questions — anything else is a no-op
    with an error message, and unknown ids are reported."""
    from nodes.curation import curate_knowledge_node
    from db.knowledge import get_question

    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": "Q?", "priority": "high"}])
    q = get_pending_question(plan_id)

    result = curate_knowledge_node(
        {"plan_ref": plan_id, "_human_response": f"accept {q['id']}"}
    )
    assert "not failed" in result["_error"]
    assert get_question(q["id"])["status"] == "pending"

    result = curate_knowledge_node(
        {"plan_ref": plan_id, "_human_response": "accept q_nope"}
    )
    assert "No question" in result["_error"]


def test_filter_quality_keeps_degraded_evidence_placeholders():
    """Placeholders tagged 'extraction-failed' survive filter_quality even
    if their wording looks generic — they are deliberate, not filler."""
    from utils.quality import filter_quality

    placeholder = {
        "summary": "Extraction failed (timeout). 1 raw source(s) were found, "
        "but no structured findings could be produced.",
        "facts": [],
        "tags": ["extraction-failed"],
    }
    filler = {"summary": "The content indicates x", "facts": [], "tags": []}

    valid, discarded = filter_quality([placeholder, filler])
    assert len(valid) == 1
    assert valid[0]["tags"] == ["extraction-failed"]
    assert len(discarded) == 1
    assert discarded[0] is filler


# ── Cost Tracking (P1-6: tracker layering) ─────────────────────────


def test_cost_tracker_records_and_estimates():
    from utils.cost import CostTracker

    t = CostTracker()
    t.record_call("extraction", "gpt-4o", 1_000_000, 100_000)
    assert t.get_total_calls() == 1
    s = t.get_summary()
    assert s["input_tokens"] == 1_000_000
    assert s["output_tokens"] == 100_000
    assert s["per_task"]["extraction"] == 1
    # gpt-4o: $2.50/M input, $10.00/M output → 2.50 + 1.00
    assert t.get_estimated_cost() == 3.50
    assert "Cost Summary" in t.format_footer()


def test_cost_tracker_unknown_model_costs_zero():
    from utils.cost import CostTracker

    t = CostTracker()
    t.record_call("report", "some-local-model", 500, 250)
    assert t.get_estimated_cost() == 0.0
    assert t.get_summary()["model"] == "some-local-model"


def test_cost_tracker_mixed_models_priced_per_model():
    """A session that mixes models (e.g. an Anthropic research loop + a
    local report-writing process) must be priced per model, and the footer
    must show the mix — not attribute everything to the last model seen."""
    from utils.cost import CostTracker

    t = CostTracker()
    # Anthropic loop: 1M in / 100k out on claude-sonnet-5-5 ($3/$15 per M)
    t.record_call("extraction", "claude-sonnet-5-5", 1_000_000, 100_000)
    # Local report: 1M in / 100k out on qwen36_35 ($0)
    t.record_call("report", "qwen36_35", 1_000_000, 100_000)

    # Only the Anthropic tokens cost anything: 3.00 + 1.50
    assert t.get_estimated_cost() == 4.50

    s = t.get_summary()
    # Last-model field is kept for backward compat...
    assert s["model"] == "qwen36_35"
    # ...but the per-model breakdown shows both
    assert set(s["models"]) == {"claude-sonnet-5-5", "qwen36_35"}
    assert s["models"]["claude-sonnet-5-5"]["cost"] == 4.50
    assert s["models"]["qwen36_35"]["cost"] == 0.0
    assert s["models"]["qwen36_35"]["priced"] is True

    footer = t.format_footer(label="Plan: plan_test123")
    assert "Plan: plan_test123" in footer
    assert "claude-sonnet-5-5" in footer and "qwen36_35" in footer
    assert "per-model pricing" in footer
    assert "$4.5000" in footer


def test_cost_footer_flags_unknown_model_and_unrecorded_calls():
    """Unknown models are $0 but flagged (not silent); calls without usage
    metadata are counted and reported in the footer."""
    from utils.cost import CostTracker

    t = CostTracker()
    t.record_call("report", "mystery-model-9000", 500, 250)
    t.note_unrecorded()
    t.note_unrecorded()

    assert t.get_estimated_cost() == 0.0
    s = t.get_summary()
    assert s["models"]["mystery-model-9000"]["priced"] is False
    assert s["unrecorded_calls"] == 2

    footer = t.format_footer()
    assert "not in pricing table" in footer
    assert "2 call(s) had no usage metadata" in footer


def test_invoke_with_cost_records_to_active_tracker():
    """Layering (P1-6): utils/llm records to the utils/cost active tracker
    — no graph import — and no-ops cleanly when no app has been built."""
    from utils import llm as llm_utils
    from utils.cost import CostTracker, set_active_tracker, get_active_tracker

    saved = get_active_tracker()
    try:
        tracker = CostTracker()
        set_active_tracker(tracker)

        class Model:
            model_name = "gpt-4o"

            def invoke(self, prompt):
                return type(
                    "R",
                    (),
                    {"usage_metadata": {"input_tokens": 100, "output_tokens": 50}},
                )()

        llm_utils.invoke_with_cost(Model(), "hello", task="extraction")
        s = tracker.get_summary()
        assert s["per_task"]["extraction"] == 1
        assert s["input_tokens"] == 100
        assert s["output_tokens"] == 50

        # No active tracker → no crash, no recording
        set_active_tracker(None)
        llm_utils.invoke_with_cost(Model(), "hello", task="extraction")
        assert tracker.get_total_calls() == 1
    finally:
        set_active_tracker(saved)


def test_build_app_creates_fresh_cost_tracker():
    """Each build_app() gets its own tracker — no cross-app cost leakage
    (e.g. between test cases or sequential sessions in one process)."""
    from graph.workflow import build_app
    from langgraph.checkpoint.memory import MemorySaver
    from utils.cost import get_active_tracker

    build_app(MemorySaver())
    t1 = get_active_tracker()
    t1.record_call("planning", "gpt-4o", 10, 10)

    build_app(MemorySaver())
    t2 = get_active_tracker()
    assert t1 is not t2
    assert t2.get_total_calls() == 0
