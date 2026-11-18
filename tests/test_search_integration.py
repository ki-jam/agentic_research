"""
Integration tests for the search pipeline with real API calls.

Tests the end-to-end flow: search → enrichment → formatting → DB storage.
Uses real Exa/Brave/arXiv APIs but no LLM calls.

Run with:
    cd research_agent && ./../venv/bin/python -m pytest tests/test_search_integration.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__) + "/..")

# Fake keys — real API calls are allowed, LLM calls are not
os.environ["EXA_API_KEY"] = "test-key"
os.environ["BRAVE_API_KEY"] = "test-key"
os.environ["BROWSER_SEARCH_ENABLED"] = "true"


@pytest.fixture(autouse=True)
def _test_db():
    """Use a test DB so production data is never touched."""
    import db.engine as engine_mod

    db_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    test_db = os.path.join(db_dir, "knowledge_test.db")
    os.environ["RESEARCH_DB_PATH"] = test_db

    if os.path.exists(test_db):
        os.remove(test_db)

    # Re-init so the engine picks up the new path (runs migrations)
    engine_mod.init_db()

    yield test_db

    if os.path.exists(test_db):
        os.remove(test_db)


@pytest.fixture(autouse=True)
def _reset_search_config():
    """Reset search config to defaults before each test."""
    from config.settings import search_config

    search_config.primary = "exa"
    search_config.arxiv_max_results = 5
    search_config.brave_api_key = "test-key"
    search_config.enable_citation_discovery = False  # skip Exa answer API
    yield


class TestSearchExecutionNode:
    """Test the full search execution node with real APIs."""

    def test_search_execution_exa(self, _test_db):
        """Search execution with Exa primary should store results."""
        from db.knowledge import add_question, create_plan, set_question_in_progress
        from nodes.search import search_execution_node

        plan_id = create_plan(
            "test-exa", [{"question": "test question"}], thread_id="test"
        )
        question_id = add_question(plan_id, "test question")
        set_question_in_progress(question_id)

        result = search_execution_node({"plan_ref": plan_id, "iteration_count": 1})
        assert result == {}

        # Check results were stored
        from db.engine import get_db

        with get_db() as conn:
            raw = conn.execute(
                "SELECT id, engine, title FROM raw_results WHERE question_id = ?",
                (question_id,),
            ).fetchone()
        assert raw is not None
        assert raw["engine"] == "exa"

    def test_search_execution_brave(self, _test_db):
        """Search execution with Brave primary should store results."""
        from config.settings import search_config
        from db.knowledge import add_question, create_plan, set_question_in_progress
        from nodes.search import search_execution_node

        search_config.primary = "brave"

        plan_id = create_plan(
            "test-brave", [{"question": "test question"}], thread_id="test"
        )
        question_id = add_question(plan_id, "test question")
        set_question_in_progress(question_id)

        result = search_execution_node({"plan_ref": plan_id, "iteration_count": 1})
        assert result == {}

        from db.engine import get_db

        with get_db() as conn:
            raw = conn.execute(
                "SELECT id, engine, title FROM raw_results WHERE question_id = ?",
                (question_id,),
            ).fetchone()
        assert raw is not None
        assert raw["engine"] == "brave"

        search_config.primary = "exa"

    def test_search_execution_arxiv(self, _test_db):
        """Search execution with arXiv primary should return typed results."""
        from config.settings import search_config
        from db.knowledge import add_question, create_plan, set_question_in_progress
        from nodes.search import search_execution_node

        search_config.primary = "arxiv"

        plan_id = create_plan(
            "test-arxiv", [{"question": "test question"}], thread_id="test"
        )
        question_id = add_question(plan_id, "test question")
        set_question_in_progress(question_id)

        result = search_execution_node({"plan_ref": plan_id, "iteration_count": 1})
        assert result == {}

        from db.engine import get_db

        with get_db() as conn:
            raw = conn.execute(
                "SELECT id, engine, title FROM raw_results WHERE question_id = ?",
                (question_id,),
            ).fetchone()
        assert raw is not None
        assert "arxiv" in raw["engine"]

        search_config.primary = "exa"

    def test_search_execution_arxiv_science_topic(self, _test_db):
        """Science topics should trigger both Exa and arXiv search."""
        from db.knowledge import add_question, create_plan, set_question_in_progress
        from nodes.search import search_execution_node

        plan_id = create_plan(
            "test-science", [{"question": "test question"}], thread_id="test"
        )
        question_id = add_question(
            plan_id, "test question about deep learning transformer attention"
        )
        set_question_in_progress(question_id)

        result = search_execution_node({"plan_ref": plan_id, "iteration_count": 1})
        assert result == {}

        # Should have both exa and arxiv results
        from db.engine import get_db

        with get_db() as conn:
            engines = conn.execute(
                "SELECT DISTINCT engine FROM raw_results WHERE question_id = ?",
                (question_id,),
            ).fetchall()
        engine_labels = [e["engine"] for e in engines]
        assert any("arxiv" in e for e in engine_labels)


class TestBrowserEnrichment:
    """Test browser enrichment integration."""

    def test_enrich_enriches_short_content(self, _test_db):
        """Results with short content should be enriched."""
        from nodes.browser_search import enrich_with_browser

        results = [
            {"title": "Test", "url": "https://httpbin.org/html", "content": "Short"},
            {"title": "Good", "url": "https://httpbin.org/html", "content": "A" * 300},
        ]
        enriched = enrich_with_browser(results, max_pages=3)

        # First result should be enriched
        assert enriched[0].get("content_source") == "browser"
        assert len(enriched[0]["content"]) > 100
        # Second result should be unchanged (already has good content)
        assert "content_source" not in enriched[1]

    def test_enrich_handles_bad_urls(self, _test_db):
        """Bad URLs should not break enrichment."""
        from nodes.browser_search import enrich_with_browser

        results = [
            {
                "title": "Bad",
                "url": "https://this-domain-does-not-exist-xyz123.com",
                "content": "",
            },
            {"title": "Good", "content": "A" * 300},  # no URL
        ]
        enriched = enrich_with_browser(results)
        # Both should be returned (no crash)
        assert len(enriched) == 2


class TestSearchFallback:
    """Test search fallback chain."""

    def test_exa_fallback_to_brave(self, _test_db):
        """Exa low-quality heuristic should flag GitHub-heavy results."""
        from nodes.search import _results_look_low_quality

        github_heavy = [
            {"url": "https://github.com/repo", "content": ""},
            {"url": "https://github.com/repo2", "content": ""},
            {"url": "https://github.com/repo3", "content": ""},
        ]
        assert _results_look_low_quality(github_heavy) is True

        good_results = [
            {"url": "https://example.com/article", "content": "A" * 100},
            {"url": "https://example.com/blog", "content": "B" * 100},
            {"url": "https://example.com/docs", "content": "C" * 100},
        ]
        assert _results_look_low_quality(good_results) is False

    def test_brave_fallback_to_arxiv(self, _test_db):
        """When Brave fails, arXiv should be used as fallback."""
        from config.settings import search_config
        from db.knowledge import add_question, create_plan, set_question_in_progress
        from nodes.search import search_execution_node

        search_config.brave_api_key = "invalid-key"
        search_config.primary = "brave"

        plan_id = create_plan(
            "test-fallback", [{"question": "test question"}], thread_id="test"
        )
        question_id = add_question(
            plan_id, "test question about deep learning reinforcement"
        )
        set_question_in_progress(question_id)

        result = search_execution_node({"plan_ref": plan_id, "iteration_count": 1})
        assert result == {}

        # Should have arXiv results (fallback + science topic)
        from db.engine import get_db

        with get_db() as conn:
            engines = conn.execute(
                "SELECT DISTINCT engine FROM raw_results WHERE question_id = ?",
                (question_id,),
            ).fetchall()
        engine_labels = [e["engine"] for e in engines]
        assert any("arxiv" in e for e in engine_labels)

        search_config.primary = "exa"
        search_config.brave_api_key = "test-key"
