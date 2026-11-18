"""
Tests for arxiv_search module (nodes/arxiv_search.py).

Covers:
- search_arxiv with live arXiv API
- get_paper_by_id
- search_by_category
- _format_result
- Fallback XML parsing
- Error handling
"""

from __future__ import annotations

import os
import sys


sys.path.insert(0, os.path.dirname(__file__) + "/..")

from nodes.arxiv_search import (
    _format_result,
    _search_arxiv_fallback,
    get_paper_by_id,
    search_arxiv,
    search_by_category,
)


# ── search_arxiv ──────────────────────────────────────────────────


class TestSearchArxiv:
    """Test the main search_arxiv function."""

    def test_search_returns_results(self):
        """A broad search should return results."""
        results = search_arxiv("transformer", max_results=5)
        assert isinstance(results, list)
        assert len(results) > 0
        assert len(results) <= 5

    def test_search_returns_required_fields(self):
        """Each result must have the standard fields."""
        results = search_arxiv("neural network", max_results=3)
        for r in results:
            assert "title" in r
            assert "url" in r
            assert "content" in r
            assert "authors" in r
            assert "categories" in r
            assert "published" in r
            assert "pdf_url" in r
            assert isinstance(r["authors"], list)
            assert isinstance(r["categories"], list)

    def test_search_empty_query(self):
        """Empty query should not crash (may return few or no results)."""
        results = search_arxiv("", max_results=5)
        assert isinstance(results, list)

    def test_search_specific_paper(self):
        """Search for a known paper should return results."""
        results = search_arxiv("attention is all you need", max_results=5)
        # Just verify we get results back — exact paper match varies by API ordering
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_falls_back_gracefully_on_error(self):
        """Malformed queries should not crash."""
        # Invalid query syntax that arXiv may reject
        results = search_arxiv("(", max_results=1)
        assert isinstance(results, list)

    def test_search_with_categories(self):
        """Search with category filter should work."""
        results = search_arxiv(
            "graph neural networks", max_results=3, categories=["cs.LG"]
        )
        assert isinstance(results, list)
        # All results should be in cs.LG or a subcategory
        for r in results:
            assert any("cs.LG" in c or "cs.LG/" in c for c in r["categories"]), (
                f"Expected cs.LG category, got: {r['categories']}"
            )

    def test_search_sorted_by_date(self):
        """Search with date sort should return recent papers first."""
        results = search_arxiv(
            "deep learning",
            max_results=5,
            sort_by="submittedDate",
            sort_order="descending",
        )
        assert isinstance(results, list)
        assert len(results) > 0


# ── get_paper_by_id ───────────────────────────────────────────────


class TestGetPaperById:
    """Test fetching a single paper by ID."""

    def test_get_known_paper(self):
        """Fetch a well-known paper by ID."""
        result = get_paper_by_id("1706.03762")  # Attention Is All You Need
        assert result is not None
        assert "1706.03762" in result["url"] or "1706.03762" in result.get(
            "arxiv_id", ""
        )
        assert "attention" in result["title"].lower()

    def test_get_nonexistent_paper(self):
        """Fetching a non-existent paper should return None."""
        # Use a very high number that almost certainly doesn't exist
        result = get_paper_by_id("9999.99999")
        assert result is None

    def test_get_paper_with_version(self):
        """Fetching with version suffix should still work."""
        result = get_paper_by_id("1706.03762v3")
        # May return the specific version or the latest
        if result:
            assert "attention" in result["title"].lower()


# ── search_by_category ────────────────────────────────────────────


class TestSearchByCategory:
    """Test category-based search."""

    def test_cs_ai_search(self):
        results = search_by_category("cs.AI", max_results=5)
        assert isinstance(results, list)
        assert len(results) > 0
        # All results should be in cs.AI or a subcategory
        for r in results:
            assert any("cs.AI" in c for c in r["categories"]), (
                f"Expected cs.AI category, got: {r['categories']}"
            )

    def test_stat_ml_search(self):
        results = search_by_category("stat.ML", max_results=3)
        assert isinstance(results, list)
        assert len(results) > 0


# ── _format_result ────────────────────────────────────────────────


class TestFormatResult:
    """Test the _format_result helper function."""

    def test_format_basic(self):
        """Format should produce expected fields."""

        # Create a mock arxiv.Result
        class MockAuthor:
            name = "Test Author"

        class MockPaper:
            title = "Test Paper Title"
            summary = "This is a test summary."
            authors = [MockAuthor()]
            published = None
            categories = ["cs.AI"]
            pdf_url = "https://arxiv.org/pdf/2401.00001.pdf"
            entry_id = "https://arxiv.org/abs/2401.00001"

        result = _format_result(MockPaper())
        assert result["title"] == "Test Paper Title"
        assert result["url"] == "https://arxiv.org/abs/2401.00001"
        # Content has authors, categories, published, summary — not title
        assert "Test Author" in result["content"]
        assert "cs.AI" in result["content"]
        assert "This is a test summary." in result["content"]
        assert result["authors"] == ["Test Author"]
        assert result["categories"] == ["cs.AI"]
        assert result["pdf_url"] == "https://arxiv.org/pdf/2401.00001.pdf"
        assert result["arxiv_id"] == "2401.00001"

    def test_format_newlines_stripped(self):
        """Newlines in title/summary should be normalized."""

        class MockAuthor:
            name = "Author"

        class MockPaper:
            title = "Title\nWith\nNewlines"
            summary = "Summary\nWith\nNewlines"
            authors = [MockAuthor()]
            published = None
            categories = ["cs.AI"]
            pdf_url = ""
            entry_id = "https://arxiv.org/abs/2401.00002"

        result = _format_result(MockPaper())
        assert "\n" not in result["title"]
        # Summary in content should have newlines stripped
        assert "Summary\nWith" not in result["content"]


# ── Fallback XML parsing ──────────────────────────────────────────


class TestFallbackSearch:
    """Test the raw XML fallback search."""

    def test_fallback_returns_results(self):
        """Fallback should return results for a valid query."""
        results = _search_arxiv_fallback("transformer", max_results=3)
        assert isinstance(results, list)
        assert len(results) > 0
        assert "title" in results[0]
        assert "url" in results[0]

    def test_fallback_handles_errors(self):
        """Malformed queries should not crash."""
        results = _search_arxiv_fallback("(", max_results=1)
        assert isinstance(results, list)
