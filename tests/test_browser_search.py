"""
Tests for browser_search module (nodes/browser_search.py).

Covers:
- HTML text extraction (static parser)
- URL fetching (static HTML)
- Content enrichment on result lists
- Edge cases (empty content, bad URLs, JS-only pages)
"""

from __future__ import annotations

import os
import sys


sys.path.insert(0, os.path.dirname(__file__) + "/..")

from nodes.browser_search import (
    _extract_text_from_html,
    _fetch_page,
    enrich_with_browser,
    scrape_url,
)


# ── HTML text extraction ──────────────────────────────────────────


class TestTextExtractor:
    """Test the lightweight HTML text extractor."""

    def test_basic_html(self):
        html = "<html><body><p>Hello world</p></body></html>"
        text = _extract_text_from_html(html)
        assert "Hello world" in text

    def test_skips_script_tags(self):
        html = """
        <div>Hello</div>
        <script>alert('xss')</script>
        <div>World</div>
        """
        text = _extract_text_from_html(html)
        assert "Hello" in text
        assert "World" in text
        assert "alert" not in text

    def test_skips_style_tags(self):
        html = """
        <div>Hello</div>
        <style>.foo { color: red; }</style>
        <div>World</div>
        """
        text = _extract_text_from_html(html)
        assert "color" not in text
        assert "Hello" in text

    def test_skips_nav_header_footer(self):
        html = """
        <nav>Navigation links</nav>
        <header>Site header</header>
        <main>Main content here</main>
        <footer>Footer text</footer>
        """
        text = _extract_text_from_html(html)
        assert "Main content here" in text
        assert "Navigation links" not in text
        assert "Site header" not in text
        assert "Footer text" not in text

    def test_nested_non_content(self):
        html = "<div><script><p>Nested script</p></script></div>"
        text = _extract_text_from_html(html)
        assert "Nested script" not in text

    def test_heading_structure(self):
        html = """
        <h1>Heading 1</h1>
        <h2>Heading 2</h2>
        <h3>Heading 3</h3>
        """
        text = _extract_text_from_html(html)
        assert "Heading 1" in text
        assert "Heading 2" in text
        assert "Heading 3" in text

    def test_collapse_whitespace(self):
        html = "<p>  Multiple   spaces   here  </p>"
        text = _extract_text_from_html(html)
        assert "Multiple spaces here" == text.strip()

    def test_empty_html(self):
        text = _extract_text_from_html("")
        assert text == ""

    def test_no_text_tags(self):
        text = _extract_text_from_html("<html></html>")
        assert text == ""


# ── URL fetching ──────────────────────────────────────────────────


class TestFetchPage:
    """Test static HTML fetching."""

    def test_fetch_realsite(self):
        """Fetch a real URL and verify we get HTML back."""
        html = _fetch_page("https://httpbin.org/html", timeout=15)
        assert html is not None
        assert len(html) > 100
        assert "<html" in html

    def test_fetch_non_html(self):
        """Non-HTML content types should return None."""
        # JSON endpoint
        html = _fetch_page("https://httpbin.org/json", timeout=15)
        assert html is None

    def test_fetch_bad_url(self):
        """Unreachable URLs should return None."""
        html = _fetch_page("https://this-domain-does-not-exist-12345.com", timeout=5)
        assert html is None

    def test_fetch_short_timeout(self):
        """Very short timeout should catch slow/unreachable sites."""
        html = _fetch_page("https://httpbin.org/delay/5", timeout=2)
        assert html is None


# ── Content enrichment ────────────────────────────────────────────


class TestEnrichWithBrowser:
    """Test the enrich_with_browser function."""

    def test_empty_results(self):
        assert enrich_with_browser([]) == []

    def test_already_good_content(self):
        """Results with >= 200 chars content should not be enriched."""
        results = [
            {"title": "Test", "url": "https://httpbin.org/html", "content": "A" * 200}
        ]
        enriched = enrich_with_browser(results, max_pages=3)
        assert all(len(r.get("content", "")) >= 200 for r in enriched)

    def test_enrich_short_content(self):
        """Results with short content should be enriched (static fetch)."""
        results = [
            {"title": "Test", "url": "https://httpbin.org/html", "content": "Short"}
        ]
        enriched = enrich_with_browser(results, max_pages=3)
        assert len(enriched) == 1
        assert enriched[0].get("content_source") == "browser"
        assert len(enriched[0]["content"]) > 50

    def test_enrich_empty_content(self):
        """Results with empty content should be enriched."""
        results = [{"title": "Test", "url": "https://httpbin.org/html", "content": ""}]
        enriched = enrich_with_browser(results, max_pages=3)
        assert enriched[0].get("content_source") == "browser"

    def test_no_url_skipped(self):
        """Results without a URL should be skipped."""
        results = [{"title": "Test", "content": "Short"}]
        enriched = enrich_with_browser(results, max_pages=3)
        assert enriched == results

    def test_max_pages_limit(self):
        """Only up to max_pages should be scraped."""
        results = [
            {"title": f"Result {i}", "url": "https://httpbin.org/html", "content": ""}
            for i in range(10)
        ]
        enriched = enrich_with_browser(results, max_pages=2)
        # First 2 should be enriched, rest should stay unchanged
        enriched_count = sum(1 for r in enriched if r.get("content_source"))
        assert enriched_count <= 2


# ── scrape_url convenience function ───────────────────────────────


class TestScrapeUrl:
    """Test the scrape_url convenience function."""

    def test_scrape_real_url(self):
        result = scrape_url("https://httpbin.org/html", use_playwright=False)
        assert result is not None
        assert "url" in result
        assert "content" in result
        assert "title" in result
        assert len(result["content"]) > 50

    def test_scrape_bad_url(self):
        result = scrape_url(
            "https://this-domain-does-not-exist-12345.com", use_playwright=False
        )
        assert result is None

    def test_scrape_non_html(self):
        result = scrape_url("https://httpbin.org/json", use_playwright=False)
        assert result is None
