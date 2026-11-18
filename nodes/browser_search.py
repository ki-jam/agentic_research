"""
Browser search — Playwright-powered page content enrichment.

Enriches Exa/Brave search results by scraping full page content from URLs
that have short or empty text content. Used as a fallback/enhancement to
API-based search, not as a primary search engine.

Usage:
    results = _search_exa(query)
    results = enrich_with_browser(results)  # adds content where missing
"""

from __future__ import annotations

import logging
import re
import urllib.request
import urllib.parse
from html.parser import HTMLParser
from typing import Optional

logger = logging.getLogger(__name__)


# ── Lightweight HTML content extractor (no JS needed) ─────────────


class _TextExtractor(HTMLParser):
    """Minimal HTML parser that extracts visible text from a page.

    Skips script, style, nav, header, footer, and other non-content elements.
    Returns cleaned text with normalised whitespace.
    """

    NON_CONTENT_TAGS = frozenset(
        {"script", "style", "noscript", "iframe", "svg", "nav", "header", "footer"}
    )

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip = False
        self._skip_depth = 0
        self._in_heading = False
        self._heading_stack: list[int] = []  # heading levels for structure

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]):
        tag = tag.lower()
        if tag in self.NON_CONTENT_TAGS:
            self._skip = True
            self._skip_depth = 1
            return

        if self._skip:
            self._skip_depth += 1
            return

        # Track heading levels for structural separators
        m = re.match(r"h([1-6])", tag)
        if m:
            level = int(m.group(1))
            # Close any deeper or equal headings (insert separator)
            while self._heading_stack and self._heading_stack[-1] >= level:
                self._parts.append("\n\n")
                self._heading_stack.pop()
            self._heading_stack.append(level)
            self._parts.append("\n\n")

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag in self.NON_CONTENT_TAGS:
            if self._skip_depth == 1:
                self._skip = False
            else:
                self._skip_depth -= 1
            return

        if self._skip:
            self._skip_depth -= 1
            return

        m = re.match(r"h([1-6])", tag)
        if m:
            level = int(m.group(1))
            if self._heading_stack and self._heading_stack[-1] == level:
                self._parts.append("\n\n")
                self._heading_stack.pop()

    def handle_data(self, data: str):
        if not self._skip:
            text = data.strip()
            if text:
                self._parts.append(text + " ")


def _extract_text_from_html(html: str) -> str:
    """Extract visible text from HTML, skipping non-content elements."""
    parser = _TextExtractor()
    parser.feed(html)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", "".join(parser._parts)).strip()
    return text


def _fetch_page(url: str, timeout: int = 15) -> Optional[str]:
    """Fetch raw HTML from a URL."""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0)",
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if (
                "text/html" not in content_type
                and "application/xhtml" not in content_type
            ):
                return None
            raw = resp.read()
            # Try UTF-8, fall back to latin-1
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                return raw.decode("latin-1", errors="replace")
    except Exception:
        return None


def enrich_with_browser(results: list[dict], max_pages: int = 3) -> list[dict]:
    """Enrich search results by extracting full content from pages.

    For each result with short/empty content (< 200 chars), fetches the page
    and extracts visible text. Only enriches up to max_pages to avoid slow
    scraping.

    Uses only urllib + HTML parsing — no browser needed for static pages.
    Falls back to Playwright for JS-rendered pages if configured.

    Args:
        results: list of search result dicts with 'url' and 'content' keys
        max_pages: maximum number of pages to scrape (default: 3)

    Returns:
        Results list with enriched content (modified in-place + new fields)
    """
    if not results:
        return results

    scraped = 0
    for r in results:
        if scraped >= max_pages:
            break

        content = r.get("content", "")
        if content and len(content) >= 200:
            continue  # already has good content

        url = r.get("url", "")
        if not url:
            continue

        # Try static HTML fetch first
        html = _fetch_page(url)
        if html:
            text = _extract_text_from_html(html)
            if text and len(text) > 50:
                r["content"] = text[:12000]  # cap content length
                r["content_source"] = "browser"
                r["title"] = r.get("title", "") or text[:80]
                scraped += 1
                continue

        # Static fetch failed — try Playwright
        if _try_playwright_fetch(r):
            scraped += 1

    return results


def _try_playwright_fetch(result: dict) -> bool:
    """Try to fetch content using Playwright (JS-rendered pages).

    Args:
        result: a search result dict to enrich (modified in-place)

    Returns:
        True if Playwright successfully enriched the result
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False

    url = result.get("url", "")
    if not url:
        return False

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                page.goto(url, timeout=15000, wait_until="domcontentloaded")
                # Try to extract main content
                text = page.inner_text(
                    "article, main, .content, .post, .article-body, #content, .post-content"
                )
                if not text:
                    text = page.inner_text("body")
            except Exception:
                text = ""
            finally:
                browser.close()

        if text and len(text) > 50:
            result["content"] = text[:12000]
            result["content_source"] = "playwright"
            result["title"] = (
                result.get("title", "") or page.title()
                if "page" in dir()
                else result.get("title", "")
            )
            return True
    except Exception:
        pass

    return False


def scrape_url(
    url: str, use_playwright: bool = False, timeout: int = 15
) -> Optional[dict]:
    """Scrape a single URL and return extracted content.

    Convenience function for one-off scraping (e.g., from the CLI).

    Args:
        url: URL to scrape
        use_playwright: whether to use Playwright for JS-rendered pages
        timeout: page load timeout in seconds

    Returns:
        Dict with 'title', 'url', 'content' or None on failure
    """
    # Static fetch
    html = _fetch_page(url, timeout=timeout)
    if html:
        text = _extract_text_from_html(html)
        if text and len(text) > 50:
            return {"title": text[:80], "url": url, "content": text[:12000]}

    # Playwright fallback
    if use_playwright:
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                try:
                    page.goto(
                        url, timeout=timeout * 1000, wait_until="domcontentloaded"
                    )
                    text = page.inner_text(
                        "article, main, .content, .post, .article-body, #content"
                    )
                    if not text:
                        text = page.inner_text("body")
                    title = page.title()
                except Exception:
                    text = ""
                    title = url.split("/")[-1] or "Untitled"
                finally:
                    browser.close()

            if text and len(text) > 50:
                return {"title": title, "url": url, "content": text[:12000]}
        except ImportError:
            pass

    return None
