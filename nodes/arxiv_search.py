"""
arXiv search — typed, rate-limited search via the arxiv Python package.

Replaces the raw XML parsing in _search_arxiv with the official arxiv
package (pip install arxiv), which provides:
  - Typed Result objects (title, authors, summary, published, pdf_url, etc.)
  - Proper rate-limit handling (3s between requests)
  - Full query syntax (ti:, au:, abs:, cat:, all:, AND, OR, ANDNOT)
  - Pagination support (start, max_results)
  - Sorting (relevance, submittedDate, lastUpdatedDate)

Usage:
    results = search_arxiv("transformer attention mechanism")
    for r in results:
        print(r.title, r.summary[:200])
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from typing import Optional

from config.settings import search_config

logger = logging.getLogger(__name__)

# ── Query rewriting for arXiv ────────────────────────────────────────
#
# arXiv ANDs every term in a query, so conversational filler ("what", "how",
# "the", "about") silently excludes papers that don't contain those exact
# words. We strip a conservative stopword list before searching, but pass the
# query through UNCHANGED when it already uses arXiv's field prefixes
# (ti:/au:/abs:/cat:/all:) or explicit boolean operators (AND/OR/ANDNOT) —
# those are intentional and must not be mangled.

_ARXIV_STOPWORDS = frozenset(
    {
        # articles / prepositions / conjunctions
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "as",
        "into",
        "over",
        "under",
        "about",
        "between",
        "among",
        "via",
        "per",
        "than",
        "then",
        "so",
        "if",
        # pronouns / determiners
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "it",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "this",
        "that",
        "these",
        "those",
        "such",
        "there",
        "here",
        # auxiliaries / modals
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "can",
        "could",
        "should",
        "would",
        "will",
        "shall",
        "may",
        "might",
        "must",
        # interrogatives / fillers
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "please",
        "not",
        "no",
        "yes",
        "too",
        "very",
        "also",
        "just",
        "again",
        "further",
        "once",
        "out",
        "up",
        "down",
        "off",
        # common research-prompt verbs / nouns (safe to drop)
        "tell",
        "explain",
        "describe",
        "compare",
        "overview",
        "advances",
        "latest",
        "new",
        "state",
        "used",
        "using",
        "use",
        "make",
        "makes",
        "get",
        "gets",
        "find",
        "finds",
        "show",
        "shows",
        "long",
    }
)

_FIELD_PREFIX_RE = re.compile(r"^\s*(ti|au|abs|cat|all):", re.IGNORECASE)
_BOOL_OP_RE = re.compile(r"\b(AND|OR|ANDNOT)\b")


def rewrite_for_arxiv(query: str) -> str:
    """Strip conversational stopwords before sending a query to arXiv.

    arXiv ANDs every term, so filler words exclude papers that lack them.
    Removes a conservative stopword list, but returns the query unchanged
    when it already uses field prefixes (ti:/au:/abs:/cat:/all:) or explicit
    boolean operators, and never returns an empty query (if every token was
    a stopword, the original is kept).
    """
    q = (query or "").strip()
    if not q:
        return q
    if _FIELD_PREFIX_RE.match(q):
        return q
    if _BOOL_OP_RE.search(q):
        return q
    kept = [
        w for w in q.split() if w.strip("()[]{}\"'").lower() not in _ARXIV_STOPWORDS
    ]
    if not kept:
        return q
    return " ".join(kept)


def search_arxiv(
    query: str,
    max_results: Optional[int] = None,
    sort_by: str = "relevance",
    sort_order: str = "descending",
    categories: Optional[list[str]] = None,
) -> list[dict]:
    """Search arXiv and return formatted results.

    Uses the arxiv Python package for typed results and built-in
    rate-limit handling.

    Args:
        query: arXiv search query (supports field prefixes: ti:, au:, abs:,
               cat:, all: — combined with AND, OR, ANDNOT)
        max_results: number of results (default: from search_config)
        sort_by: "relevance", "submittedDate", or "lastUpdatedDate"
        sort_order: "ascending" or "descending"
        categories: optional list of arXiv categories to filter (e.g., ["cs.AI"])

    Returns:
        List of result dicts: {title, url, content, authors, categories,
        published, pdf_url}
    """
    max_results = max_results or search_config.arxiv_max_results

    try:
        import arxiv

        # Map sort_by string to arxiv.SortCriterion enum
        sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "submittedDate": arxiv.SortCriterion.SubmittedDate,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
        }
        sort_criterion = sort_map.get(sort_by, arxiv.SortCriterion.Relevance)
        sort_order_enum = (
            arxiv.SortOrder.Ascending
            if sort_order == "ascending"
            else arxiv.SortOrder.Descending
        )

        # Build the search (strip conversational stopwords first)
        search_query = rewrite_for_arxiv(query)
        if categories:
            # If query doesn't already have a category filter, prepend one
            if not any(c.strip() in query for c in categories):
                cat_clause = " AND ".join(f"cat:{c}" for c in categories)
                search_query = f"({search_query}) AND ({cat_clause})"

        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=sort_criterion,
            sort_order=sort_order_enum,
        )

        # Fetch results (Client handles rate limiting: 3s between requests)
        client = arxiv.Client()
        results = []
        for paper in client.results(search):
            results.append(_format_result(paper))

        logger.info(f"[arXiv] Found {len(results)} results for: {query[:60]}")
        return results

    except ImportError:
        logger.warning("arxiv package not installed — falling back to raw API")
        return _search_arxiv_fallback(query, max_results)
    except Exception as e:
        logger.error(f"[arXiv] Search failed: {e}")
        return []


def get_paper_by_id(paper_id: str) -> Optional[dict]:
    """Fetch a single paper by arXiv ID (e.g., '2402.03300' or '2402.03300v1').

    Args:
        paper_id: arXiv ID without the 'abs/' prefix

    Returns:
        Result dict or None if not found
    """
    try:
        import arxiv

        search = arxiv.Search(id_list=[paper_id])
        client = arxiv.Client()
        papers = list(client.results(search))
        if papers:
            return _format_result(papers[0])
        return None
    except Exception as e:
        logger.error(f"[arXiv] Fetch by ID failed: {e}")
        return None


def search_by_category(
    category: str,
    max_results: Optional[int] = None,
    sort_by: str = "submittedDate",
) -> list[dict]:
    """Search arXiv by category (e.g., 'cs.AI', 'cs.LG', 'stat.ML').

    Args:
        category: arXiv category code
        max_results: number of results
        sort_by: sort criterion

    Returns:
        List of result dicts
    """
    query = f"cat:{category}"
    return search_arxiv(
        query,
        max_results=max_results,
        sort_by=sort_by,
        sort_order="descending",
    )


def _format_result(paper) -> dict:
    """Format an arxiv.Result into a standard search result dict.

    Args:
        paper: arxiv.Result object from the arxiv package

    Returns:
        Dict with title, url, content, authors, categories, published, pdf_url
    """
    # Clean newlines from title and summary
    title = paper.title.replace("\n", " ").strip()
    summary = paper.summary.replace("\n", " ").strip()

    # Build authors list
    authors = [author.name for author in paper.authors]

    # Published date
    published = paper.published.strftime("%Y-%m-%d") if paper.published else ""

    # Categories
    categories = [str(c) for c in paper.categories] if paper.categories else []

    # PDF link
    pdf_url = paper.pdf_url or ""

    # arXiv abstract URL
    abs_url = (
        paper.entry_id
        or f"https://arxiv.org/abs/{paper.entry_id.split('/')[-1] if paper.entry_id else ''}"
    )

    return {
        "title": title,
        "url": abs_url,
        "content": f"Authors: {', '.join(authors)}\nCategories: {', '.join(categories)}\nPublished: {published}\n\n{summary}",
        "authors": authors,
        "categories": categories,
        "published": published,
        "pdf_url": pdf_url,
        "arxiv_id": paper.entry_id.split("/")[-1] if paper.entry_id else "",
    }


# ── Fallback: raw XML parsing (when arxiv package not installed) ────


def _search_arxiv_fallback(query: str, max_results: int = 10) -> list[dict]:
    """Fallback: raw XML parsing via urllib (no arxiv package needed).

    Used when the arxiv package is not installed. Less reliable than the
    typed package version.
    """
    import xml.etree.ElementTree as ET

    # Strip conversational stopwords (arXiv ANDs every term) and use https.
    query = rewrite_for_arxiv(query)
    url = (
        f"https://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}"
        f"&max_results={max_results}"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            xml_data = resp.read()
        root = ET.fromstring(xml_data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        results = []
        for entry in entries:
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            id_el = entry.find("atom:id", ns)
            title = title_el.text.replace("\n", " ") if title_el is not None else ""
            summary = (
                summary_el.text.replace("\n", " ") if summary_el is not None else ""
            )
            link = id_el.text if id_el is not None else ""
            authors = [
                a.find("atom:name", ns).text
                for a in entry.findall("atom:author", ns)
                if a.find("atom:name", ns) is not None
            ]
            results.append(
                {
                    "title": title,
                    "url": link,
                    "content": f"Authors: {', '.join(authors)}\n{summary}",
                    "authors": authors,
                }
            )
        return results
    except Exception as e:
        logger.error(f"[arXiv fallback] Failed: {e}")
        return []
