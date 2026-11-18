"""
Finding quality filter — separate findings into (kept, discarded).

Used by consolidation, workflow report builder, and tests.
"""

from __future__ import annotations

from config.settings import search_config
from utils.search_routing import plan_search

_GENERIC_PHRASES = (
    "the provided text",
    "the article",
    "the content",
    "the document",
    "the source",
    "the paper",
    "the text mentions",
    "the content indicates",
    "according to",
    "the text describes",
    "the text outlines",
)


def filter_quality(findings: list[dict]) -> tuple[list[dict], list[dict]]:
    """Separate findings into (kept, discarded) based on quality signals.

    Discards findings whose summary or facts contain generic/filler phrases
    that add no informational value.
    """
    valid: list[dict] = []
    discarded: list[dict] = []
    for f in findings:
        # Degraded-evidence placeholders (extraction failed, sources exist)
        # are deliberate, not filler — they must reach the report.
        if "extraction-failed" in f.get("tags", []):
            valid.append(f)
            continue
        summary = f.get("summary", "").lower()
        if any(p in summary for p in _GENERIC_PHRASES):
            discarded.append(f)
            continue
        facts_text = " ".join(
            fact.get("claim", "") if isinstance(fact, dict) else str(fact)
            for fact in f.get("facts", [])
        ).lower()
        if any(p in facts_text for p in _GENERIC_PHRASES):
            discarded.append(f)
            continue
        valid.append(f)
    return valid, discarded


def is_science_topic(query: str) -> bool:
    """Detect if a query is science/math/ML related — arXiv should be used.

    Delegates to the tiered, word-boundary-aware routing in
    ``utils.search_routing`` and honors the configured ``force_arxiv``
    override so the consolidation gate stays consistent with the search
    node's per-question decision.
    """
    return plan_search(query, force=search_config.force_arxiv).use_arxiv
