"""
Tests for citation validation & repair (utils/citations.py).

Regression context (live, 20261110_120357 report): the report LLM wrote its
own References section numbered differently from the canonical list it was
given, and cited TGI hardware support / framework licensing to the vLLM
arXiv paper. The references section must be rebuilt deterministically and
the mismatch surfaced in the stats.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/..")

from utils.citations import (
    canonical_references_text,
    extract_references_section,
    validate_citations,
    citation_stats_line,
)

SOURCES = [
    ("https://arxiv.org/abs/2309.06180", "Efficient Memory Management for LLM Serving"),
    ("https://github.com/llm-d/llm-d", "llm-d"),
    ("https://huggingface.co/docs/text-generation-inference", "TGI docs"),
]


def test_canonical_references_text_numbering():
    text = canonical_references_text(SOURCES)
    assert (
        "[1] Efficient Memory Management for LLM Serving — https://arxiv.org/abs/2309.06180"
        in text
    )
    assert (
        "[3] TGI docs — https://huggingface.co/docs/text-generation-inference" in text
    )
    assert canonical_references_text([]) == ""


def test_no_references_section_appends_canonical_list():
    report = "# Report\n\nBody with a citation [2] and another [99].\n"
    repaired, stats = validate_citations(report, SOURCES)
    assert "[2] llm-d — https://github.com/llm-d/llm-d" in repaired
    assert "## References" in repaired
    # [99] is outside the 3-entry canonical list
    assert stats["inline_refs"] == 2
    assert stats["invalid_inline"] == [99]
    assert stats["llm_ref_entries"] == 0
    assert "Citations:" in citation_stats_line(stats)


def test_llm_references_section_rebuilt_from_canonical():
    """The LLM's own Sources section (wrong numbering, one hallucinated
    entry) is dropped and replaced by the canonical list."""
    report = (
        "# Report\n\n"
        "TGI supports many accelerators [1].\n\n"
        "## Sources\n\n"
        "[1] Some blog post — https://blog.example.com/tgi\n"
        "[2] llm-d — https://github.com/llm-d/llm-d\n"
        "[3] vLLM paper — https://arxiv.org/abs/2309.06180\n"
    )
    repaired, stats = validate_citations(report, SOURCES)

    # Hallucinated blog entry is gone; canonical list present
    assert "blog.example.com" not in repaired
    assert "[1] Efficient Memory Management for LLM Serving" in repaired
    assert (
        "[3] TGI docs — https://huggingface.co/docs/text-generation-inference"
        in repaired
    )
    # Exactly one References section survives
    assert repaired.count("## References") == 1
    # 1 of 3 LLM entries matched a canonical source (llm-d, vLLM paper)
    assert stats["llm_ref_entries"] == 3
    assert stats["unmatched_refs"] == 1
    # Inline [1] in the body is within range → not flagged
    assert stats["invalid_inline"] == []


def test_repair_is_idempotent():
    """Running validation twice (e.g. report regenerated then re-checked)
    must be a no-op the second time."""
    report = "# Report\n\nBody [1].\n"
    first, _ = validate_citations(report, SOURCES)
    second, stats2 = validate_citations(first, SOURCES)
    assert second == first
    # The canonical section's own numbers are not counted as inline refs
    assert stats2["invalid_inline"] == []
    assert stats2["inline_refs"] == 1


def test_extract_references_section_stops_at_next_heading():
    report = (
        "# Report\n\n"
        "## Sources\n\n"
        "[1] A — https://a.example.com\n"
        "trailing non-entry line\n\n"
        "## Evidence Matrix\n\n"
        "| x | y | z |\n"
    )
    body, entries = extract_references_section(report)
    assert entries == [
        (1, "A — https://a.example.com"),
    ]
    assert "## Sources" not in body
    assert "## Evidence Matrix" in body
    assert "| x | y | z |" in body


def test_unmatched_entry_without_url():
    report = (
        "# Report\n\n"
        "Body [1].\n\n"
        "## References\n\n"
        "[1] A well-known textbook (no URL given)\n"
    )
    _, stats = validate_citations(report, SOURCES)
    # No URL → unverifiable → counted as unmatched
    assert stats["llm_ref_entries"] == 1
    assert stats["unmatched_refs"] == 1


def test_url_matching_ignores_trailing_punctuation_and_scheme():
    report = (
        "# Report\n\n"
        "Body [1].\n\n"
        "## Sources\n\n"
        "[1] llm-d — http://github.com/llm-d/llm-d/\n"
    )
    _, stats = validate_citations(report, SOURCES)
    # http vs https, trailing slash — still matches the canonical URL
    assert stats["unmatched_refs"] == 0
