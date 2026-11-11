"""
Citation validation and repair for final reports.

The report LLM is given a canonical numbered reference list (built from the
plan's raw_results + curated entries) and asked to cite it with [n] markers.
In practice it sometimes:

- writes its own "References"/"Sources" section with numbers that don't
  match the canonical list it was given, and
- attaches a claim to a source that has nothing to do with it (observed
  live on 20261110: TGI hardware support + framework licensing cited to
  the vLLM arXiv paper, which was reference [1]).

This module makes the references section deterministic (rebuilt from the
canonical list, which is the only source of truth) and reports how badly
the LLM's inline citations and its own reference entries missed the
canonical list. The stats are surfaced in the report's Research Stats
block so misattribution is visible in the artifact, not hidden in it.
"""

from __future__ import annotations

import re

# Headings the report LLM uses for its own reference list. Exact match (no
# trailing words) so body sections like "## Sources and methods" are never
# mistaken for the reference list.
_REF_HEADING = re.compile(
    r"^#{1,4}\s*(references|sources|bibliography|works cited)\s*$", re.IGNORECASE
)
# Any markdown heading (used to find the end of a section).
_ANY_HEADING = re.compile(r"^#{1,6}\s")
# Inline citation markers like [1] or [12] — not part of a longer number.
_INLINE_REF = re.compile(r"(?<!\d)\[(\d{1,3})\](?!\d)")
# Numbered reference entries: "[1] Title — https://..."
_REF_ENTRY = re.compile(r"^\[(\d{1,3})\]\s*(.+)$")
_URL = re.compile(r"https?://\S+")


def canonical_references_text(sources_list: list[tuple[str, str]]) -> str:
    """Numbered reference list for the canonical (url, title) source list.

    The same text is (a) shown to the report LLM so it knows the
    numbering, and (b) appended to the final report, so the numbers in
    the body and the list at the bottom always agree.
    """
    if not sources_list:
        return ""
    lines = ["\n\n## References\n"]
    for i, (url, title) in enumerate(sources_list, 1):
        lines.append(f"[{i}] {title} — {url}")
    return "\n".join(lines)


def _normalize_url(url: str) -> str:
    url = (url or "").strip().rstrip(".,;)]")
    url = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
    return url.lower().rstrip("/")


def extract_references_section(
    report_text: str,
) -> tuple[str, list[tuple[int, str]]]:
    """Split report_text into (body_without_refs, llm_ref_entries).

    llm_ref_entries is a list of (number, entry_text) parsed from the
    report's own References/Sources section, if it has one. Returns
    (report_text, []) when no such section exists.
    """
    lines = report_text.split("\n")
    start = None
    for i, line in enumerate(lines):
        if _REF_HEADING.match(line.strip()):
            start = i
            break
    if start is None:
        return report_text, []

    # Section ends at the next heading, or the Research Stats / Evidence
    # Matrix / Cost Summary blocks (which are appended after it).
    end = len(lines)
    for j in range(start + 1, len(lines)):
        line = lines[j].strip()
        if _ANY_HEADING.match(line):
            end = j
            break
        if line.startswith("--- Research Stats") or line.startswith("--- Cost Summary"):
            end = j
            break

    entries: list[tuple[int, str]] = []
    for line in lines[start + 1 : end]:
        m = _REF_ENTRY.match(line.strip())
        if m:
            entries.append((int(m.group(1)), m.group(2).strip()))

    body = "\n".join(lines[:start] + lines[end:])
    # Collapse the blank lines left where the section was
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body, entries


def validate_citations(
    report_text: str, sources_list: list[tuple[str, str]]
) -> tuple[str, dict]:
    """Repair the references section and report citation health.

    Returns (repaired_report, stats) where stats has:
      inline_refs        — count of [n] markers in the report body
      invalid_inline     — sorted list of [n] numbers with no canonical entry
      llm_ref_entries    — entries in the LLM's own reference section
      unmatched_refs     — of those, entries whose URL matches no canonical
                           source (hallucinated or misattributed entries)
    """
    canonical_urls = {_normalize_url(url) for url, _ in sources_list}
    n_canonical = len(sources_list)

    body, llm_entries = extract_references_section(report_text)

    # Inline citation health (body only — the references section's own
    # numbers are not inline citations)
    inline = [int(n) for n in _INLINE_REF.findall(body)]
    invalid_inline = sorted({n for n in inline if n < 1 or n > n_canonical})

    # How many of the LLM's own reference entries match a real source
    unmatched = 0
    for _, entry_text in llm_entries:
        urls = _URL.findall(entry_text)
        if urls:
            if not any(_normalize_url(u) in canonical_urls for u in urls):
                unmatched += 1
        else:
            # No URL at all — can't be verified against the source list
            unmatched += 1

    # Deterministic repair: drop the LLM's section, append the canonical
    # one. Idempotent — the canonical section is matched by the same
    # heading pattern, so a second pass is a no-op.
    repaired = body.rstrip()
    if sources_list:
        repaired += canonical_references_text(sources_list)
        repaired += "\n"

    stats = {
        "inline_refs": len(inline),
        "invalid_inline": invalid_inline,
        "llm_ref_entries": len(llm_entries),
        "unmatched_refs": unmatched,
    }
    return repaired, stats


def citation_stats_line(stats: dict) -> str:
    """One-line Research Stats summary of citation health."""
    bits = [f"{stats['inline_refs']} inline refs"]
    if stats["invalid_inline"]:
        bits.append(
            f"{len(stats['invalid_inline'])} ref number(s) outside the "
            f"reference list: {stats['invalid_inline'][:5]}"
        )
    if stats["llm_ref_entries"]:
        bits.append(
            f"reference list rebuilt ({stats['unmatched_refs']}/"
            f"{stats['llm_ref_entries']} LLM entries did not match a known source)"
        )
    return "  Citations: " + "; ".join(bits) + "\n"
