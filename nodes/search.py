"""
Search execution node — cache-aware, multi-engine search.

Checks SQLite cache first. If miss, calls the search API, stores
results in both the cache and raw_results table.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request

from config.settings import search_config
from cache.search_cache import check_cache, store_cache
from db.engine import get_db
from db.knowledge import get_in_progress_question, store_raw_result, search_knowledge


def _enforce_domain_cap(results: list[dict], max_per_domain: int = 3) -> list[dict]:
    """Limit how many results come from any single domain.

    Prevents popular sites (arXiv, Wikipedia, etc.) from dominating the
    report. Inspired by deep-researcher-node's per-domain caps.

    Args:
        results: list of search result dicts with 'url' keys
        max_per_domain: maximum results per domain (default: 3)

    Returns:
        Filtered list with at most max_per_domain results per domain
    """
    domain_counts: dict[str, int] = {}
    kept: list[dict] = []
    for r in results:
        url = r.get("url", "")
        if url:
            # Extract domain from URL (e.g., "arxiv.org" from "https://arxiv.org/abs/...")
            domain = urllib.parse.urlparse(url).netloc
            # Remove leading www. for normalization
            domain = domain.removeprefix("www.")
        else:
            domain = "(unknown)"
        count = domain_counts.get(domain, 0)
        if count < max_per_domain:
            kept.append(r)
            domain_counts[domain] = count + 1
    return kept


def search_execution_node(state) -> dict:
    """Search with cache check, store results in DB."""

    plan_id = state.get("plan_ref")
    if not plan_id:
        return {}

    # Get the in-progress question from DB
    question = get_in_progress_question(plan_id)
    if not question:
        return {}

    query = question["question"]

    # ── 0.3. Knowledge base search — check if we already know this ──
    # Search THIS PLAN'S raw entries from previous iterations
    kb_hits = []
    if state.get("iteration_count", 0) > 1:
        kb_hits = search_knowledge(query, status="raw", limit=3, plan_id=plan_id)

    # ── 0.5. Citation discovery — find foundational papers (first plan question only) ──
    # Plan-scoped via the DB (the old global flag file meant only the first
    # plan in a data dir ever ran this). Self-healing: if the API failed and
    # nothing was stored, the next question retries.
    if search_config.enable_citation_discovery:
        with get_db() as conn:
            has_citations = conn.execute(
                "SELECT 1 FROM raw_results WHERE plan_id = ? AND engine = 'exa-answer' LIMIT 1",
                (plan_id,),
            ).fetchone()
        if not has_citations:
            _discover_foundational_papers(plan_id, query, question["id"])
            print("  [CITATION DISCOVERY] Foundational papers searched")
        else:
            print(
                "  [CITATION DISCOVERY] Skipped (this plan already has citation results)"
            )

    # ── 1. Detect science/math/ML topics → also search arXiv ──────
    from utils.quality import is_science_topic as _is_science

    search_arxiv = _is_science(query)

    # ── 2. Check SQLite cache ───────────────────────────────────
    cache_key = f"{query}||{search_config.primary}||{search_arxiv}"
    cached = check_cache(cache_key)
    if cached is not None:
        content = _format_results(cached)
        title = cached[0].get("title", query)
        url = cached[0].get("url", "")
        engine_label = f"{search_config.primary} (cached)"
        if search_arxiv:
            engine_label += "+arxiv"
        print(
            f"  [CACHE HIT] {query[:60]} → {len(cached)} results cached from {engine_label}"
        )
        store_raw_result(
            question_id=question["id"],
            plan_id=plan_id,
            query=query,
            engine=engine_label,
            content=content,
            title=title,
            source_url=url,
        )
        return {}

    # ── 3. Execute primary search ─────────────────────────────────
    results = None
    error = None
    try:
        if search_config.primary == "exa":
            results = _search_exa(query)
        elif search_config.primary == "brave":
            results = _search_brave(query)
        elif search_config.primary == "arxiv":
            results = _search_arxiv(query)
        elif search_config.primary == "exa+brave":
            results = _search_exa(query)
            # If Exa results look low-quality, fall back to Brave
            if results and _results_look_low_quality(results):
                print("  [SEARCH] Exa results look low-quality → trying Brave")
                brave_results = _search_brave(query)
                if brave_results and len(brave_results) >= len(results):
                    results = brave_results
        else:
            results = _fallback_search(query)
    except Exception as e:
        error = str(e)

    # ── 4. Fallback if primary failed ───────────────────────────
    if not results or len(results) == 0:
        fallback = "brave" if search_config.primary != "brave" else "arxiv"
        try:
            if fallback == "brave":
                results = _search_brave(query)
            elif fallback == "arxiv":
                results = _search_arxiv(query)
            else:
                results = _fallback_search(query)
        except Exception:
            results = [
                {"title": f"Error: {error or 'unknown'}", "content": "", "url": ""}
            ]

    # ── 5. Also search arXiv for science/math/ML topics ─────────
    arxiv_results = None
    if search_arxiv:
        try:
            arxiv_results = _search_arxiv(query)
        except Exception:
            pass

    # ── 6. Merge results (primary + arXiv if applicable) ────────
    all_results = list(results)
    if arxiv_results:
        # Avoid exact duplicate URLs
        existing_urls = {r.get("url", "") for r in all_results if r.get("url")}
        for r in arxiv_results:
            if r.get("url") not in existing_urls:
                all_results.append(r)

    # ── 7. Enforce per-domain caps (source diversity) ────────
    all_results = _enforce_domain_cap(all_results)

    # ── 8. Store in SQLite cache ────────────────────────────────
    engine_label = f"{search_config.primary}"
    if arxiv_results:
        engine_label += "+arxiv"
    store_cache(cache_key, all_results, engine=engine_label)

    # ── 9. Store raw results in DB ──────────────────────────────
    content = _format_results(all_results)

    # Prepend knowledge base hits if any (so extraction sees KB context too)
    if kb_hits:
        kb_text = "\n\n[KNOWLEDGE BASE HITS]\n" + "\n\n---\n\n".join(
            f"Title: {h.get('title', '')}\nSummary: {h.get('summary', '')}\nTags: {h.get('topic_tags', '')}"
            for h in kb_hits
        )
        content = kb_text + "\n\n---\n\n" + content
        print(
            f"  [KB HIT] {len(kb_hits)} knowledge entry(s) from prior iterations: {[h['title'][:50] for h in kb_hits]}"
        )

    store_raw_result(
        question_id=question["id"],
        plan_id=plan_id,
        query=query,
        engine=engine_label,
        content=content,
        title=all_results[0].get("title", query) if all_results else query,
        source_url=all_results[0].get("url", "") if all_results else "",
    )

    return {}


def _format_results(results: list) -> str:
    """Format search results into a single text block."""
    return "\n\n---\n\n".join(
        f"Title: {r.get('title', 'Unknown')}\nURL: {r.get('url', '')}\n{r.get('content', '')}"
        for r in results
    )


# ── Search engine implementations ──────────────────────────────────


def _search_exa(query: str) -> list[dict]:
    """Search via Exa (exa.ai)."""
    from exa_py import Exa

    client = Exa(api_key=search_config.exa_api_key)
    results = client.search(query, num_results=5, contents={"text": True})
    out = []
    for r in results.results:
        url = r.url or r.id
        out.append(
            {
                "title": r.title or url.split("/")[-1] or r.id,
                "url": url,
                "content": r.text or "",
            }
        )
        # Also store metadata that Exa provides
        if hasattr(r, "highlights"):
            out[-1]["highlights"] = r.highlights
        if hasattr(r, "summary"):
            out[-1]["summary"] = r.summary
        if hasattr(r, "author"):
            out[-1]["author"] = r.author
    return out


def _search_brave(query: str) -> list[dict]:
    """Search via Brave Search API."""
    if not search_config.brave_api_key:
        return []
    url = (
        f"https://api.search.brave.com/res/v1/web/search?q={urllib.parse.quote(query)}"
    )
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": search_config.brave_api_key,
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("description", ""),
        }
        for r in data.get("web", {}).get("results", [])[:5]
    ]


def _search_arxiv(query: str) -> list[dict]:
    """Search via arXiv API (no key needed)."""
    import xml.etree.ElementTree as ET

    url = (
        f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}"
        f"&max_results={search_config.arxiv_max_results}"
    )
    with urllib.request.urlopen(url, timeout=15) as resp:
        xml_data = resp.read()
    root = ET.fromstring(xml_data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)
    results = []
    for entry in entries:
        title = entry.find("atom:title", ns).text.replace("\n", " ")
        summary = entry.find("atom:summary", ns).text.replace("\n", " ")
        link = entry.find("atom:id", ns).text
        authors = [
            a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)
        ]
        results.append(
            {
                "title": title,
                "url": link,
                "content": f"Authors: {', '.join(authors)}\n{summary}",
            }
        )
    return results


def _fallback_search(query: str) -> list[dict]:
    """Fallback: return a simple placeholder."""
    return [{"title": f"Search result for: {query}", "url": "", "content": ""}]


def _results_look_low_quality(results: list[dict]) -> bool:
    """Heuristic: detect if search results are likely unhelpful for extraction.

    Low-quality signals:
    - Most results come from GitHub repos (not narrative content)
    - Most results have empty or very short content
    - Few or no results have descriptive content text

    Returns True if results should be replaced with Brave results.
    """
    if not results:
        return True
    if len(results) < 2:
        # Too few results to judge; let the normal fallback handle it
        return False

    n = len(results)

    # Signal 1: Most results are GitHub repos
    github_count = sum(1 for r in results if "github.com" in r.get("url", "").lower())
    if github_count > n * 0.6:
        return True

    # Signal 2: Most results have empty or very short content
    short_content = sum(
        1 for r in results if not r.get("content") or len(r.get("content", "")) < 50
    )
    if short_content > n * 0.6:
        return True

    return False


# ── Citation discovery via Exa answer API ────────────────────────────


def _search_exa_answer(query: str) -> str | None:
    """Call Exa answer API to find foundational/authoritative sources.

    Returns the AI-generated answer text, or None on failure.
    """
    try:
        from exa_py import Exa

        client = Exa(api_key=search_config.exa_api_key)
        response = client.answer(
            f"What are the most important and foundational papers on {query}? "
            f"List the key papers with their titles and arXiv IDs if available. "
            f"Focus on the original source papers and widely-cited works.",
        )
        if hasattr(response, "answer") and response.answer:
            return response.answer
    except Exception:
        pass
    return None


def _parse_papers_from_answer(answer: str) -> list[tuple[str, str]]:
    """Extract (title, arxiv_url) pairs from Exa answer text.

    Handles formats like:
      - Title (arXiv:2402.03300): description
      - Title (arXiv:2402.03300) [citation markers]
      - Title (arXiv 2402.03300) description

    Returns list of (title, url) tuples.
    """
    papers = []
    seen_urls = set()

    # Two-step: find arXiv ID first, then extract title before it
    # arXiv reference: (arXiv:DDDD.DDDDD) or [arXiv:DDDD.DDDDD]
    arxiv_ref_pattern = re.compile(
        r"[([{\s]arXiv:\s*(\d{4}\.\d{4,5})(?:v\d+)?\s*[})\]]"
    )

    for line in answer.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        m = arxiv_ref_pattern.search(stripped)
        if m:
            arxiv_id = m.group(1)
            url = f"https://arxiv.org/abs/{arxiv_id}"

            if url not in seen_urls:
                seen_urls.add(url)
                # Extract everything before the arXiv reference as the title
                title = stripped[: m.start()].strip()
                # Clean up: remove leading prose fragments (not trailing!)
                # e.g., "was introduced in the paper DeepSeekMath..." → keep "DeepSeekMath..."
                title = re.sub(
                    r"^.*?\bin the paper \b", "", title, flags=re.IGNORECASE
                ).strip()
                title = re.sub(
                    r"^.*?\b(?:was introduced|introduced|foundational paper)\b\s*[:\s]",
                    "",
                    title,
                    flags=re.IGNORECASE,
                ).strip()
                # Remove trailing citation-style markers like [1][2]
                title = re.sub(r"\s*\[.*?\]\s*$", "", title).strip()
                # Clean up whitespace
                title = re.sub(r"\s+", " ", title).strip()
                # Remove leading/trailing punctuation
                title = title.strip(":,;.- ")

                if title and 3 < len(title) < 200:
                    papers.append((title, url))

    # Fallback: simple arXiv ID extraction
    if not papers:
        arxiv_pattern = re.compile(
            r"(?:arXiv[:\s]*|arxiv\.org/abs/)(\d{4}\.\d{4,5})(?:v\d+)?"
        )
        for line in answer.split("\n"):
            matches = arxiv_pattern.findall(line.strip())
            if matches:
                for arxiv_id in matches:
                    url = f"https://arxiv.org/abs/{arxiv_id}"
                    if url not in seen_urls:
                        seen_urls.add(url)
                        title = arxiv_pattern.sub("", line.strip()).strip()
                        title = re.sub(r"\[\d+(?:,\d+)*\]", "", title).strip()
                        title = title.strip(":,;.- ")
                        if title and 3 < len(title) < 200:
                            papers.append((title, url))

    return papers


def _search_exa_for_papers(titles: list[str], limit: int = 10) -> list[dict]:
    """Search Exa for a list of paper titles. Returns formatted results.

    Uses ThreadPoolExecutor to parallelize individual title searches while
    preserving input order.
    """
    from concurrent.futures import ThreadPoolExecutor

    from exa_py import Exa

    client = Exa(api_key=search_config.exa_api_key)

    def _search_one(title: str) -> list[dict]:
        """Search Exa for a single paper title."""
        try:
            search_results = client.search(
                title, num_results=2, contents={"text": True}
            )
            return [
                {
                    "title": r.title or url.split("/")[-1] or r.id,
                    "url": url,
                    "content": r.text or "",
                }
                for r in search_results.results
                if (url := r.url or r.id)
            ]
        except Exception:
            return []

    # Parallelize: up to 10 titles → 10 sequential HTTP calls → threaded
    all_results: list[list[dict]] = []
    with ThreadPoolExecutor(max_workers=min(limit, 10)) as pool:
        futures = {
            i: pool.submit(_search_one, title) for i, title in enumerate(titles[:limit])
        }
        for i in range(len(futures)):
            all_results.append(futures[i].result())

    results = [r for batch in all_results for r in batch]

    # Deduplicate by URL
    seen = set()
    unique = []
    for r in results:
        if r["url"] not in seen:
            seen.add(r["url"])
            unique.append(r)

    return unique


def _discover_foundational_papers(plan_id: str, query: str, question_id: str) -> None:
    """Run Exa answer API to find foundational papers, then search for them.

    Stores discovered papers as raw_results with engine="exa-answer".
    Prints a summary of discovered papers.
    """
    print(f"  [CITATION DISCOVERY] Finding foundational papers for: {query[:60]}...")

    answer = _search_exa_answer(query)
    if not answer:
        print("  [CITATION DISCOVERY] No answer returned from Exa API")
        return

    papers = _parse_papers_from_answer(answer)
    if not papers:
        print(
            f"  [CITATION DISCOVERY] No papers parsed from answer (answer length: {len(answer)} chars)"
        )
        # Fallback: store the answer as-is for extraction to use
        store_raw_result(
            question_id=question_id,
            plan_id=plan_id,
            query=query,
            engine="exa-answer (raw)",
            content=answer,
            title=f"Foundational papers for: {query[:50]}",
            source_url="",
        )
        print(
            f"  [CITATION DISCOVERY] Stored raw answer for extraction ({len(answer)} chars)"
        )
        return

    # Search Exa for each discovered paper to get full content
    print(f"  [CITATION DISCOVERY] Found {len(papers)} foundational papers:")
    for title, url in papers:
        short_title = title[:80]
        print(f"    • {short_title}")
        print(f"      {url}")

    # Get full content for these papers
    papers_data = _search_exa_for_papers([t for t, _ in papers])

    if papers_data:
        # Store as a single raw result (citations are metadata, not extracted findings)
        content = _format_results(papers_data)
        store_raw_result(
            question_id=question_id,
            plan_id=plan_id,
            query=f"[foundational papers] {query}",
            engine="exa-answer",
            content=content,
            title=f"Foundational papers: {papers[0][0][:60]}",
            source_url=papers[0][1],
        )
        print(f"  [CITATION DISCOVERY] Stored {len(papers_data)} papers via Exa search")
    else:
        # Store the answer directly - extraction will use it
        store_raw_result(
            question_id=question_id,
            plan_id=plan_id,
            query=f"[foundational papers] {query}",
            engine="exa-answer",
            content=answer,
            title=f"Foundational papers for: {query[:50]}",
            source_url="",
        )
        print(
            f"  [CITATION DISCOVERY] Stored raw answer for extraction ({len(answer)} chars)"
        )
