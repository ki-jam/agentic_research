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


# Domains exempt from the per-domain cap. The cap exists to stop one
# popular site (Wikipedia, a vendor blog, ...) from eating the whole
# extraction budget and dominating the report. On these domains every
# URL is a distinct primary source — a separate repo, a separate paper —
# so capping would drop evidence instead of noise. Suffix match, so
# dl.acm.org counts as acm.org and link.springer.com as springer.com.
UNCAPPED_DOMAINS: frozenset[str] = frozenset(
    {
        "github.com",
        "arxiv.org",
        # academic publishers
        "nature.com",
        "science.org",
        "sciencedirect.com",
        "springer.com",
        "acm.org",
        "ieee.org",
        "pnas.org",
        "plos.org",
        "mdpi.com",
        "cell.com",
        "frontiersin.org",
        "tandfonline.com",
        "wiley.com",
        "oup.com",
        "cambridge.org",
        "jamanetwork.com",
        "royalsocietypublishing.org",
        # conference proceedings
        "aclanthology.org",
        "openreview.net",
        "neurips.cc",
        "icml.cc",
        "iclr.cc",
        "mlr.press",
    }
)


def _is_uncapped_domain(domain: str, uncapped: frozenset[str]) -> bool:
    """True when the domain (or its registrable parent) is exempt from the cap."""
    return domain in uncapped or any(domain.endswith("." + d) for d in uncapped)


def _enforce_domain_cap(
    results: list[dict],
    max_per_domain: int = 3,
    uncapped: frozenset[str] = UNCAPPED_DOMAINS,
) -> list[dict]:
    """Limit how many results come from any single domain.

    Prevents popular sites (Wikipedia, vendor blogs, ...) from dominating
    the report. Inspired by deep-researcher-node's per-domain caps.

    Args:
        results: list of search result dicts with 'url' keys
        max_per_domain: maximum results per domain (default: 3)
        uncapped: domains that are never capped (default: UNCAPPED_DOMAINS).
            Every URL there is a distinct primary source (a separate repo
            or paper), so the cap would drop evidence instead of noise.

    Returns:
        Filtered list with at most max_per_domain results per capped domain
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
        if _is_uncapped_domain(domain, uncapped):
            kept.append(r)
            continue
        count = domain_counts.get(domain, 0)
        if count < max_per_domain:
            kept.append(r)
            domain_counts[domain] = count + 1
    return kept


def _kb_hits_text(kb_hits: list[dict]) -> str:
    """Format knowledge-base hits into a prefix block (empty string if none).

    Shared by the cache-hit and fresh-search paths so both prepend the same
    "[KNOWLEDGE BASE HITS]" context for extraction.
    """
    if not kb_hits:
        return ""
    return "\n\n[KNOWLEDGE BASE HITS]\n" + "\n\n---\n\n".join(
        f"Title: {h.get('title', '')}\nSummary: {h.get('summary', '')}\nTags: {h.get('topic_tags', '')}"
        for h in kb_hits
    )


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

    # ── 1. Routing decision — web / arXiv / both (+ GitHub leg) ───
    # Tiered, word-boundary-aware science detection plus the OSS
    # vocabulary for the GitHub repositories leg (see
    # utils/search_routing.py). Honors the SEARCH_FORCE_ARXIV and
    # SEARCH_FORCE_GITHUB overrides.
    from utils.search_routing import github_search_query, plan_search

    force = (search_config.force_arxiv or "auto").strip().lower()
    if force not in ("auto", "web", "arxiv", "both"):
        force = "auto"
    plan = plan_search(query, force=force, force_github=search_config.force_github)
    search_arxiv = plan.use_arxiv
    search_github = plan.use_github

    # GitHub's search API ANDs every keyword, so the leg searches a
    # compact phrase derived from the question (see
    # utils.search_routing.github_search_query), never the full question.
    github_q = github_search_query(query) if search_github else ""
    if search_github and not github_q:
        search_github = False  # nothing salient to search for

    # force="arxiv" promotes arXiv to the primary engine; otherwise the
    # configured primary drives the web leg.
    primary = "arxiv" if force == "arxiv" else search_config.primary

    routing_note = plan.reason
    if plan.use_github:
        routing_note = f"{routing_note}; github: {plan.github_reason} q={github_q!r}"
    print(
        f"  [ROUTING] {query[:60]} → web={plan.use_web} arxiv={plan.use_arxiv} "
        f"github={search_github} ({routing_note})"
    )

    # ── 2. Check SQLite cache ───────────────────────────────────
    # Include the per-engine result caps so a change to them invalidates
    # stale merges.
    cache_key = (
        f"{query}||{primary}||{search_arxiv}||{search_config.arxiv_max_results}"
        f"||{search_github}||{search_config.github_max_results}"
    )
    if search_github:
        # The phrase determines what the GitHub leg returns, so it is part
        # of the merge's identity. (Also invalidates entries cached before
        # phrase derivation existed — those merges have no repo grounding.)
        cache_key += f"||{github_q}"
    cached = check_cache(cache_key)
    if cached is not None:
        content = _format_results(cached)
        title = cached[0].get("title", query)
        url = cached[0].get("url", "")
        engine_label = f"{primary} (cached)"
        if search_arxiv and primary != "arxiv":
            engine_label += "+arxiv"
        if search_github:
            engine_label += "+github"
        print(
            f"  [CACHE HIT] {query[:60]} → {len(cached)} results cached from {engine_label}"
        )
        # Prepend KB hits so the cached path matches the fresh path.
        kb_prefix = _kb_hits_text(kb_hits)
        if kb_prefix:
            content = kb_prefix + "\n\n---\n\n" + content
            print(
                f"  [KB HIT] {len(kb_hits)} knowledge entry(s) from prior iterations: {[h['title'][:50] for h in kb_hits]}"
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

    # ── 3. Execute the web leg (primary engine) ──────────────────
    results = None
    arxiv_fetched = False  # True once arXiv has produced `results`
    results_source = None  # "web" or "arxiv" — where `results` came from
    try:
        if plan.use_web:
            if primary == "exa":
                results = _search_exa(query)
                results_source = "web"
            elif primary == "brave":
                results = _search_brave(query)
                results_source = "web"
            elif primary == "arxiv":
                results = _search_arxiv(query)
                arxiv_fetched = True
                results_source = "arxiv"
            elif primary == "exa+brave":
                results = _search_exa(query)
                # If Exa results look low-quality, fall back to Brave
                if results and _results_look_low_quality(results):
                    print("  [SEARCH] Exa results look low-quality → trying Brave")
                    brave_results = _search_brave(query)
                    if brave_results and len(brave_results) >= len(results):
                        results = brave_results
                results_source = "web"
            else:
                results = _fallback_search(query)
    except Exception as e:
        error = str(e)
        results_source = None
        print(f"  [SEARCH] primary '{primary}' failed: {error}")

    # ── 4. Fallback if the web leg failed / returned nothing ─────
    if plan.use_web and (not results or len(results) == 0):
        # arXiv is the keyless engine of last resort, but only when the
        # routing plan actually wants arXiv. For web-only topics, falling
        # back to papers would be noise (e.g. "what is an F1 visa") — an
        # honest empty row is better than junk results.
        if primary != "brave":
            fallback = "brave"
        elif search_arxiv:
            fallback = "arxiv"
        else:
            fallback = None
        if fallback:
            try:
                if fallback == "brave":
                    results = _search_brave(query)
                    results_source = "web"
                elif fallback == "arxiv":
                    results = _search_arxiv(query)
                    arxiv_fetched = True
                    results_source = "arxiv"
                else:
                    results = _fallback_search(query)
            except Exception:
                # Honest empty (no fake "Error:" row): a genuine no-results
                # state stays honest, and a stored empty row still counts as
                # "an actual search was attempted" for the consolidation gate.
                results = None
                results_source = None

    # ── 5. arXiv leg (science topics) — skip if arXiv already ran ──
    arxiv_results = None
    if search_arxiv and not arxiv_fetched:
        try:
            arxiv_results = _search_arxiv(query)
        except Exception:
            arxiv_results = None

    # ── 5.5. GitHub leg (OSS topics) — supplementary repos, by stars ──
    # A GitHub failure (rate limit, network) degrades to a web-only merge;
    # the search itself still succeeds.
    github_results = None
    if search_github:
        try:
            github_results = _search_github(github_q)
            if not github_results and " " in github_q:
                # GitHub ANDs every keyword, so a two-term phrase can still
                # match nothing. Retry with the leading term alone.
                github_results = _search_github(github_q.split()[0])
            if github_results:
                print(f"  [GITHUB] {len(github_results)} repos for: {github_q!r}")
        except Exception as e:
            print(f"  [GITHUB] search failed: {e}")
            github_results = None

    # ── 6. Merge results (primary + arXiv + GitHub if applicable) ──
    all_results = list(results or [])
    for leg in (arxiv_results, github_results):
        if not leg:
            continue
        # Avoid exact duplicate URLs
        existing_urls = {r.get("url", "") for r in all_results if r.get("url")}
        for r in leg:
            if r.get("url") not in existing_urls:
                all_results.append(r)

    # ── 7. Browser enrichment — fill in content where missing ────
    # For results with short/empty content, fetch the page and extract
    # visible text. This catches docs, blogs, and tutorials that APIs
    # return as links but without meaningful snippet text.
    if search_config.browser_search_enabled:
        all_results = _enrich_with_browser(all_results)

    # ── 8. Enforce per-domain caps (source diversity) ────────
    # Stops one site (Wikipedia, a vendor blog) from eating the whole
    # extraction budget. github.com, arxiv.org and academic publishers
    # are exempt (UNCAPPED_DOMAINS) — every URL there is a distinct
    # primary source. Previously they only got a raised cap (5), shared
    # with the primary engine's own github.com URLs: Exa's repos filled
    # it first and the GitHub leg's repos were silently truncated to
    # whatever slots were left.
    all_results = _enforce_domain_cap(all_results)

    # ── 9. Store in SQLite cache ────────────────────────────────
    # Label reflects what actually produced content: the primary engine,
    # plus "+arxiv" when arXiv results were merged in — either via the
    # science leg, or because arXiv was reached through the fallback
    # chain — and "+github" when repo results were merged in.
    if primary == "arxiv":
        engine_label = "arxiv"
    elif arxiv_results or results_source == "arxiv":
        engine_label = f"{primary}+arxiv"
    else:
        engine_label = f"{primary}"
    if github_results:
        engine_label += "+github"
    store_cache(cache_key, all_results, engine=engine_label)

    # ── 10. Store raw results in DB ──────────────────────────────
    content = _format_results(all_results)

    # Prepend knowledge base hits if any (so extraction sees KB context too)
    kb_prefix = _kb_hits_text(kb_hits)
    if kb_prefix:
        content = kb_prefix + "\n\n---\n\n" + content
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
    """Search via arXiv using the arxiv Python package (typed, rate-limited).

    Falls back to raw XML parsing if the arxiv package is not installed.
    """
    from nodes.arxiv_search import search_arxiv as _arxiv_search

    return _arxiv_search(query, max_results=search_config.arxiv_max_results)


def _search_github(query: str) -> list[dict]:
    """Search GitHub repositories via the REST search API (no key needed).

    Grounds OSS/library questions in real repos: owner/name, description,
    stars, language, topics. Sorted by stars so established projects rank
    first (the extraction prompt already weights established tools over
    niche ones). Unauthenticated: 10 search req/min; with GITHUB_TOKEN:
    30 req/min. Same urllib style as _search_brave.
    """
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": str(search_config.github_max_results),
    }
    url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode(params)
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "research-agent",
    }
    if search_config.github_token:
        headers["Authorization"] = f"Bearer {search_config.github_token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    out = []
    for r in data.get("items", []):
        owner = (r.get("owner") or {}).get("login", "")
        name = r.get("name", "")
        meta = []
        if r.get("stargazers_count") is not None:
            meta.append(f"{r['stargazers_count']} stars")
        if r.get("language"):
            meta.append(r["language"])
        if r.get("topics"):
            meta.append("topics: " + ", ".join(r["topics"][:5]))
        content = r.get("description") or ""
        if meta:
            content = f"{content} | {' | '.join(meta)}" if content else " | ".join(meta)
        out.append(
            {
                "title": f"{owner}/{name}" if owner else name,
                "url": r.get("html_url", ""),
                "content": content,
            }
        )
    return out


def _enrich_with_browser(results: list[dict]) -> list[dict]:
    """Enrich results with browser-extracted content where missing.

    Wraps browser_search.enrich_with_browser in a try/except so browser
    failures never break the search pipeline.
    """
    try:
        from nodes.browser_search import enrich_with_browser as _enrich

        enriched = _enrich(results, max_pages=3)
        enriched_count = sum(1 for r in enriched if r.get("content_source"))
        if enriched_count:
            print(f"  [BROWSER] Enriched {enriched_count} result(s) with page content")
        return enriched
    except ImportError:
        print("  [BROWSER] playwright not installed — skipping enrichment")
        return results
    except Exception:
        return results  # non-critical: don't break search on scrape failure


def _fallback_search(query: str) -> list[dict]:
    """Unknown search primary → no results.

    Returning [] (instead of a placeholder row) matters: a fake result
    would be stored in raw_results and presented to extraction as if it
    were real data. With [], the caller's fallback chain (Brave/arXiv)
    engages, and a genuine no-results state stays honest.
    """
    return []


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

    # Signal 1: Most results are GitHub repos WITH short content. Exa
    # returns README text for OSS repos (the core use case), so a GitHub
    # result with real content is NOT a low-quality signal.
    github_thin = sum(
        1
        for r in results
        if "github.com" in r.get("url", "").lower()
        and (not r.get("content") or len(r.get("content", "")) < 50)
    )
    if github_thin > n * 0.6:
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
