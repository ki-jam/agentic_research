# Search Modules

## Overview

The research agent supports multiple search backends, a browser-based content enrichment system, and typed arXiv paper search.

```
Query ──→ Exa / Brave / arXiv ──→ Merge ──→ Browser Enrich* ──→ Cache ──→ DB ──→ Extraction
            (+ GitHub repos leg,
             + citation discovery)
                                                                  ↑
                                                          (opt-in, BROWSER_SEARCH_ENABLED=true)
```

## Search Engines

### Exa (`exa`)

AI-native search with full-page text extraction.

```python
from nodes.search import _search_exa
results = _search_exa("LLM serving frameworks")
```

**Pros:** Full text content per result, AI-generated highlights  
**Cons:** Requires API key, costs money

### Brave (`brave`)

Standard web search via Brave's API.

```python
from nodes.search import _search_brave
results = _search_brave("LLM serving frameworks")
```

**Pros:** Free tier, broad coverage  
**Cons:** Snippet-level content only (no full text)

### arXiv (`arxiv`)

Typed academic paper search via the `arxiv` Python package.

```python
from nodes.arxiv_search import search_arxiv, get_paper_by_id, search_by_category

# Full-text search
results = search_arxiv("transformer attention mechanism", max_results=10,
                       sort_by="submittedDate", sort_order="descending")

# Fetch by ID
paper = get_paper_by_id("1706.03762")  # Attention Is All You Need

# Category search
papers = search_by_category("cs.AI", max_results=10)
```

**Pros:** No API key, typed results, rate-limited, full query syntax  
**Cons:** arXiv only, academic papers only

### Combined (`exa+brave`)

Runs Exa first, falls back to Brave if Exa results look low-quality.

```bash
export SEARCH_PRIMARY=exa+brave
```

### GitHub repositories leg

For OSS-flavored questions, a supplementary leg searches GitHub repositories
sorted by stars (`SEARCH_FORCE_GITHUB=auto` is the default; `on` always runs
it, `off` never does). It triggers on open-source vocabulary (`OSS_TERMS` in
`utils/search_routing.py`: `open source`, `opensource`, `oss`, `github`,
`library`, `framework`, `sdk`, `tool`, …) and logs the decision:

```
[ROUTING] best OSS tool for X → web=True arxiv=False github=True (no science signal — web only; github: OSS vocabulary: tool q='oss tool')
[GITHUB] 5 repos for: 'oss tool'
```

Notes:

- The GitHub search API **ANDs every keyword**, so the leg searches a compact
  1–2 token phrase derived from the question (`github_search_query()` in
  `utils/search_routing.py`) — never the full question.
- `GITHUB_MAX_RESULTS` (default 5) caps the leg.
- Keyless rate limit is 10 search req/min; a `GITHUB_TOKEN` raises it to 30.
- `github.com` is in `UNCAPPED_DOMAINS`, so every repo result survives the
  per-domain source-diversity cap.

## Search Routing (web / arXiv)

Each question is routed to **web only**, **arXiv only**, or **both** before any
engine is called. The decision lives in `utils/search_routing.py`
(`plan_search(query, force)`), which returns an explainable `SearchPlan`
(`use_web`, `use_arxiv`, `matched` terms, `reason`). The node logs it:

```
[ROUTING] what is the state of machine learning research → web=True arxiv=True (strong science term(s): machine learning)
```

### How it decides (auto mode)

Instead of the old flat substring match against a term list (where
`"suppose"` contained `"ppo"` and `"machine learning"` wasn't in the list at
all), routing uses a three-tier science vocabulary matched at **word
boundaries** (case-insensitive, plurals and version suffixes like `gpt-4o`
allowed; phrases match across spaces/hyphens):

| Rule | Example | Result |
|------|---------|--------|
| Any **STRONG** term | "machine learning", "transformer", "protein folding", "robotics" | web + arXiv |
| ≥ 2 **MEDIUM** terms | "CNN report on benchmark results" | web + arXiv |
| 1 MEDIUM + ≥ 1 **WEAK** term | "loss optimization" | web + arXiv |
| ≥ 3 **WEAK** terms | "training loss and epoch curves" | web + arXiv |
| Explicit "arxiv" mention | "find papers on arxiv about …" | web + arXiv |
| Otherwise | "what is an F1 visa", "suppose", "CNN report" | web only |

- **STRONG** — unambiguous ML/science terms (one hit is enough).
- **MEDIUM** — scientific but also everyday ("cnn", "benchmark", "regression", "gpu").
- **WEAK** — very common words with a technical meaning ("loss", "policy", "epoch", "f1"); only count in groups.

Word-boundary matching means `ppo` no longer matches inside *suppose* or
*property*, and `f1` no longer matches *F1 visa* on its own.

### Overriding with `SEARCH_FORCE_ARXIV`

| Value | Behavior |
|-------|----------|
| `auto` (default) | Vocabulary-based routing above |
| `web` | Never query arXiv, even on science topics |
| `arxiv` | arXiv becomes the primary engine (no web leg) |
| `both` | Always run web + arXiv, even on everyday topics |

```bash
export SEARCH_FORCE_ARXIV=auto   # or: web | arxiv | both
```

Unknown values fall back to `auto`. `is_science_topic()` (used by the
consolidation gate) delegates to the same `plan_search`, so it stays
consistent with the node.

### Pipeline effect of a routing decision

- **web + arXiv**: the web leg runs first; the arXiv leg merges in papers
  (deduped by URL). The arXiv domain cap is raised to
  `max(3, min(ARXIV_MAX_RESULTS, 6))` so papers survive the per-domain cap.
- **arXiv only** (`force=arxiv` or `SEARCH_PRIMARY=arxiv`): arXiv runs exactly
  once (not as primary *and* as the arXiv leg).
- **web only**: no arXiv call at all — including as a fallback. If the
  primary web engine fails on a web-only topic, an honest empty row is
  stored (papers would be noise for everyday topics).
- When the primary web engine fails on a web **+ arXiv** topic, the fallback
  chain reaches arXiv (it is the keyless engine of last resort) and the
  engine label reflects that (e.g. `brave+arxiv`).
- The cache key is `{query}||{primary}||{search_arxiv}||{arxiv_max_results}||{search_github}||{github_max_results}` (plus the GitHub search phrase when that leg runs),
  so any routing-relevant config change invalidates stale merges.
  Cache hits log `[CACHE HIT] <query> → N results cached from <engine>`.

### arXiv query rewriting

`rewrite_for_arxiv()` strips conversational stopwords ("what is the state of
…") before calling arXiv, because arXiv ANDs every term together. It passes
queries through unchanged when they use field prefixes (`ti:`, `au:`, `abs:`,
`cat:`, `all:`) or uppercase `AND`/`OR`/`ANDNOT`, and never returns an empty
string.

## Citation Discovery

When `ENABLE_CITATION_DISCOVERY=true` (default), the search node first asks
Exa's answer API for the **foundational papers** behind the question and
stores them as `exa-answer` raw results, so extraction can ground itself in
primary sources. It is plan-scoped: once the plan has citation results, later
questions log a skip (`[CITATION DISCOVERY] Skipped (this plan already has
citation results)`); if the API failed and nothing was stored, the next
question retries. Set `ENABLE_CITATION_DISCOVERY=false` to turn it off (it
costs one Exa answer API call per plan).

## Browser Enrichment

Fills in full page content for results that APIs return as links without meaningful text.

### How it works

1. **Static fetch** — `urllib` fetches HTML, custom parser extracts visible text
2. **Playwright fallback** — headless Chromium for JS-rendered pages

Only triggers on results with `< 200 chars` of content, max 3 pages per query.

### Enable

```bash
export BROWSER_SEARCH_ENABLED=true
```

### Usage

```python
from nodes.browser_search import enrich_with_browser, scrape_url

# Enrich existing results
results = _search_brave("some query")
results = enrich_with_browser(results, max_pages=3)
# → results with short/empty content now have full page text

# Scrape a single URL
result = scrape_url("https://example.com/docs", use_playwright=True)
# → {"title": "...", "url": "...", "content": "..."}
```

### Query Syntax (arXiv)

| Prefix | Field | Example |
|--------|-------|---------|
| `ti:` | Title | `ti:attention is all you need` |
| `au:` | Author | `au:vaswani` |
| `abs:` | Abstract | `abs:reinforcement learning` |
| `cat:` | Category | `cat:cs.AI` |
| `all:` | All fields | `all:graph neural networks` |

**Boolean operators:** `AND`, `OR`, `ANDNOT`

```python
search_arxiv("ti:transformer AND au:vaswani")
search_arxiv("abs:reinforcement OR abs:policy gradient")
```

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `SEARCH_PRIMARY` | `exa` | Primary search engine: `exa`, `brave`, `arxiv`, or `exa+brave` |
| `SEARCH_FORCE_ARXIV` | `auto` | Routing override: `auto` / `web` / `arxiv` / `both` |
| `SEARCH_FORCE_GITHUB` | `auto` | GitHub leg override: `auto` (OSS vocabulary decides) / `on` / `off` |
| `EXA_API_KEY` | — | Exa API key (search + citation discovery) |
| `BRAVE_API_KEY` | — | Brave API key |
| `ARXIV_MAX_RESULTS` | `10` | Max arXiv results per query |
| `GITHUB_MAX_RESULTS` | `5` | Max repos from the GitHub leg |
| `GITHUB_TOKEN` | — | GitHub token (30 search req/min; keyless = 10) |
| `ENABLE_CITATION_DISCOVERY` | `true` | Exa answer API for foundational papers (one call per plan) |
| `SEARCH_CACHE_TTL_DAYS` | `30` | Cache lifetime in days (`0` = never expires) |
| `BROWSER_SEARCH_ENABLED` | `false` | Enable browser content enrichment |

## Adding a New Search Engine

1. Implement `_search_<name>(query: str) -> list[dict]` in `nodes/search.py`
2. Each result dict must have: `title`, `url`, `content`
3. Add the new engine to the switch in `search_execution_node`
4. (Optional) Add to the `_results_look_low_quality` heuristic

Example:

```python
def _search_serper(query: str) -> list[dict]:
    """Search via SerpAPI/Google."""
    # ... implementation ...
    return [{"title": "...", "url": "...", "content": "..."}]

# In search_execution_node:
elif search_config.primary == "serper":
    results = _search_serper(query)
```
