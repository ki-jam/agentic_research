# Research Agent Workflow

A cyclical research agent built with LangGraph that plans, searches, extracts, and curates knowledge with human-in-the-loop review and resumable sessions.

## Quick Start

### Using uv (recommended)

```bash
# Install uv if you don't have it: https://docs.astral.sh/uv/
cd research_agent

uv sync          # installs dependencies + dev tools
uv run pytest    # run tests
uv run ruff .    # lint & format
```

### Using pip

```bash
cd research_agent

# Activate the venv
source ../venv/bin/activate  # or: ./venv/bin/activate from parent dir

# Run with a query
python run.py "The future of micro-frontends in 2026"

# Switch LLM provider / model at runtime
python run.py "your query" --provider openai
python run.py "your query" --provider anthropic --model claude-sonnet-4-20250514

# Change the loop budget
python run.py "your query" --max-iterations 15

# Resume an interrupted session (by plan ID or session/thread ID)
python run.py --resume plan_abc123def4

# Continue an existing plan (creates a new thread, no new plan row)
python run.py --plan-id plan_abc123def4 "your query"

# Pre-flight: the agent warns if similar active plans already exist
python run.py "similar query"
```

API keys and workflow settings come from `.env` or environment variables — nothing is hardcoded in the code.

## Architecture

```
START → make_plan → manage_backlog → search_execution → extract_and_verify → consolidation
                 ↑                                              |
                 └──────────── loop (next question) ───────────┤
                                                               ↓
                                   prepare_review (analysis) → human_review (interrupt)
                                                               ↓
                                                         curate_knowledge
                                                               ↓ (back to consolidation, loop)
                                                               ↓
consolidation → write_findings → END
              → write_findings_no_evidence → END
               (when the definition of done is met, the curator types "stop" at a review,
                or the quality gate finds no reliable findings)
```

Each research iteration picks the highest-priority open question, searches for it (cache-aware), extracts structured findings, and consolidates progress. The loop exits when:

- the LLM readiness check says the research is **ready** (all questions resolved, enough findings), or
- **max iterations** is reached (interim analysis screen), or
- the curator types **`stop`** at a periodic or interim review.

Quality gaps from the readiness check are promoted back into the question bank, so the loop can actually address them instead of spinning. Gaps are added both during backlog routing (when the loop continues) and at max iterations (before the interim analysis screen).

## Key Features

- **Configurable LLM** — local (llama.cpp/Ollama), OpenAI, or Anthropic; per-stage model overrides (`PLANNING_MODEL`, `EXTRACTION_MODEL`, `CURATION_MODEL`, `REPORT_MODEL`, `SUMMARIZE_MODEL`)
- **Genuine human-in-the-loop** — `interrupt()` suspends the graph; the review analysis is computed once in a separate node before the interrupt and checkpointed, so resuming re-displays the screen without re-running LLM calls
- **Resumable sessions** — SQLite checkpointer (`data/checkpoints.db`); Ctrl-C, crashes, or reboots can be continued with `--resume`
- **Search caching** — results cached by `query||engine||arxiv_flag` key; no redundant API calls
- **Multi-engine search** — Exa (primary; switch via `SEARCH_PRIMARY`), ArXiv (auto-triggered for science terms), citation discovery (Exa answer API, toggleable)
- **SQLite knowledge base** — plans, questions, raw results, findings, curated entries, and reports, all plan-scoped
- **Quality filter** — shared `utils/quality.py` module; generic filler findings are discarded and tracked
- **Failure recovery** — extraction errors mark questions `failed` (non-terminal); `retry <id>` resets them for re-processing, or `accept <id>` resolves them with a placeholder finding that records the gap
- **Evidence Matrix** — per-source coverage stats shown at reviews to surface under-represented sources
- **Pre-flight plan matching** — before creating a new plan, the agent checks for similar active plans (Jaccard keyword similarity, threshold 0.25) and warns the user to avoid duplication. Use `--plan-id` to continue an existing plan instead.
- **DoD gap self-correction** — when the readiness check identifies coverage gaps (e.g. "no TensorFlow coverage"), they are automatically promoted to questions during backlog routing and at max iterations, so the loop can actually investigate the missing angles.

## Review Commands

Every `REVIEW_EVERY` iterations (and at max iterations) the graph suspends and prints a review screen. Type:

| Command | Effect |
|---|---|
| `continue` | Skip curation, continue research |
| `curate <id>` | Promote a raw entry to the curated library |
| `discard <id>` | Remove an entry |
| `tag <id> <t>` | Add topic tags to an entry |
| `findings` | Show all extracted findings (including discarded) |
| `contradictions` | Show all flagged contradiction pairs |
| `retry <id>` | Reset a `failed` question to `pending` (re-process it) |
| `accept <id>` | Resolve a `failed` question with a placeholder finding (sources exist, extraction failed) |
| `rewind` | Force one more backlog-pass cycle before the next review |
| `gap <n>` | Add the n-th gap from the interim analysis as a research question |
| `stats` | Full database stats |
| `stop` | End research and write the final report |

## Sessions & Resume

Every graph step is checkpointed to `data/checkpoints.db`. If a run is interrupted (Ctrl-C, crash, machine sleep):

```bash
python run.py --resume plan_abc123def4      # by plan ID (shown after make_plan)
python run.py --resume <thread-uuid>        # or by the session ID from the banner
```

A session suspended at a review re-displays the review screen from the checkpoint; a session that died mid-step continues from its last checkpoint.

## Configuration

Environment variables (or `.env` in this directory):

| Variable | Default | Description |
|---|---|---|
| `PROVIDER` | `local` | `local`, `openai`, or `anthropic` |
| `LOCAL_MODEL` | `qwen36_35` | Local model name |
| `LOCAL_BASE_URL` | `http://localhost:8080/v1` | Local API endpoint |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | — / `gpt-4o` | OpenAI credentials |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | — / `claude-sonnet-4-20250514` | Anthropic credentials |
| `EXA_API_KEY` | — | Exa search API key |
| `BRAVE_API_KEY` | — | Brave Search API key |
| `SEARCH_PRIMARY` | `exa` | Primary search engine (`exa`, `brave`, or `arxiv`) |
| `ARXIV_MAX_RESULTS` | `10` | Max results from the arXiv engine |
| `ENABLE_CITATION_DISCOVERY` | `true` | Use Exa answer API for foundational papers |
| `MAX_ITERATIONS` | `10` | Loop budget before interim analysis |
| `REVIEW_EVERY` | `5` | Pause for human review every N iterations |
| `MIN_FINDINGS` | `3` | Minimum valid findings required for the ready check |
| `REFINE_QUERY_THRESHOLD` | `20` | Queries shorter than this (chars) skip LLM refinement |
| `PLANNING_MODEL` / `EXTRACTION_MODEL` / `CURATION_MODEL` / `REPORT_MODEL` / `SUMMARIZE_MODEL` | *(provider default)* | Per-stage model overrides |
| `RESEARCH_BASE_DIR` | `research_agent/data` | Data directory (DB, checkpoints, reports) |

## Database

SQLite at `data/knowledge.db` (7 tables):

| Table | Contents | Notes |
|---|---|---|
| `plans` | One row per research session | `thread_id` links the plan to its LangGraph checkpoint thread (enables `--resume`) |
| `questions` | The question bank | `status`: `pending` → `in_progress` → `resolved` / `failed` / `discarded`; `last_error` records why a question failed |
| `search_cache` | Cached search results | Keyed by `query_hash` of `query||engine||arxiv_flag` |
| `raw_results` | Raw search hits per question | `plan_id` + `engine` (`exa`, `arxiv`, `exa-answer` for citation discovery) |
| `knowledge_entries` | Curated library | `status`: `raw` → `curated` / `discarded`; title, topic tags, summary, full text |
| `curated_findings` | Structured findings per question | Facts with confidence, summary, tags |
| `reports` | Final reports | Also written to a timestamped `.md` file in `data/` |

Pre-existing databases are migrated in place on startup (`ALTER TABLE` guards add `plans.thread_id` and `questions.last_error`).

## File Structure

```
research_agent/
├── config/
│   ├── settings.py      # LLM, search, paths, workflow settings (env-wired)
│   └── prompts.py       # LLM prompt templates
├── state/research_state.py  # LangGraph state definition
├── db/
│   ├── engine.py           # SQLite connection, schema, migrations
│   └── knowledge.py        # CRUD operations (plan-scoped)
├── cache/search_cache.py   # Search query cache
├── nodes/
│   ├── planning.py         # Query refinement + plan creation
│   ├── backlog.py          # Question selection + re-prioritization
│   ├── search.py           # Cache-aware multi-engine search
│   ├── extraction.py       # Fact extraction + follow-up questions
│   ├── consolidation.py    # Progress assessment + DoD check + routing
│   └── curation.py         # Review analysis + interrupt + curation commands
├── utils/
│   ├── cost.py             # LLM cost tracking (per-app tracker + active slot)
│   ├── llm.py              # Shared JSON extraction/parsing for LLM output
│   └── quality.py          # Finding quality filter + science-topic detection
├── graph/workflow.py       # LangGraph StateGraph assembly + build_app()
├── tests/test_nodes.py     # Test suite (scripted LLMs, no network)
├── run.py                  # CLI execution script
└── data/                   # knowledge.db, checkpoints.db, report .md files
```

## Tests

```bash
cd research_agent && ../venv/bin/python -m pytest tests/ -q
```

The suite is hermetic: LLM calls use scripted mocks dispatched on prompt content, search calls are patched, and all data lands in a throwaway test DB. It covers the DB layer, every node, the interrupt/resume cycle end-to-end (two full graph runs), and regressions (plan scoping, question terminal states, search enforcement).
