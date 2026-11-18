"""
Tests for search routing (web / arXiv / which) and the search node's
routing-related behaviors.

Run with:
    cd research_agent && ./../venv/bin/python -m pytest tests/test_search_routing.py -v

No live LLM calls and no network access: the routing unit tests are pure,
and the node-level tests monkeypatch ``_search_exa`` / ``_search_brave`` /
``_search_arxiv`` with recording fakes.
"""

from __future__ import annotations

import os
import sys

import pytest

# Ensure we can import from the package root
sys.path.insert(0, os.path.dirname(__file__) + "/..")

# Fake key — tests never call the real APIs (search functions are patched)
os.environ["EXA_API_KEY"] = "test-key"

from db.knowledge import (  # noqa: E402
    create_finding,
    create_knowledge_entry,
    create_plan,
    create_questions,
    get_raw_results_for_question,
    set_question_in_progress,
)
from cache.search_cache import check_cache, store_cache  # noqa: E402
from config.settings import search_config  # noqa: E402
from nodes import search as search_mod  # noqa: E402
from nodes.search import search_execution_node  # noqa: E402
from nodes.arxiv_search import rewrite_for_arxiv  # noqa: E402
from utils.quality import is_science_topic  # noqa: E402
from utils.search_routing import github_search_query, plan_search  # noqa: E402


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def fresh_test_db():
    """Use a test DB so production data is never touched.

    Sets RESEARCH_DB_PATH to knowledge_test.db, drops/recreates it.
    (Same pattern as tests/test_nodes.py.)
    """
    import db.engine as engine_mod

    db_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    test_db = os.path.join(db_dir, "knowledge_test.db")

    os.environ["RESEARCH_DB_PATH"] = test_db
    if os.path.exists(test_db):
        os.remove(test_db)
    engine_mod.init_db()

    yield

    if os.path.exists(test_db):
        os.remove(test_db)
    os.environ.pop("RESEARCH_DB_PATH", None)


@pytest.fixture(autouse=True)
def _pin_search_config():
    """Pin the search config to deterministic values for every test."""
    monkeypatch_search_config()
    yield


def monkeypatch_search_config():
    """Set the search config fields to known defaults (no monkeypatch — the
    node reads the singleton live, so we set and the caller is responsible
    for per-test overrides via direct attribute assignment)."""
    search_config.enable_citation_discovery = False
    search_config.browser_search_enabled = False
    search_config.primary = "exa"
    search_config.force_arxiv = "auto"
    search_config.arxiv_max_results = 5
    search_config.force_github = "auto"
    search_config.github_max_results = 5
    search_config.github_token = ""


# ── Recording fakes for the engine functions ─────────────────────────


def make_fake(payload, calls: list):
    """Build a recording fake for an engine function.

    ``payload`` is either a list of result dicts (returned as a fresh copy on
    each call, since the node mutates results in place) or an Exception
    instance (raised on call). Every call appends the query to ``calls``.
    """

    def fake(query):
        calls.append(query)
        if isinstance(payload, Exception):
            raise payload
        return list(payload)

    return fake


def _run_node(question: str, iteration_count: int = 1) -> list:
    """Create a plan with one in-progress question, run the search node, and
    return the stored raw results for that question."""
    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": question, "priority": "high"}])
    qid = f"q_{plan_id}_0"
    set_question_in_progress(qid)
    search_execution_node({"plan_ref": plan_id, "iteration_count": iteration_count})
    return get_raw_results_for_question(qid)


# ── Routing vocabulary: everyday queries that must stay web-only ─────
#
# These were the false positives of the old flat substring matcher
# ("suppose" ⊇ "ppo", "CNN" ⊇ "cnn", "F1 visa" ⊇ "f1", ...).

EVERYDAY_QUERIES = [
    "what do you suppose the best restaurant in town is",
    "why was the appointment cancelled and who is responsible",
    "what is the financial loss of the company this quarter",
    "explain the company privacy policy and how to opt out",
    "how do I batch cook meals for the week",
    "what is an F1 visa and how long does it take",
    "what did CNN report about the election results",
    "what do film critics say about the new movie",
    "how do I train my puppy to sit",
    "how do cashback rewards work at the store",
    "what is the Unix epoch and why 1970",
    "how can I get more customer attention for my shop",
    "what is the precision of this watch movement",
    "how is property valuation done in the UK",
    "what is the credit classification system used by banks",
    "why did the stock market drop after the disappointing report",
]

# ── Routing vocabulary: science queries that must trigger arXiv ──────
#
# These were the false negatives of the old matcher ("machine learning" was
# not in the list at all, nor quantum computing, computer vision, ...).

SCIENCE_QUERIES = [
    "what is the state of machine learning research",
    "explain quantum computing and its applications",
    "what are the advances in computer vision",
    "overview of natural language processing techniques",
    "what is NLP used for",
    "what are the latest developments in robotics",
    "overview of speech recognition technology",
    "what is protein folding and how is it solved",
]


# ── Routing unit tests (pure) ────────────────────────────────────────


@pytest.mark.parametrize("query", EVERYDAY_QUERIES)
def test_everyday_queries_route_web_only(query):
    plan = plan_search(query)
    assert plan.use_arxiv is False, (
        f"{query!r} should not trigger arXiv ({plan.reason})"
    )
    assert plan.use_web is True


@pytest.mark.parametrize("query", SCIENCE_QUERIES)
def test_science_queries_route_arxiv(query):
    plan = plan_search(query)
    assert plan.use_arxiv is True, f"{query!r} should trigger arXiv ({plan.reason})"
    assert plan.use_web is True


def test_word_boundary_ppo_not_in_common_words():
    # "ppo" must not match inside everyday words...
    for q in (
        "what do you suppose",
        "the appointment",
        "property valuation",
        "disappointing report",
    ):
        assert plan_search(q).use_arxiv is False, q
    # ...but a standalone PPO does route to arXiv
    assert plan_search("PPO training for agents").use_arxiv is True


def test_cnn_standalone_is_medium_not_enough_alone():
    # "CNN" is a medium term; a single medium hit is not enough on its own.
    assert plan_search("what did CNN report").use_arxiv is False
    # ...but corroborated by a second medium term it does route to arXiv
    assert plan_search("CNN report on benchmark results").use_arxiv is True


def test_f1_visa_is_weak_not_enough_alone():
    # "F1" is a weak term; a single weak hit is not enough on its own.
    assert plan_search("what is an F1 visa").use_arxiv is False


def test_acronym_version_suffixes_match():
    # gpt-4o / LLMs / theorems should still match their base terms.
    assert plan_search("gpt-4o capabilities").use_arxiv is True
    assert plan_search("LLMs in production").use_arxiv is True
    assert plan_search("open theorems in complexity").use_arxiv is True


def test_medium_plus_weak_routes_arxiv():
    # one medium + one weak term -> arXiv
    assert plan_search("how is the model loss optimization done").use_arxiv is True
    # two weak terms alone are not enough (covered by the 3-weak test below)
    assert plan_search("how is the model loss and training tuned").use_arxiv is False


def test_three_weak_terms_route_arxiv():
    # three weak terms together -> arXiv
    assert plan_search("training loss and epoch curves").use_arxiv is True
    # ...but two weak terms are not enough
    assert plan_search("training loss curves").use_arxiv is False


def test_explicit_arxiv_mention_routes_arxiv():
    plan = plan_search("find papers on arxiv about transformers")
    assert plan.use_arxiv is True
    assert "arXiv" in plan.reason


def test_plan_is_frozen_and_explainable():
    plan = plan_search("machine learning")
    assert plan.use_arxiv is True
    assert isinstance(plan.matched, tuple)
    assert plan.reason
    with pytest.raises(Exception):
        plan.use_arxiv = False  # frozen dataclass


# ── Force overrides (unit) ───────────────────────────────────────────


def test_force_arxiv_overrides_vocabulary():
    plan = plan_search("what is a zorblatt", force="arxiv")
    assert plan.use_arxiv is True
    assert plan.use_web is False


def test_force_web_disables_arxiv_even_for_science():
    plan = plan_search("machine learning", force="web")
    assert plan.use_arxiv is False
    assert plan.use_web is True


def test_force_both_enables_arxiv_for_everyday():
    plan = plan_search("what is a zorblatt", force="both")
    assert plan.use_arxiv is True
    assert plan.use_web is True


def test_force_unknown_falls_back_to_auto():
    # Unknown force values are treated as "auto".
    assert plan_search("machine learning", force="garbage").use_arxiv is True
    assert plan_search("what is a zorblatt", force="garbage").use_arxiv is False
    assert plan_search("machine learning", force=None).use_arxiv is True


# ── Routing vocabulary: OSS queries that must trigger the GitHub leg ──
#
# The GitHub leg is a supplementary repositories leg for OSS-flavored
# queries (see OSS_TERMS in utils/search_routing.py).

OSS_QUERIES = [
    "what are the top open-source libraries for evaluating agents",
    "compare open source frameworks for LLM inference serving",
    "which github projects implement a policy gradient sdk",
    "best oss tools for agent evaluation",
    "what is the best framework for building chatbots",
    "is there a library for parsing arxiv metadata",
]

NON_OSS_QUERIES = [
    "what is a zorblatt",
    "what is an F1 visa and how long does it take",
    "why did the stock market drop after the disappointing report",
    "how do I train my puppy to sit",
]


@pytest.mark.parametrize("query", OSS_QUERIES)
def test_oss_queries_enable_github_leg(query):
    plan = plan_search(query)
    assert plan.use_github is True, (
        f"{query!r} should trigger the GitHub leg ({plan.github_reason})"
    )


@pytest.mark.parametrize("query", NON_OSS_QUERIES)
def test_non_oss_queries_skip_github_leg(query):
    plan = plan_search(query)
    assert plan.use_github is False, (
        f"{query!r} should not trigger the GitHub leg ({plan.github_reason})"
    )


def test_github_vocabulary_matches_at_word_boundaries():
    # "tool" must not match inside everyday words...
    for q in ("what is a toolbox", "the toolkit is complete"):
        assert plan_search(q).use_github is False, q
    # ...but standalone "tools" does
    assert plan_search("what tools exist for log analysis").use_github is True


def test_force_github_on_enables_leg_for_everyday():
    plan = plan_search("what is a zorblatt", force_github="on")
    assert plan.use_github is True


def test_force_github_off_disables_leg_for_oss():
    plan = plan_search("top open-source libraries", force_github="off")
    assert plan.use_github is False


def test_force_github_unknown_falls_back_to_auto():
    # Unknown force_github values are treated as "auto".
    assert (
        plan_search("top open-source libraries", force_github="garbage").use_github
        is True
    )
    assert plan_search("what is a zorblatt", force_github="garbage").use_github is False


# ── GitHub phrase derivation (unit) ──────────────────────────────────
#
# GitHub's repository search ANDs every keyword, so the full question
# matches zero repositories. The leg searches a compact phrase derived
# from the question instead (see github_search_query in
# utils/search_routing.py).


def test_github_phrase_keeps_leading_named_tools():
    assert (
        github_search_query(
            "Which of the open-source tools Promptfoo and Opik is more usable "
            "for non-ML users setting up agent evaluations without Python or ML expertise?"
        )
        == "promptfoo opik"
    )


def test_github_phrase_drops_generic_oss_vocabulary():
    # "open source" / "library" triggered the leg but do not identify repos.
    assert (
        github_search_query(
            "Which open-source agent evaluation library (DeepEval, Ragas, "
            "Inspect AI, Promptfoo, or Opik) best detects failure modes?"
        )
        == "agent evaluation"
    )


def test_github_phrase_caps_at_two_terms():
    assert (
        github_search_query("top open-source frameworks alpha beta gamma")
        == "alpha beta"
    )


def test_github_phrase_empty_when_nothing_salient_remains():
    assert github_search_query("") == ""
    assert github_search_query("what open source tools are there") == ""


# ── is_science_topic wrapper consistency ─────────────────────────────


def test_is_science_topic_matches_plan_auto():
    for q in EVERYDAY_QUERIES + SCIENCE_QUERIES:
        assert is_science_topic(q) == plan_search(q, force="auto").use_arxiv, q


def test_is_science_topic_honors_force_override():
    saved = search_config.force_arxiv
    try:
        search_config.force_arxiv = "arxiv"
        assert is_science_topic("what is a zorblatt") is True
        search_config.force_arxiv = "web"
        assert is_science_topic("machine learning") is False
        search_config.force_arxiv = "auto"
        assert is_science_topic("machine learning") is True
        assert is_science_topic("what is a zorblatt") is False
    finally:
        search_config.force_arxiv = saved


# ── rewrite_for_arxiv ────────────────────────────────────────────────


def test_rewrite_strips_conversational_stopwords():
    assert (
        rewrite_for_arxiv("what is the state of machine learning research")
        == "machine learning research"
    )
    assert (
        rewrite_for_arxiv("how do I compare vLLM and SGLang for LLM serving")
        == "vLLM SGLang LLM serving"
    )


def test_rewrite_preserves_field_prefixes_and_booleans():
    assert (
        rewrite_for_arxiv("ti:attention is all you need")
        == "ti:attention is all you need"
    )
    assert rewrite_for_arxiv("cat:cs.AI") == "cat:cs.AI"
    assert (
        rewrite_for_arxiv("abs:reinforcement OR abs:policy gradient")
        == "abs:reinforcement OR abs:policy gradient"
    )


def test_rewrite_never_returns_empty():
    # If every token is a stopword, keep the original (don't send an empty query).
    assert rewrite_for_arxiv("what is") == "what is"
    assert rewrite_for_arxiv("") == ""


def test_rewrite_keeps_single_kept_word():
    assert rewrite_for_arxiv("what is transformer") == "transformer"


# ── Node-level: arXiv deduplication (bug B) ──────────────────────────


def test_primary_arxiv_calls_arxiv_once(monkeypatch):
    """When arXiv is the primary engine, it must be called once, not twice
    (the old code ran it as the primary AND again as the arXiv leg)."""
    arxiv_calls = []
    monkeypatch.setattr(search_config, "primary", "arxiv")
    monkeypatch.setattr(
        search_mod,
        "_search_arxiv",
        make_fake(
            [{"title": "P", "url": "https://arxiv.org/abs/1", "content": "x" * 100}],
            arxiv_calls,
        ),
    )

    _run_node("what is a zorblatt")  # non-science, arxiv primary
    assert len(arxiv_calls) == 1, f"arXiv called {len(arxiv_calls)} times, expected 1"


def test_fallback_arxiv_calls_arxiv_once(monkeypatch):
    """When arXiv is reached via the fallback chain (Brave empty), it must be
    called once, not again in the separate arXiv leg."""
    arxiv_calls = []
    brave_calls = []
    monkeypatch.setattr(search_config, "primary", "brave")
    monkeypatch.setattr(search_mod, "_search_brave", make_fake([], brave_calls))
    monkeypatch.setattr(
        search_mod,
        "_search_arxiv",
        make_fake(
            [{"title": "P", "url": "https://arxiv.org/abs/1", "content": "x" * 100}],
            arxiv_calls,
        ),
    )

    results = _run_node(
        "deep learning survey"
    )  # science topic; brave empty -> arxiv fallback
    assert len(brave_calls) >= 1
    assert len(arxiv_calls) == 1, f"arXiv called {len(arxiv_calls)} times, expected 1"
    # The content came from arXiv via the fallback chain, so the label says so
    assert results[0]["engine"] == "brave+arxiv"


def test_web_only_brave_failure_skips_arxiv_fallback(monkeypatch):
    """When the plan says web-only, a Brave failure must NOT fall back to
    arXiv (papers would be noise for everyday topics) — it stores an honest
    empty row labeled with the primary engine instead."""
    brave_calls = []
    arxiv_calls = []
    monkeypatch.setattr(search_config, "primary", "brave")
    monkeypatch.setattr(
        search_mod, "_search_brave", make_fake(RuntimeError("brave down"), brave_calls)
    )
    monkeypatch.setattr(search_mod, "_search_arxiv", make_fake([], arxiv_calls))

    results = _run_node("what is a zorblatt")  # everyday topic, brave primary
    assert len(brave_calls) == 1
    assert arxiv_calls == [], "arXiv must not be used as a fallback for web-only topics"
    assert len(results) == 1
    assert results[0]["engine"] == "brave"
    assert results[0]["content"] == ""


def test_science_topic_merges_web_and_arxiv(monkeypatch):
    """A science topic with a working web engine runs both legs and merges
    them, labeling the engine 'exa+arxiv'."""
    exa_calls = []
    arxiv_calls = []
    monkeypatch.setattr(
        search_mod,
        "_search_exa",
        make_fake(
            [{"title": "W", "url": "https://example.com/a", "content": "x" * 100}],
            exa_calls,
        ),
    )
    monkeypatch.setattr(
        search_mod,
        "_search_arxiv",
        make_fake(
            [{"title": "P", "url": "https://arxiv.org/abs/1", "content": "x" * 100}],
            arxiv_calls,
        ),
    )

    results = _run_node("deep learning survey")
    assert len(exa_calls) == 1
    assert len(arxiv_calls) == 1
    row = results[0]
    assert row["engine"] == "exa+arxiv"
    assert "example.com/a" in row["content"]
    assert "arxiv.org/abs/1" in row["content"]


def test_everyday_topic_skips_arxiv_leg(monkeypatch):
    """A non-science topic must not call the arXiv leg at all."""
    arxiv_calls = []
    exa_calls = []
    monkeypatch.setattr(
        search_mod,
        "_search_exa",
        make_fake(
            [{"title": "W", "url": "https://example.com/a", "content": "x" * 100}],
            exa_calls,
        ),
    )
    monkeypatch.setattr(search_mod, "_search_arxiv", make_fake([], arxiv_calls))

    results = _run_node("what is a zorblatt")
    assert len(exa_calls) == 1
    assert arxiv_calls == []
    assert results[0]["engine"] == "exa"


# ── Node-level: GitHub leg (OSS topics) ──────────────────────────────


def test_oss_topic_merges_github_results(monkeypatch):
    """An OSS-flavored query runs the GitHub leg and merges repo results,
    labeling the engine 'exa+github' (URLs deduped against the web leg)."""
    exa_calls = []
    github_calls = []
    monkeypatch.setattr(
        search_mod,
        "_search_exa",
        make_fake(
            [{"title": "W", "url": "https://example.com/a", "content": "x" * 100}],
            exa_calls,
        ),
    )
    monkeypatch.setattr(search_mod, "_search_arxiv", make_fake([], []))
    monkeypatch.setattr(
        search_mod,
        "_search_github",
        make_fake(
            [
                {
                    "title": "org/repo",
                    "url": "https://github.com/org/repo",
                    "content": "desc | 5000 stars | Python",
                },
                {
                    "title": "org/dup",
                    "url": "https://example.com/a",
                    "content": "dup url",
                },
            ],
            github_calls,
        ),
    )

    results = _run_node("top open-source libraries for agent evaluation")
    assert len(exa_calls) == 1
    assert len(github_calls) == 1
    row = results[0]
    assert row["engine"] == "exa+github"
    content = row["content"]
    assert "github.com/org/repo" in content
    assert content.count("example.com/a") == 1  # deduped by URL


def test_non_oss_topic_skips_github_leg(monkeypatch):
    """A non-OSS topic must not call the GitHub leg at all."""
    github_calls = []
    monkeypatch.setattr(
        search_mod,
        "_search_exa",
        make_fake(
            [{"title": "W", "url": "https://example.com/a", "content": "x" * 100}], []
        ),
    )
    monkeypatch.setattr(search_mod, "_search_arxiv", make_fake([], []))
    monkeypatch.setattr(search_mod, "_search_github", make_fake([], github_calls))

    results = _run_node("what is a zorblatt")
    assert github_calls == []
    assert results[0]["engine"] == "exa"


def test_github_failure_does_not_break_search(monkeypatch):
    """A GitHub API failure (rate limit, network) degrades to a web-only
    merge — the search still succeeds and is labeled without github."""
    github_calls = []
    monkeypatch.setattr(
        search_mod,
        "_search_exa",
        make_fake(
            [{"title": "W", "url": "https://example.com/a", "content": "x" * 100}], []
        ),
    )
    monkeypatch.setattr(search_mod, "_search_arxiv", make_fake([], []))
    monkeypatch.setattr(
        search_mod,
        "_search_github",
        make_fake(RuntimeError("rate limited"), github_calls),
    )

    results = _run_node("top open-source libraries for deepeval")
    assert len(github_calls) == 1
    assert results[0]["engine"] == "exa"


def test_github_leg_skipped_when_no_salient_terms(monkeypatch):
    """When the question carries only generic OSS vocabulary (nothing that
    identifies a repository), the leg is skipped rather than searching a
    generic or empty query."""
    github_calls = []
    monkeypatch.setattr(
        search_mod,
        "_search_exa",
        make_fake(
            [{"title": "W", "url": "https://example.com/a", "content": "x" * 100}], []
        ),
    )
    monkeypatch.setattr(search_mod, "_search_arxiv", make_fake([], []))
    monkeypatch.setattr(search_mod, "_search_github", make_fake([], github_calls))

    results = _run_node("top open-source libraries")
    assert github_calls == []
    assert results[0]["engine"] == "exa"


def test_github_leg_searches_derived_phrase_not_full_question(monkeypatch):
    """GitHub ANDs every keyword, so the full question would match zero
    repos. The leg searches the compact derived phrase instead."""
    github_calls = []
    monkeypatch.setattr(
        search_mod,
        "_search_exa",
        make_fake(
            [{"title": "W", "url": "https://example.com/a", "content": "x" * 100}], []
        ),
    )
    monkeypatch.setattr(search_mod, "_search_arxiv", make_fake([], []))
    monkeypatch.setattr(
        search_mod,
        "_search_github",
        make_fake(
            [
                {
                    "title": "promptfoo/promptfoo",
                    "url": "https://github.com/promptfoo/promptfoo",
                    "content": "27372 stars | TypeScript",
                }
            ],
            github_calls,
        ),
    )

    results = _run_node(
        "Which of the open-source tools Promptfoo and Opik is more usable "
        "for non-ML users setting up agent evaluations without Python or ML expertise?"
    )
    assert github_calls == ["promptfoo opik"]
    assert results[0]["engine"] == "exa+github"


def test_github_phrase_empty_retries_with_first_term(monkeypatch):
    """When the two-term phrase matches nothing (GitHub ANDs keywords),
    the leg retries once with the leading term alone."""
    github_calls = []

    def fake_github(query):
        github_calls.append(query)
        if query == "evalai":
            return [
                {
                    "title": "Cloud-CV/EvalAI",
                    "url": "https://github.com/Cloud-CV/EvalAI",
                    "content": "2049 stars | Python",
                }
            ]
        return []

    monkeypatch.setattr(
        search_mod,
        "_search_exa",
        make_fake(
            [{"title": "W", "url": "https://example.com/a", "content": "x" * 100}], []
        ),
    )
    monkeypatch.setattr(search_mod, "_search_arxiv", make_fake([], []))
    monkeypatch.setattr(search_mod, "_search_github", fake_github)

    results = _run_node(
        "Is EvalAI practical for non-ML users — should it be used as a "
        "platform, service, or library, and how does it compare with the "
        "open-source options Promptfoo and Opik?"
    )
    assert github_calls == ["evalai practical", "evalai"]
    assert results[0]["engine"] == "exa+github"
    assert "Cloud-CV/EvalAI" in results[0]["content"]


def test_cache_key_includes_github_phrase(monkeypatch):
    """The cache key for a GitHub-routed query carries the derived phrase,
    so entries cached before phrase derivation (no repo grounding) are
    re-fetched rather than served stale."""
    q = (
        "Do the open-source libraries DeepEval or Ragas offer templates, "
        "no-code, or low-code workflows for users without ML expertise?"
    )
    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": q, "priority": "high"}])
    qid = f"q_{plan_id}_0"
    set_question_in_progress(qid)

    # An old-format (pre-derivation) entry exists for this query...
    base = (
        f"{q}||exa||False||{search_config.arxiv_max_results}||True"
        f"||{search_config.github_max_results}"
    )
    store_cache(
        base,
        [{"title": "Stale", "url": "https://stale.example", "content": "stale"}],
        engine="exa",
    )
    # ...but the node must not serve it: its key is phrase-qualified.
    exa_calls = []
    monkeypatch.setattr(
        search_mod,
        "_search_exa",
        make_fake(
            [{"title": "Fresh", "url": "https://fresh.example", "content": "x" * 100}],
            exa_calls,
        ),
    )
    monkeypatch.setattr(search_mod, "_search_arxiv", make_fake([], []))
    monkeypatch.setattr(search_mod, "_search_github", make_fake([], []))

    search_execution_node({"plan_ref": plan_id})
    assert len(exa_calls) == 1, "stale pre-derivation cache entry must not be served"
    row = get_raw_results_for_question(qid)[0]
    assert "fresh.example" in row["content"]
    assert "stale.example" not in row["content"]
    # The fresh merge was stored under the phrase-qualified key:
    assert check_cache(base + "||deepeval ragas") is not None


# ── Node-level: force_arxiv routing ──────────────────────────────────


def test_force_arxiv_runs_only_arxiv(monkeypatch):
    exa_calls = []
    arxiv_calls = []
    monkeypatch.setattr(search_config, "primary", "exa")
    monkeypatch.setattr(search_config, "force_arxiv", "arxiv")
    monkeypatch.setattr(search_mod, "_search_exa", make_fake([], exa_calls))
    monkeypatch.setattr(
        search_mod,
        "_search_arxiv",
        make_fake(
            [{"title": "P", "url": "https://arxiv.org/abs/1", "content": "x" * 100}],
            arxiv_calls,
        ),
    )

    results = _run_node("what is a zorblatt")  # everyday, but forced to arxiv
    assert exa_calls == [], "web engine must not run when force=arxiv"
    assert len(arxiv_calls) == 1
    assert results[0]["engine"] == "arxiv"
    assert "arxiv.org/abs/1" in results[0]["content"]


def test_force_web_skips_arxiv_even_for_science(monkeypatch):
    exa_calls = []
    arxiv_calls = []
    monkeypatch.setattr(search_config, "force_arxiv", "web")
    monkeypatch.setattr(
        search_mod,
        "_search_exa",
        make_fake(
            [{"title": "W", "url": "https://example.com/a", "content": "x" * 100}],
            exa_calls,
        ),
    )
    monkeypatch.setattr(search_mod, "_search_arxiv", make_fake([], arxiv_calls))

    results = _run_node("deep learning survey")  # science, but forced web-only
    assert len(exa_calls) == 1
    assert arxiv_calls == []
    assert results[0]["engine"] == "exa"


def test_force_both_runs_arxiv_for_everyday(monkeypatch):
    exa_calls = []
    arxiv_calls = []
    monkeypatch.setattr(search_config, "force_arxiv", "both")
    monkeypatch.setattr(
        search_mod,
        "_search_exa",
        make_fake(
            [{"title": "W", "url": "https://example.com/a", "content": "x" * 100}],
            exa_calls,
        ),
    )
    monkeypatch.setattr(
        search_mod,
        "_search_arxiv",
        make_fake(
            [{"title": "P", "url": "https://arxiv.org/abs/1", "content": "x" * 100}],
            arxiv_calls,
        ),
    )

    results = _run_node("what is a zorblatt")  # everyday, but forced both
    assert len(exa_calls) == 1
    assert len(arxiv_calls) == 1
    assert results[0]["engine"] == "exa+arxiv"


# ── Node-level: per-domain cap (source diversity) ────────────────────


def test_domain_cap_arxiv_uncapped_on_science(monkeypatch):
    """arxiv.org is exempt from the per-domain cap, so every paper the
    arXiv leg returns survives — previously only a raised cap of
    max(3, min(arxiv_max_results, 6)) applied."""
    exa_calls = []
    arxiv_calls = []
    monkeypatch.setattr(
        search_mod,
        "_search_exa",
        make_fake(
            [{"title": "W", "url": "https://example.com/a", "content": "x" * 100}],
            exa_calls,
        ),
    )
    arxiv_results = [
        {"title": f"P{i}", "url": f"https://arxiv.org/abs/{i}", "content": "y" * 100}
        for i in range(6)
    ]
    monkeypatch.setattr(
        search_mod, "_search_arxiv", make_fake(arxiv_results, arxiv_calls)
    )

    results = _run_node("deep learning survey")
    content = results[0]["content"]
    assert content.count("arxiv.org/abs/") == 6  # all papers survive
    assert content.count("example.com/a") == 1


def test_domain_cap_default_for_nonscience(monkeypatch):
    """Without an arXiv leg, the default per-domain cap of 3 applies."""
    exa_calls = []
    monkeypatch.setattr(
        search_mod,
        "_search_exa",
        make_fake(
            [
                {
                    "title": f"W{i}",
                    "url": f"https://example.com/{i}",
                    "content": "x" * 100,
                }
                for i in range(5)
            ],
            exa_calls,
        ),
    )
    monkeypatch.setattr(search_mod, "_search_arxiv", make_fake([], []))

    results = _run_node("what is a zorblatt")
    content = results[0]["content"]
    assert content.count("example.com/") == 3  # default cap of 3


def test_domain_cap_github_leg_survives_exa_github_urls(monkeypatch):
    """Regression: the github.com cap used to be shared between Exa's own
    github.com URLs and the GitHub leg's repos (first-come-first-served),
    so the leg's repos were silently truncated to the leftover slots.
    github.com is now uncapped: every leg repo survives whatever Exa
    returned."""
    github_calls = []
    exa_github = [
        {
            "title": f"exa-repo{i}/x",
            "url": f"https://github.com/exa-repo{i}/x",
            "content": "Exa-scraped GitHub page with a README excerpt. " + "x" * 40,
        }
        for i in range(4)
    ]
    leg_repos = [
        {
            "title": f"leg-repo{i}/y",
            "url": f"https://github.com/leg-repo{i}/y",
            "content": f"desc | {100 - i} stars | Python",
        }
        for i in range(5)
    ]
    monkeypatch.setattr(search_mod, "_search_exa", make_fake(exa_github, []))
    monkeypatch.setattr(search_mod, "_search_arxiv", make_fake([], []))
    monkeypatch.setattr(
        search_mod, "_search_github", make_fake(leg_repos, github_calls)
    )
    monkeypatch.setattr(search_mod, "_search_brave", make_fake([], []))

    results = _run_node(
        "Which of the open-source tools Promptfoo and Opik is more usable "
        "for non-ML users setting up agent evaluations without Python or ML expertise?"
    )
    content = results[0]["content"]
    assert results[0]["engine"] == "exa+github"
    for i in range(5):  # all five leg repos survive
        assert f"leg-repo{i}/y" in content
    for i in range(4):  # Exa's own GitHub pages survive too
        assert f"exa-repo{i}/x" in content


def test_enforce_domain_cap_caps_generic_domain():
    """The default cap of 3 still applies to ordinary domains."""
    results = [{"url": f"https://en.wikipedia.org/wiki/P{i}"} for i in range(5)]
    assert len(search_mod._enforce_domain_cap(results)) == 3


def test_enforce_domain_cap_github_and_publishers_uncapped():
    """github.com and academic publishers (incl. subdomains) are never
    capped — every URL there is a distinct primary source."""
    results = (
        [{"url": f"https://github.com/o/repo{i}"} for i in range(9)]
        + [{"url": f"https://arxiv.org/abs/{i}"} for i in range(6)]
        + [{"url": f"https://dl.acm.org/doi/10.1145/{i}"} for i in range(4)]
        + [{"url": f"https://link.springer.com/article/{i}"} for i in range(4)]
    )
    assert len(search_mod._enforce_domain_cap(results)) == 23


def test_enforce_domain_cap_uncapped_domains_do_not_block_others():
    """Exempt domains neither consume nor block the cap for other domains."""
    results = [{"url": "https://github.com/o/r"} for _ in range(5)] + [
        {"url": f"https://example.com/{i}"} for i in range(5)
    ]
    kept = search_mod._enforce_domain_cap(results)
    assert sum("github.com" in r["url"] for r in kept) == 5
    assert sum("example.com" in r["url"] for r in kept) == 3


# ── Node-level: GitHub low-quality signal (bug K) ────────────────────


def test_github_results_with_content_are_not_low_quality():
    """GitHub results that carry real content (OSS READMEs via Exa) must NOT
    be flagged low-quality — only thin GitHub results count."""
    github_rich = [
        {"url": "https://github.com/repo", "content": "A" * 200},
        {"url": "https://github.com/repo2", "content": "B" * 200},
        {"url": "https://github.com/repo3", "content": "C" * 200},
    ]
    assert search_mod._results_look_low_quality(github_rich) is False

    github_thin = [
        {"url": "https://github.com/repo", "content": ""},
        {"url": "https://github.com/repo2", "content": ""},
        {"url": "https://github.com/repo3", "content": ""},
    ]
    assert search_mod._results_look_low_quality(github_thin) is True


# ── Node-level: honest empty on total failure (bug P) ────────────────


def test_total_failure_stores_honest_empty(monkeypatch):
    """When every engine fails, the stored raw result is honest (title =
    query, empty content) — not a fake 'Error: ...' placeholder row."""
    exa_calls = []
    brave_calls = []
    monkeypatch.setattr(
        search_mod, "_search_exa", make_fake(RuntimeError("exa down"), exa_calls)
    )
    monkeypatch.setattr(
        search_mod, "_search_brave", make_fake(RuntimeError("brave down"), brave_calls)
    )
    monkeypatch.setattr(search_mod, "_search_arxiv", make_fake([], []))

    results = _run_node("what is a zorblatt")
    assert len(results) == 1
    row = results[0]
    assert row["title"] == "what is a zorblatt"
    assert "Error:" not in row["content"]
    assert row["content"] == ""


# ── Node-level: KB hits prepended on the cache-hit path (bug J) ──────


def test_cache_hit_prepends_kb_hits(monkeypatch):
    """The cache-hit path must prepend knowledge-base hits, matching the
    fresh-search path (previously it dropped them)."""
    q = "zorblatt coverage for the test"
    plan_id = create_plan("Test", [])
    create_questions(plan_id, [{"question": q, "priority": "high"}])
    qid = f"q_{plan_id}_0"
    set_question_in_progress(qid)

    # KB entry whose summary contains every word of the question, linked to
    # the plan via a finding (for plan scoping).
    entry = create_knowledge_entry(
        title="Zorblatt entry",
        summary="zorblatt coverage for the test details",
        topic_tags=["zorblatt"],
        full_text_path="",
        source_url="https://kb.example/zorblatt",
    )
    create_finding(
        question_id=qid,
        raw_result_id=None,
        knowledge_entry_id=entry,
        facts=[],
        summary="prior finding",
        tags=[],
    )

    # Pre-populate the cache (non-science query -> arxiv=False; no OSS
    # vocabulary -> github=False).
    store_cache(
        f"{q}||exa||False||{search_config.arxiv_max_results}||False"
        f"||{search_config.github_max_results}",
        [{"title": "Cached", "url": "https://example.com", "content": "cached body"}],
        engine="exa",
    )

    # iteration_count > 1 so the KB search actually runs.
    search_execution_node({"plan_ref": plan_id, "iteration_count": 2})

    results = get_raw_results_for_question(qid)
    content = results[0]["content"]
    assert "[KNOWLEDGE BASE HITS]" in content
    assert "Zorblatt entry" in content
    assert "cached body" in content


# ── Node-level: cache key includes arxiv_max_results ─────────────────


def test_cache_key_includes_arxiv_max_results(monkeypatch):
    """A change to arxiv_max_results must invalidate the cache (the key
    includes it), so a stale merge is not served."""
    exa_calls = []
    arxiv_calls = []
    q = "deep learning survey"
    monkeypatch.setattr(
        search_mod,
        "_search_exa",
        make_fake(
            [{"title": "W", "url": "https://example.com/a", "content": "x" * 100}],
            exa_calls,
        ),
    )
    monkeypatch.setattr(
        search_mod,
        "_search_arxiv",
        make_fake(
            [{"title": "P", "url": "https://arxiv.org/abs/1", "content": "x" * 100}],
            arxiv_calls,
        ),
    )

    # First run at max_results=5 stores under a key that includes 5.
    monkeypatch.setattr(search_config, "arxiv_max_results", 5)
    _run_node(q)
    assert len(arxiv_calls) == 1

    # Re-run with a DIFFERENT max_results -> different key -> cache miss ->
    # the engines are called again.
    monkeypatch.setattr(search_config, "arxiv_max_results", 6)
    _run_node(q)
    assert len(arxiv_calls) == 2, "changing arxiv_max_results should bypass the cache"
