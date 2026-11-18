"""
Search routing — decide whether a query needs web search, arXiv, or both,
and whether to run the supplementary GitHub repositories leg (OSS topics).

Replaces the old flat ``SCIENCE_TERMS`` substring check, which both
over-triggered (``"suppose"`` contains ``"ppo"``, ``"CNN report"`` contains
``"cnn"``) and under-triggered (``"machine learning"`` was not in the list
at all).

Routing is based on a three-tier science vocabulary matched at word
boundaries, and produces an explainable :class:`SearchPlan`:

  - any STRONG term            -> arXiv
  - >= 2 MEDIUM terms          -> arXiv
  - 1 MEDIUM + >= 1 WEAK term  -> arXiv
  - >= 3 WEAK terms            -> arXiv
  - otherwise                  -> web only

Single words are matched at word boundaries (case-insensitive) so everyday
words that merely contain a term as a substring do not trigger. Plurals and
simple version suffixes (``gpt-4o``, ``theorems``, ``LLMs``) are allowed.
Phrases match across spaces/hyphens with an optional plural on the last word.

An explicit ``force`` override (``auto`` / ``web`` / ``arxiv`` / ``both``)
lets a user pin the routing regardless of the vocabulary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import List, Tuple

# ── Tiered science vocabulary ────────────────────────────────────────
#
# STRONG: a single hit is enough to route to arXiv. These are unambiguous
# ML/science terms or phrases.
STRONG_TERMS = frozenset(
    {
        # phrases
        "deep learning",
        "machine learning",
        "reinforcement learning",
        "neural network",
        "language model",
        "large language",
        "attention mechanism",
        "diffusion model",
        "generative model",
        "computer vision",
        "natural language processing",
        "speech recognition",
        "graph neural",
        "contrastive learning",
        "knowledge distillation",
        "gradient descent",
        "value function",
        "process reward",
        "dynamic programming",
        "monte carlo",
        "kl divergence",
        "quantum computing",
        "quantum physics",
        "quantum mechanics",
        "protein folding",
        "backpropagation",
        # single words / acronyms
        "neural",
        "convolutional",
        "fine-tune",
        "pretrain",
        "overfitting",
        "regularization",
        "regularisation",
        "hyperparameter",
        "transformer",
        "agentic",
        "robotics",
        "bioinformatics",
        "genomics",
        "llm",
        "gpt",
        "ppo",
        "grpo",
        "rlhf",
        "sft",
        "mcts",
        "lstm",
        "rnn",
        "nlp",
        "tpu",
        "theorem",
        "lemma",
        "corollary",
        "markov",
        "arxiv",
    }
)

# MEDIUM: needs corroboration (2 medium, or 1 medium + 1 weak). These are
# terms that are scientific but also appear in everyday language.
MEDIUM_TERMS = frozenset(
    {
        "algorithm",
        "benchmark",
        "optimization",
        "optimisation",
        "regression",
        "classification",
        "statistics",
        "statistical",
        "probability",
        "dataset",
        "evaluation",
        "reasoning",
        "generative",
        "diffusion",
        "convergence",
        "generalization",
        "generalisation",
        "alignment",
        "latency",
        "gradient",
        "embedding",
        "inference",
        "proof",
        "mathematical",
        "equation",
        "compiler",
        "gpu",
        "cuda",
        "quantum",
        "bandit",
        "cnn",
        "ml",
        "reward function",
        "loss function",
    }
)

# WEAK: only count when several appear together (>= 3). These are very
# common words that happen to have a technical meaning.
WEAK_TERMS = frozenset(
    {
        "training",
        "loss",
        "policy",
        "batch",
        "epoch",
        "reward",
        "attention",
        "precision",
        "accuracy",
        "recall",
        "critic",
        "f1",
    }
)

# ── OSS vocabulary (GitHub leg) ─────────────────────────────────────
#
# A flat word-boundary vocabulary (no tiers): a single hit is enough to
# run the GitHub repositories leg, because these terms almost always mean
# the answer should be grounded in real repos (stars, language, README).
# "open source" also matches "open-source" (phrase patterns allow hyphens).
OSS_TERMS = frozenset(
    {
        "open source",
        "opensource",
        "oss",
        "github",
        "library",
        "framework",
        "sdk",
        "tool",
    }
)

# Pre-compiled patterns, keyed by (tier, term).
_COMPILED: dict[Tuple[str, str], re.Pattern] = {}


def _compile_term(term: str) -> re.Pattern:
    """Compile a word-boundary pattern for a single term or phrase."""
    t = term.lower()
    if " " in t:
        # Phrase: join words with space/hyphen; allow plural + version
        # suffix on the last word only.
        words = t.split()
        parts = [re.escape(w) for w in words[:-1]]
        parts.append(re.escape(words[-1]) + r"(?:s|es)?(?:[-.]?\d+[a-z]?)?")
        body = r"[-\s]+".join(parts)
    else:
        body = re.escape(t) + r"(?:s|es)?(?:[-.]?\d+[a-z]?)?"
    # Lookarounds enforce word boundaries against alphanumerics, so
    # "ppo" does not match inside "suppose"/"appointment"/"property".
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])", re.IGNORECASE)


def _pattern(tier: str, term: str) -> re.Pattern:
    key = (tier, term)
    if key not in _COMPILED:
        _COMPILED[key] = _compile_term(term)
    return _COMPILED[key]


def _match_terms(query: str) -> List[Tuple[str, str]]:
    """Return the (tier, term) pairs that appear in ``query``.

    Each distinct term is reported at most once.
    """
    q = (query or "").lower()
    if not q:
        return []
    seen: set[str] = set()
    matches: List[Tuple[str, str]] = []
    for tier, terms in (
        ("strong", STRONG_TERMS),
        ("medium", MEDIUM_TERMS),
        ("weak", WEAK_TERMS),
    ):
        for term in terms:
            if term in seen:
                continue
            if _pattern(tier, term).search(q):
                seen.add(term)
                matches.append((tier, term))
    return matches


@dataclass(frozen=True)
class SearchPlan:
    """Explainable routing decision for a single query."""

    use_web: bool
    use_arxiv: bool
    matched: tuple  # tuple of (tier, term) pairs
    reason: str
    use_github: bool = False
    github_reason: str = ""


def plan_search(
    query: str, force: str = "auto", force_github: str = "auto"
) -> SearchPlan:
    """Decide the search routing for ``query``.

    Args:
        query: the question / search string.
        force: web/arXiv override — "auto" (default), "web", "arxiv", or
            "both". Unknown values fall back to "auto".
        force_github: GitHub override — "auto" (default, OSS vocabulary
            decides), "on" (always run the repositories leg), or "off"
            (never). Unknown values fall back to "auto".

    Returns:
        A frozen :class:`SearchPlan`.
    """
    return _with_github(_plan_web_arxiv(query, force), query, force_github)


def _plan_web_arxiv(query: str, force: str) -> SearchPlan:
    """The web/arXiv half of the routing decision (see plan_search)."""
    force = (force or "auto").strip().lower()
    if force not in ("auto", "web", "arxiv", "both"):
        force = "auto"

    matched = _match_terms(query)
    matched_t = tuple(matched)

    if force == "web":
        return SearchPlan(True, False, matched_t, "forced web-only")
    if force == "arxiv":
        return SearchPlan(False, True, matched_t, "forced arXiv-only")
    if force == "both":
        return SearchPlan(True, True, matched_t, "forced both")

    # auto: tiered decision
    strong = [t for tier, t in matched if tier == "strong"]
    medium = [t for tier, t in matched if tier == "medium"]
    weak = [t for tier, t in matched if tier == "weak"]

    if "arxiv" in strong:
        return SearchPlan(True, True, matched_t, "explicit arXiv mention")
    if strong:
        return SearchPlan(
            True, True, matched_t, f"strong science term(s): {', '.join(strong[:3])}"
        )
    if len(medium) >= 2:
        return SearchPlan(
            True,
            True,
            matched_t,
            f"multiple medium science terms: {', '.join(medium[:3])}",
        )
    if medium and weak:
        return SearchPlan(
            True,
            True,
            matched_t,
            f"medium + weak science terms: {medium[0]} + {weak[0]}",
        )
    if len(weak) >= 3:
        return SearchPlan(
            True, True, matched_t, f"multiple weak science terms: {', '.join(weak[:4])}"
        )
    return SearchPlan(True, False, matched_t, "no science signal — web only")


def _match_oss_terms(query: str) -> List[str]:
    """Return the OSS vocabulary terms present in ``query`` (sorted).

    Same word-boundary, case-insensitive matching as the science tiers
    (see _compile_term); a single hit is enough (see OSS_TERMS).
    """
    q = (query or "").lower()
    if not q:
        return []
    return [term for term in sorted(OSS_TERMS) if _pattern("oss", term).search(q)]


# ── Compact GitHub query derivation ─────────────────────────────────
#
# GitHub's repository search ANDs every keyword against repository
# name, description, and README, so a full natural-language question
# (twenty-plus words) matches zero repositories. The leg therefore
# searches a compact phrase: the leading salient terms of the question,
# with stopwords and the generic OSS vocabulary that triggered the leg
# stripped (those words describe the topic, not the repositories). The
# search node retries with the first term alone if the two-term phrase
# still matches nothing (e.g. "evalai practical" -> "evalai").

_GITHUB_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "from",
        "with",
        "without",
        "into",
        "onto",
        "about",
        "around",
        "against",
        "between",
        "among",
        "as",
        "be",
        "been",
        "being",
        "am",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "doing",
        "can",
        "could",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "would",
        "will",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "here",
        "there",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "them",
        "his",
        "her",
        "their",
        "our",
        "your",
        "my",
        "me",
        "us",
        "not",
        "no",
        "nor",
        "so",
        "if",
        "then",
        "than",
        "too",
        "very",
        "just",
        "also",
        "only",
        "own",
        "same",
        "such",
        "any",
        "all",
        "each",
        "more",
        "most",
        "other",
        "some",
        "few",
        "both",
        "new",
        "top",
        "best",
        "use",
        "used",
        "using",
        "make",
        "makes",
        "made",
        "get",
        "gets",
        "got",
        "find",
        "finds",
        "looking",
        "need",
        "needs",
        "want",
        "wants",
        "offer",
        "offers",
        "provide",
        "provides",
        "compare",
        "compared",
        "compares",
        "comparison",
        "differ",
        "differs",
        "difference",
        "differences",
    }
)

# Generic OSS vocabulary: these terms trigger the GitHub leg but do not
# identify repositories, so they are stripped from the search phrase.
# OSS_TERMS expanded to the surface forms the tokenizer produces (plurals,
# and "open source" / "open-source" split or kept whole).
_GITHUB_GENERIC = frozenset(
    {
        "open",
        "source",
        "open-source",
        "opensource",
        "oss",
        "github",
        "library",
        "libraries",
        "framework",
        "frameworks",
        "sdk",
        "sdks",
        "tool",
        "tools",
    }
)


def github_search_query(question: str) -> str:
    """Derive the compact phrase the GitHub repositories leg searches for.

    The full question is never sent to GitHub (see the note above): keep
    at most the first two salient tokens — alphanumeric runs, with
    hyphen/dot-joined parts like ``gpt-4o`` kept whole — skipping
    stopwords and generic OSS vocabulary. Returns ``""`` when nothing
    salient remains; the caller then skips the leg rather than searching
    an empty query.
    """
    tokens = re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", (question or "").lower())
    drop = _GITHUB_STOPWORDS | _GITHUB_GENERIC
    kept = [t for t in tokens if t not in drop][:2]
    return " ".join(kept)


def _with_github(
    plan: SearchPlan, query: str, force_github: str = "auto"
) -> SearchPlan:
    """Attach the GitHub repositories-leg decision to a web/arXiv plan.

    The GitHub leg is independent of the web/arXiv decision: it is a
    supplementary repositories leg (repos sorted by stars) for
    OSS-flavored queries, not an alternative engine.
    """
    force_github = (force_github or "auto").strip().lower()
    if force_github not in ("auto", "on", "off"):
        force_github = "auto"
    if force_github == "on":
        return replace(
            plan, use_github=True, github_reason="forced on (SEARCH_FORCE_GITHUB=on)"
        )
    if force_github == "off":
        return replace(
            plan, use_github=False, github_reason="forced off (SEARCH_FORCE_GITHUB=off)"
        )
    oss = _match_oss_terms(query)
    if oss:
        return replace(
            plan,
            use_github=True,
            github_reason=f"OSS vocabulary: {', '.join(oss[:3])}",
        )
    return replace(plan, use_github=False, github_reason="no OSS vocabulary")
