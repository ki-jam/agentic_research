"""
Scope guards — keep the session anchored to the original query.

The curation/extraction/readiness prompts used to carry DISCARD rules
hard-coded to one specific past query (LLM serving: vLLM, SGLang, TGI,
Kafka, quantization libraries...). For any other topic those rules were
either silent or actively wrong.

Now every plan stores a deterministic scope block derived from its own
query (computed once at planning time, persisted on the plan row).
Prompts consume it via a {scope} placeholder. No extra LLM call:
entity extraction is a light heuristic (capitalized / all-caps tokens,
minus generic English words and domain-generic acronyms).
"""

from __future__ import annotations

import re

# Capitalized words that start sentences or are generic, not entities.
_CAPITALIZED_STOPWORDS = {
    "The",
    "A",
    "An",
    "This",
    "That",
    "These",
    "Those",
    "What",
    "Why",
    "How",
    "When",
    "Where",
    "Which",
    "Who",
    "Is",
    "Are",
    "Was",
    "Were",
    "Do",
    "Does",
    "Did",
    "In",
    "On",
    "For",
    "Of",
    "With",
    "About",
    "To",
    "And",
    "Or",
    "But",
    "Not",
    "If",
    "Then",
    "Than",
    "So",
    "Such",
    "As",
    "At",
    "By",
    "From",
    "Up",
    "Out",
    "Into",
    "Over",
    "Under",
    "Between",
    "During",
    "Before",
    "After",
    "Since",
    "While",
    "All",
    "Any",
    "Both",
    "Each",
    "Few",
    "More",
    "Most",
    "Other",
    "Some",
    "No",
    "Nor",
    "Only",
    "Own",
    "Same",
    "Can",
    "Will",
    "Just",
    "Should",
    "Could",
    "Would",
    "May",
    "Might",
    "Must",
    "There",
    "Here",
    "They",
    "Them",
    "Their",
    "We",
    "You",
    "Your",
    "Our",
    "His",
    "Her",
    "Its",
    "My",
    "Me",
    "He",
    "She",
    "It",
}

# All-caps acronyms that are domain-generic rather than named systems.
_GENERIC_ACRONYMS = {
    "AI",
    "API",
    "CPU",
    "GPU",
    "HTTP",
    "HTTPS",
    "IDE",
    "JSON",
    "LLM",
    "ML",
    "NLP",
    "SDK",
    "SQL",
    "URL",
    "XML",
    "IOT",
}

# Capitalized words that commonly lead research queries (verbs/framing),
# not named systems. Without this list, "Compare vLLM and TGI" would
# extract "Compare" as an entity.
_QUERY_LEAD_WORDS = {
    "Compare",
    "Comparing",
    "Comparison",
    "Comparisons",
    "Versus",
    "Vs",
    "Explain",
    "Explaining",
    "Describe",
    "Describing",
    "Analyze",
    "Analyzing",
    "Discuss",
    "Discussing",
    "Discussion",
    "Evaluate",
    "Evaluating",
    "Review",
    "Reviewing",
    "Summarize",
    "Summarizing",
    "Investigate",
    "Investigating",
    "Identify",
    "Identifying",
    "Explore",
    "Exploring",
    "Understanding",
    "Understand",
    "Choosing",
    "Choose",
    "Future",
    "Current",
    "Overview",
    "Trends",
    "Landscape",
    "Ecosystem",
    "Best",
    "Practices",
    "Guide",
    "Getting",
    "Started",
    "State",
    "Pros",
    "Cons",
    "Tradeoffs",
    "Scaling",
    "Optimizing",
    "Deploying",
    "Benchmarks",
    "Benchmark",
    "Alternatives",
    "Options",
    "Migration",
    "Migrating",
}

# A token: starts with a letter, may continue with letters/digits/+.-
# (covers vLLM, SGLang, K8s, C++, oneAPI). Years like "2026" don't match.
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*")


def extract_entities(query: str) -> list[str]:
    """Extract named systems/entities from a query (deterministic).

    Heuristic: tokens containing an uppercase letter (proper nouns and
    all-caps acronyms), minus generic English words and domain-generic
    acronyms. First-appearance order is kept; dedupe is case-insensitive.

    >>> extract_entities("Compare vLLM, SGLang, and TGI for LLM serving")
    ['vLLM', 'SGLang', 'TGI']
    """
    entities: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN_RE.findall(query or ""):
        if token in _CAPITALIZED_STOPWORDS or token in _QUERY_LEAD_WORDS:
            continue
        if token.upper() in _GENERIC_ACRONYMS:
            continue
        if not any(c.isupper() for c in token):
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        entities.append(token)
    return entities


def build_scope_block(query: str) -> str:
    """Build the deterministic scope block injected into the prompts.

    Anchors the session to the topic and lists the named systems from
    the query, so the DISCARD/gap rules in the prompts stay generic
    while remaining query-specific.
    """
    topic = (query or "").strip()
    entities = extract_entities(topic)

    lines = [
        "RESEARCH SCOPE (enforced):",
        f'Topic: "{topic}"',
    ]
    if entities:
        lines.append(f"Named systems in scope: {', '.join(entities)}")
        lines.append(
            "Content about a named system is in scope. Content about OTHER "
            "systems is in scope only if the source directly compares it "
            "to a named system or connects it to the topic."
        )
    else:
        lines.append(
            "No specific systems are named in the topic; stay on the topic "
            "itself and its direct sub-areas."
        )
    lines.append(
        "Discard findings, entries, or gaps about tools, frameworks, or "
        "topics OUTSIDE this scope (e.g., unrelated infrastructure, "
        "messaging, storage, or training tooling) unless the source "
        "explicitly connects them to the topic."
    )
    return "\n".join(lines)


def scope_for_plan(plan_id: str) -> str:
    """The scope block for a plan: stored value, or rebuilt from the query.

    Rebuilding keeps plans created before the scope column existed (or
    with an empty scope) working without a backfill migration.
    """
    if not plan_id:
        return ""
    try:
        from db.knowledge import get_plan

        plan = get_plan(plan_id)
    except Exception:
        return ""
    if not plan:
        return ""
    scope = (plan.get("scope") or "").strip()
    if scope:
        return scope
    return build_scope_block(plan.get("query", ""))
