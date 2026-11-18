"""
Configuration for the research agent.

All settings are configurable at runtime via environment variables (loaded
from .env) or direct overrides via update_config().
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Load .env file if present
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass  # python-dotenv not installed, rely on env vars


@dataclass(frozen=False)
class LLMConfig:
    """LLM provider configuration — switch between local, OpenAI, Anthropic."""

    # Provider: "local", "openai", "anthropic" (env: PROVIDER)
    provider: str = field(default_factory=lambda: os.getenv("PROVIDER", "local"))

    # Local models (llama.cpp / ollama compatible)
    local_model: str = field(
        default_factory=lambda: os.getenv("LOCAL_MODEL", "qwen36_35")
    )
    local_base_url: str = field(
        default_factory=lambda: os.getenv("LOCAL_BASE_URL", "http://localhost:8080/v1")
    )

    # OpenAI
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o")
    )

    # Anthropic
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    anthropic_model: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    )
    anthropic_base_url: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_BASE_URL", "")
    )

    # Shared LLM params (overridden by provider defaults at init)
    temperature: float = 0.3
    max_tokens: int = 16384

    def __post_init__(self):
        """Apply provider-specific defaults after dataclass init."""
        defaults = _get_provider_defaults(self.provider)
        if "temperature" in defaults:
            self.temperature = defaults["temperature"]

    def get_chat_model(
        self, model: str = "", max_tokens: int | None = None, task: str = ""
    ):
        """Return the appropriate LangChain chat model instance.

        Args:
            model: Optional model name override. Falls back to provider-specific model.
            max_tokens: Optional output-token budget override. Thinking models can
                spend the whole budget on reasoning and return an empty answer, so
                long-output calls (the final report) pass a larger budget.
            task: Optional task name for task-specific routing.
                Supported tasks: "extraction", "curation", "planning", "report".
                Per-step overrides live on WorkflowConfig: ``{task}_provider``
                (EXTRACTION_PROVIDER, … — empty = global PROVIDER) and
                ``{task}_model`` (EXTRACTION_MODEL, … — empty = that provider's
                default model). This makes cross-provider steps possible, e.g.
                local extraction + Anthropic report.
        """
        max_tokens = self.max_tokens if max_tokens is None else max_tokens

        # Resolve the effective provider + default model (per-step overrides
        # live on WorkflowConfig, not on this LLM config)
        if task:
            provider, default_model = resolve_step_model(task)
        else:
            provider = self.provider
            default_model = getattr(self, f"{provider}_model", self.local_model)

        effective_model = model or default_model
        effective_temp = self.temperature

        if provider == "anthropic":
            # Claude Sonnet 5+ only accepts temperature=0 or 1.0
            effective_temp = 1.0 if "sonnet-5" in effective_model else self.temperature

        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=effective_model or self.openai_model,
                api_key=self.openai_api_key or None,
                temperature=effective_temp,
                max_tokens=max_tokens,
            )
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            kwargs: dict = {
                "model": effective_model or self.anthropic_model,
                "api_key": self.anthropic_api_key or None,
                "temperature": effective_temp,
                "max_tokens": max_tokens,
            }
            if self.anthropic_base_url:
                kwargs["anthropic_api_url"] = self.anthropic_base_url
            return ChatAnthropic(**kwargs)
        else:  # local
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=effective_model or self.local_model,
                base_url=self.local_base_url,
                api_key=os.getenv("LOCAL_API_KEY") or "not-needed",
                temperature=effective_temp,
                max_tokens=max_tokens,
            )


@dataclass(frozen=False)
class SearchConfig:
    """Search engine configuration."""

    # Provider: "exa", "brave", "arxiv", "exa+brave" (dual-search, prefer narrative)
    # "exa+brave" runs Exa first, falls back to Brave if Exa results look low-quality
    primary: str = field(default_factory=lambda: os.getenv("SEARCH_PRIMARY", "exa"))

    # arXiv routing override: "auto" (tiered science vocabulary decides),
    # "web" (never search arXiv), "arxiv" (always arXiv-only, promotes arXiv
    # to the primary engine), or "both" (always web + arXiv). See
    # utils/search_routing.py for the auto rules.
    force_arxiv: str = field(
        default_factory=lambda: os.getenv("SEARCH_FORCE_ARXIV", "auto")
    )

    # Exa
    exa_api_key: str = field(default_factory=lambda: os.getenv("EXA_API_KEY", ""))

    # Citation discovery via Exa answer API (finds foundational papers)
    enable_citation_discovery: bool = field(
        default_factory=lambda: (
            os.getenv("ENABLE_CITATION_DISCOVERY", "true").lower() == "true"
        )
    )

    # Brave
    brave_api_key: str = field(default_factory=lambda: os.getenv("BRAVE_API_KEY", ""))

    # ArXiv (no key needed) — uses the arxiv Python package (typed, rate-limited)
    arxiv_max_results: int = field(
        default_factory=lambda: int(os.getenv("ARXIV_MAX_RESULTS", "10"))
    )

    # GitHub repositories leg — OSS-flavored queries (OSS_TERMS in
    # utils/search_routing.py) also search GitHub repos sorted by stars.
    # Keyless by default (10 search req/min); a GITHUB_TOKEN raises the
    # limit to 30 req/min.
    github_token: str = field(default_factory=lambda: os.getenv("GITHUB_TOKEN", ""))
    github_max_results: int = field(
        default_factory=lambda: int(os.getenv("GITHUB_MAX_RESULTS", "5"))
    )

    # GitHub routing override: "auto" (OSS vocabulary decides), "on"
    # (always run the GitHub leg), "off" (never). See utils/search_routing.py.
    force_github: str = field(
        default_factory=lambda: os.getenv("SEARCH_FORCE_GITHUB", "auto")
    )

    # Playwright browser search — enriches Exa/Brave results by scraping
    # pages with short/empty content (docs, blogs, tutorials).
    browser_search_enabled: bool = field(
        default_factory=lambda: os.getenv("BROWSER_SEARCH_ENABLED", "false").lower()
        == "true"
    )

    # Search-result cache TTL in days (env: SEARCH_CACHE_TTL_DAYS).
    # 0 = cached results never expire. Was hardcoded to 7 days in
    # db/knowledge.py; raised to 30 so follow-up sessions on the same
    # topic reuse the earlier searches (set 0 to keep them forever).
    cache_ttl_days: int = field(
        default_factory=lambda: int(os.getenv("SEARCH_CACHE_TTL_DAYS", "30"))
    )


@dataclass(frozen=False)
class PathsConfig:
    """File system paths."""

    base_dir: str = field(
        default_factory=lambda: os.getenv(
            "RESEARCH_BASE_DIR", os.path.join(os.path.dirname(__file__), "..", "data")
        )
    )

    @property
    def db_path(self) -> str:
        p = os.path.join(self.base_dir, "knowledge.db")
        os.makedirs(self.base_dir, exist_ok=True)
        return p

    @property
    def checkpoint_path(self) -> str:
        """LangGraph checkpoint store (session persistence / --resume)."""
        p = os.path.join(self.base_dir, "checkpoints.db")
        os.makedirs(self.base_dir, exist_ok=True)
        return p


@dataclass(frozen=False)
class WorkflowConfig:
    """Graph workflow settings (env: MAX_ITERATIONS, REVIEW_EVERY, MIN_FINDINGS,
    EXTRACTION_MODEL, CURATION_MODEL, PLANNING_MODEL)."""

    max_iterations: int = field(
        default_factory=lambda: int(os.getenv("MAX_ITERATIONS", "10"))
    )
    review_every_n_iterations: int = field(
        default_factory=lambda: int(os.getenv("REVIEW_EVERY", "5"))
    )
    min_findings: int = field(
        default_factory=lambda: int(os.getenv("MIN_FINDINGS", "3"))
    )
    refine_query_threshold: int = field(
        default_factory=lambda: int(os.getenv("REFINE_QUERY_THRESHOLD", "20"))
    )

    # Ready-gate / coverage floor: the readiness LLM's "ready" verdict is an
    # opinion about report quality, not proof that the question bank is
    # covered. When the gate is enabled, auto-stop (write_findings) is
    # blocked if any high-priority question is still open, or if the
    # closed/total ratio (resolved+discarded counts as closed) is below
    # ready_min_resolution — the session routes to human_review instead, so
    # the user can stop (interim report) or continue.
    ready_coverage_gate: bool = field(
        default_factory=lambda: (
            os.getenv("READY_COVERAGE_GATE", "true").lower() == "true"
        )
    )
    ready_min_resolution: float = field(
        default_factory=lambda: float(os.getenv("READY_MIN_RESOLUTION", "0.5"))
    )

    # Per-step model overrides (empty = that step's provider's default model).
    extraction_model: str = field(
        default_factory=lambda: os.getenv("EXTRACTION_MODEL", "")
    )
    curation_model: str = field(default_factory=lambda: os.getenv("CURATION_MODEL", ""))
    planning_model: str = field(default_factory=lambda: os.getenv("PLANNING_MODEL", ""))
    report_model: str = field(default_factory=lambda: os.getenv("REPORT_MODEL", ""))

    # Per-step provider overrides (empty = global PROVIDER). Lets a single
    # session mix providers, e.g. local extraction + Anthropic report:
    #   EXTRACTION_PROVIDER=local  REPORT_PROVIDER=anthropic  REPORT_MODEL=…
    extraction_provider: str = field(
        default_factory=lambda: os.getenv("EXTRACTION_PROVIDER", "")
    )
    curation_provider: str = field(
        default_factory=lambda: os.getenv("CURATION_PROVIDER", "")
    )
    planning_provider: str = field(
        default_factory=lambda: os.getenv("PLANNING_PROVIDER", "")
    )
    report_provider: str = field(
        default_factory=lambda: os.getenv("REPORT_PROVIDER", "")
    )


# ── Provider defaults ─────────────────────────────────────────────
_PROVIDER_DEFAULTS: dict[str, dict] = {
    "local": {"temperature": 0.3},
    "openai": {"temperature": 0.3},
    "anthropic": {"temperature": 1.0},
}


def _get_provider_defaults(provider: str | None = None) -> dict:
    """Return the default config for a provider.

    Takes the provider explicitly (callers pass ``self.provider``) so the
    result follows the configured instance rather than re-reading the
    PROVIDER env var, which can diverge after update_config().
    """
    provider = provider or os.getenv("PROVIDER", "local")
    return _PROVIDER_DEFAULTS.get(provider, _PROVIDER_DEFAULTS["local"])


# Global config singleton
config = LLMConfig()
search_config = SearchConfig()
paths_config = PathsConfig()
workflow_config = WorkflowConfig()

_VALID_PROVIDERS = ("local", "openai", "anthropic")


def resolve_step_model(task: str) -> tuple[str, str]:
    """Return the ``(provider, model)`` that workflow step ``task`` runs on.

    Resolution order:
      provider: ``WorkflowConfig.{task}_provider`` (e.g. REPORT_PROVIDER),
          falling back to the global ``PROVIDER``.
      model:    ``WorkflowConfig.{task}_model`` (e.g. REPORT_MODEL),
          falling back to that provider's default model (LOCAL_MODEL /
          OPENAI_MODEL / ANTHROPIC_MODEL).

    Raises ValueError for an unknown provider so a typo in e.g.
    ``REPORT_PROVIDER`` fails loudly at session start (the effective-config
    banner resolves every step) instead of silently running the step on the
    local model.
    """
    provider = getattr(workflow_config, f"{task}_provider", "") or config.provider
    if provider not in _VALID_PROVIDERS:
        raise ValueError(
            f"Unknown provider {provider!r} for step {task!r} "
            f"(expected one of: {', '.join(_VALID_PROVIDERS)} — "
            f"check the {'_'.join((task.upper(), 'PROVIDER'))} env var)"
        )
    model = getattr(workflow_config, f"{task}_model", "") or getattr(
        config, f"{provider}_model", config.local_model
    )
    return provider, model


# Mapping of config field → (target object, target attribute name).
# Adding a new config category only requires adding entries here.
_CONFIG_TARGETS: dict[str, tuple[object, str]] = {}
for _obj in (config, workflow_config):
    for _field in _obj.__dataclass_fields__:  # type: ignore[attr-defined]
        _CONFIG_TARGETS[_field] = (_obj, _field)
del _obj, _field


def update_config(**kwargs):
    """Override LLM or workflow config fields at runtime.

    Only fields on LLMConfig and WorkflowConfig are updatable; search and
    paths config are fixed at startup.

    Examples:
        update_config(provider="openai")
        update_config(max_iterations=15)
        update_config(local_model="other-model")
    """
    for key, value in kwargs.items():
        target = _CONFIG_TARGETS.get(key)
        if target is None:
            raise ValueError(f"Unknown config field: {key}")
        setattr(target[0], target[1], value)


# ── Topic classification ─────────────────────────────────────────────
#
# The science/math/ML vocabulary that decides whether a query also searches
# arXiv now lives in utils/search_routing.py (tiered STRONG/MEDIUM/WEAK terms
# with word-boundary matching). It is used both by the search node (per
# question) and the consolidation gate (via utils.quality.is_science_topic),
# which stay consistent because both read SearchConfig.force_arxiv.
