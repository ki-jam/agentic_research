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
        defaults = _get_provider_defaults()
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
            task: Optional task name for task-specific model routing.
                Supported tasks: "extraction", "curation", "planning", "report", "summarize"
                Per-stage overrides live on WorkflowConfig (EXTRACTION_MODEL,
                CURATION_MODEL, PLANNING_MODEL, REPORT_MODEL, SUMMARIZE_MODEL);
                falls back to provider-specific model if not set.
        """
        max_tokens = self.max_tokens if max_tokens is None else max_tokens

        # Resolve task-specific model (per-stage overrides live on
        # WorkflowConfig, not on this LLM config)
        if not model and task:
            task_model = getattr(workflow_config, f"{task}_model", "")
            if task_model:
                model = task_model

        # Provider-specific defaults
        provider = self.provider
        effective_model = model or getattr(self, f"{provider}_model", self.local_model)
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

        return ChatOpenAI(  # fallback
            model=self.local_model,
            base_url=self.local_base_url,
            api_key="not-needed",
            temperature=self.temperature,
            max_tokens=max_tokens,
        )


@dataclass(frozen=False)
class SearchConfig:
    """Search engine configuration."""

    # Provider: "exa", "brave", "arxiv", or a combo (env: SEARCH_PRIMARY)
    primary: str = field(default_factory=lambda: os.getenv("SEARCH_PRIMARY", "exa"))

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

    # ArXiv (no key needed)
    arxiv_max_results: int = field(
        default_factory=lambda: int(os.getenv("ARXIV_MAX_RESULTS", "10"))
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

    # Per-step model overrides (empty = use main config model)
    extraction_model: str = field(
        default_factory=lambda: os.getenv("EXTRACTION_MODEL", "")
    )
    curation_model: str = field(default_factory=lambda: os.getenv("CURATION_MODEL", ""))
    planning_model: str = field(default_factory=lambda: os.getenv("PLANNING_MODEL", ""))
    report_model: str = field(default_factory=lambda: os.getenv("REPORT_MODEL", ""))
    summarize_model: str = field(
        default_factory=lambda: os.getenv("SUMMARIZE_MODEL", "")
    )


# ── Provider defaults ─────────────────────────────────────────────
_PROVIDER_DEFAULTS: dict[str, dict] = {
    "local": {"temperature": 0.3},
    "openai": {"temperature": 0.3},
    "anthropic": {"temperature": 1.0},
}


def _get_provider_defaults() -> dict:
    """Return the default config for the currently selected provider."""
    provider = os.getenv("PROVIDER", "local")
    return _PROVIDER_DEFAULTS.get(provider, _PROVIDER_DEFAULTS["local"])


# Global config singleton
config = LLMConfig()
search_config = SearchConfig()
paths_config = PathsConfig()
workflow_config = WorkflowConfig()


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

# Terms that mark a research topic as science/math/ML. Substring match — used
# both to add arXiv to the search mix (nodes/search.py, per question) and to
# enforce at-least-one-arXiv-result in the consolidation gate (any
# science-flagged question in the plan). Broad everyday words ("model",
# "paper", "math", ...) are intentionally excluded to avoid over-triggering.
SCIENCE_TERMS = frozenset(
    {
        "algorithm",
        "training",
        "reinforcement",
        "ppo",
        "grpo",
        "gpto",
        "neural",
        "deep learning",
        "optimization",
        "loss",
        "gradient",
        "policy",
        "reward",
        "benchmark",
        "arxiv",
        "theorem",
        "proof",
        "probability",
        "statistics",
        "regression",
        "classification",
        "transformer",
        "attention",
        "embedding",
        "fine-tune",
        "sft",
        "rlhf",
        "alignment",
        "pretrain",
        "inference",
        "latency",
        "convergence",
        "generalization",
        "overfitting",
        "regularization",
        "contrastive",
        "generative",
        "diffusion",
        "cnn",
        "rnn",
        "lstm",
        "language model",
        "llm",
        "large language",
        "reasoning",
        "agentic",
        "process reward",
        "critic",
        "value function",
        "kl divergence",
        "monte carlo",
        "mcts",
        "bandit",
        "markov",
        "dynamic programming",
        "backprop",
        "autodiff",
        "hyperparameter",
        "batch",
        "epoch",
        "dataset",
        "evaluation",
        "accuracy",
        "f1",
        "precision",
        "recall",
    }
)
