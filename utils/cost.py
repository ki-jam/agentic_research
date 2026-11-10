"""
LLM cost tracking — per-session token/cost accounting.

The tracker lives here (not in config or graph) so that low-level helpers
like utils/llm.py can record usage without importing the graph layer.
build_app() creates one tracker per app and registers it as the active
tracker; invoke_with_cost() records against the active one, so separate
apps (e.g. tests) never share cost state.
"""

from __future__ import annotations


# Pricing per million tokens (USD) — updated for common models.
# Source: https://openai.com/api/pricing/ , https://docs.anthropic.com/en/pricing
# Local models are priced at $0 (free/self-hosted).
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.10, 0.40),
    "gpt-4.1-nano": (0.05, 0.20),
    # Anthropic
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-sonnet-4-5-20250929": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-5-5": (3.00, 15.00),
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-opus-4-5-20251101": (15.00, 75.00),
    "claude-opus-4-6": (15.00, 75.00),
    "claude-opus-4-7": (15.00, 75.00),
    "claude-opus-4-8": (15.00, 75.00),
    "claude-opus-5": (15.00, 75.00),
    "claude-opus-5-5": (15.00, 75.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-haiku-5-5": (0.80, 4.00),
    "claude-fable-5": (0.40, 2.00),
    "claude-fable-5-1": (0.40, 2.00),
    "claude-fable-5-2": (0.40, 2.00),
    # Local/placeholder
    "qwen36_35": (0.00, 0.00),
    "qwen38_27": (0.00, 0.00),
}


class CostTracker:
    """Track API usage and estimated costs across a research session.

    Records per-task call counts and token usage, then estimates total cost
    based on the configured model's pricing.  Inspired by open_deep_research's
    cost transparency feature.

    Usage:
        tracker = CostTracker()
        response = model.invoke(prompt)
        tracker.record_call("extraction", response)
        cost_summary = tracker.get_summary()  # dict for report footer
    """

    def __init__(self):
        self.calls: dict[str, int] = {
            "planning": 0,
            "search": 0,
            "extraction": 0,
            "consolidation": 0,
            "curation": 0,
            "report": 0,
        }
        self.tokens_input: int = 0
        self.tokens_output: int = 0
        self.model_name: str = ""
        self._task_history: list[dict] = []

    def record_call(
        self, task: str, model_name: str, input_tokens: int, output_tokens: int
    ) -> None:
        """Record a single LLM API call."""
        self.calls[task] = self.calls.get(task, 0) + 1
        self.tokens_input += input_tokens
        self.tokens_output += output_tokens
        self.model_name = model_name
        self._task_history.append(
            {
                "task": task,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )

    def get_total_calls(self) -> int:
        """Total number of API calls."""
        return sum(self.calls.values())

    def get_estimated_cost(self) -> float:
        """Estimate total cost in USD based on token counts and model pricing."""
        if not self.model_name:
            return 0.0
        # Normalize model name for lookup
        pricing = _MODEL_PRICING.get(self.model_name)
        if not pricing:
            # Try partial match (e.g., "gpt-4o" matches "gpt-4o-mini" prefix)
            for key, price in _MODEL_PRICING.items():
                if self.model_name.startswith(key) or key.startswith(self.model_name):
                    pricing = price
                    break
        if not pricing:
            # Fallback: assume $0 for unknown models (local/self-hosted)
            return 0.0
        input_price_per_m, output_price_per_m = pricing
        cost = (self.tokens_input / 1_000_000) * input_price_per_m + (
            self.tokens_output / 1_000_000
        ) * output_price_per_m
        return round(cost, 4)

    def get_summary(self) -> dict:
        """Return a summary dict for the report footer."""
        return {
            "total_calls": self.get_total_calls(),
            "input_tokens": self.tokens_input,
            "output_tokens": self.tokens_output,
            "estimated_cost": self.get_estimated_cost(),
            "model": self.model_name or "unknown",
            "per_task": dict(self.calls),
        }

    def format_footer(self) -> str:
        """Format a human-readable cost footer for the report."""
        s = self.get_summary()
        lines = [
            "\n\n--- Cost Summary ---",
            f"  Model: {s['model']}",
            f"  API calls: {s['total_calls']}",
            f"  Input tokens: {s['input_tokens']:,}",
            f"  Output tokens: {s['output_tokens']:,}",
            f"  Estimated cost: ${s['estimated_cost']:.4f}",
            "",
        ]
        if s["per_task"]:
            lines.append("  Per-task breakdown:")
            for task, count in s["per_task"].items():
                if count > 0:
                    lines.append(f"    {task}: {count} call(s)")
        return "\n".join(lines)


# ── Active tracker ─────────────────────────────────────────────────
# One active tracker per process. build_app() registers a fresh tracker
# for each app it creates; invoke_with_cost() records against the active
# one (or no-ops if no app has been built yet).
_active: CostTracker | None = None


def set_active_tracker(tracker: CostTracker | None) -> None:
    """Register the tracker that invoke_with_cost() should record against."""
    global _active
    _active = tracker


def get_active_tracker() -> CostTracker | None:
    """Return the active tracker, or None if no app has been built."""
    return _active
