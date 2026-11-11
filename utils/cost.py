"""
LLM cost tracking — per-session token/cost accounting.

The tracker lives here (not in config or graph) so that low-level helpers
like utils/llm.py can record usage without importing the graph layer.
build_app() creates one tracker per app and registers it as the active
tracker; invoke_with_cost() records against the active one, so separate
apps (e.g. tests) never share cost state.

Each call is recorded with the model it was served by, so a session that
mixes models (e.g. a local report-writing process that resumed work started
by an Anthropic loop) is priced per model and the footer shows the mix
instead of silently attributing everything to the last model seen.
"""

from __future__ import annotations


# Pricing per million tokens (USD) — updated for common models.
# Source: https://openai.com/api/pricing/ , https://docs.anthropic.com/en/pricing
# Local models are priced at $0 (free/self-hosted).
# Models not listed here are counted as $0 and flagged in the footer
# (see CostTracker.format_footer) so unknown pricing is visible, not silent.
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
    "claude-sonnet-5-5": (3.00, 15.00),
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-opus-4-5-20251101": (15.00, 75.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    # Local/placeholder
    "qwen36_35": (0.00, 0.00),
    "qwen38_27": (0.00, 0.00),
}


class CostTracker:
    """Track API usage and estimated costs across a research session.

    Records per-task call counts and per-model token usage, then estimates
    total cost by pricing each model's tokens at its own rate. Inspired by
    open_deep_research's cost transparency feature.

    Usage:
        tracker = CostTracker()
        response = model.invoke(prompt)
        tracker.record_call("extraction", "gpt-4o", 1000, 100)
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
        # Last model recorded — kept for backward compatibility with
        # summary consumers that expect a single "model" field.
        self.model_name: str = ""
        self._task_history: list[dict] = []
        # Per-model aggregates: model -> {"calls", "input_tokens",
        # "output_tokens"}. The footer and cost estimate are computed from
        # this, so mixed-model sessions are priced correctly.
        self._models: dict[str, dict[str, int]] = {}
        # Calls whose responses carried no usage metadata — recorded so the
        # footer can say "N calls not counted" instead of silently
        # under-reporting.
        self.unrecorded_calls: int = 0

    def record_call(
        self, task: str, model_name: str, input_tokens: int, output_tokens: int
    ) -> None:
        """Record a single LLM API call."""
        self.calls[task] = self.calls.get(task, 0) + 1
        self.tokens_input += input_tokens
        self.tokens_output += output_tokens
        self.model_name = model_name
        per_model = self._models.setdefault(
            model_name,
            {"calls": 0, "input_tokens": 0, "output_tokens": 0},
        )
        per_model["calls"] += 1
        per_model["input_tokens"] += input_tokens
        per_model["output_tokens"] += output_tokens
        self._task_history.append(
            {
                "task": task,
                "model": model_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )

    def note_unrecorded(self) -> None:
        """Count a call that had no usage metadata (not token-counted)."""
        self.unrecorded_calls += 1

    def _pricing_for(self, model_name: str) -> tuple[float, float] | None:
        """Look up pricing for a model (exact, then prefix match)."""
        pricing = _MODEL_PRICING.get(model_name)
        if pricing:
            return pricing
        # Try partial match (e.g., "claude-sonnet-5-5" matches a
        # "claude-sonnet-5" entry, or vice versa)
        for key, price in _MODEL_PRICING.items():
            if model_name.startswith(key) or key.startswith(model_name):
                return price
        return None

    def _cost_for_model(self, model_name: str, tokens: dict[str, int]) -> float:
        pricing = self._pricing_for(model_name)
        if not pricing:
            return 0.0
        input_price_per_m, output_price_per_m = pricing
        cost = (
            tokens["input_tokens"] / 1_000_000 * input_price_per_m
            + tokens["output_tokens"] / 1_000_000 * output_price_per_m
        )
        return round(cost, 4)

    def get_total_calls(self) -> int:
        """Total number of API calls with usage metadata."""
        return sum(self.calls.values())

    def get_estimated_cost(self) -> float:
        """Estimate total cost in USD, priced per model."""
        return round(
            sum(
                self._cost_for_model(model, tokens)
                for model, tokens in self._models.items()
            ),
            4,
        )

    def get_summary(self) -> dict:
        """Return a summary dict for the report footer / console output."""
        return {
            "total_calls": self.get_total_calls(),
            "input_tokens": self.tokens_input,
            "output_tokens": self.tokens_output,
            "estimated_cost": self.get_estimated_cost(),
            "model": self.model_name or "unknown",
            "per_task": dict(self.calls),
            "models": {
                model: {
                    **tokens,
                    "cost": self._cost_for_model(model, tokens),
                    "priced": self._pricing_for(model) is not None,
                }
                for model, tokens in self._models.items()
            },
            "unrecorded_calls": self.unrecorded_calls,
        }

    def format_footer(self, label: str | None = None) -> str:
        """Format a human-readable cost footer for the report.

        label (optional): free-form context line, e.g. "Plan: plan_abc123",
        so the footer identifies which plan/session the numbers belong to.
        """
        s = self.get_summary()
        lines = ["\n\n--- Cost Summary ---"]
        if label:
            lines.append(f"  {label}")
        models = s["models"]
        if len(models) <= 1:
            # Single-model session: one line, same look as before
            for model, m in models.items():
                price_note = "" if m["priced"] else " (not in pricing table)"
                lines.append(f"  Model: {model}{price_note}")
        else:
            lines.append("  Models (per-model pricing):")
            for model, m in models.items():
                price_note = "" if m["priced"] else " — not in pricing table, $0"
                lines.append(
                    f"    {model}: {m['calls']} call(s), "
                    f"{m['input_tokens']:,} in / {m['output_tokens']:,} out "
                    f"tokens, ${m['cost']:.4f}{price_note}"
                )
        lines += [
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
        if s["unrecorded_calls"]:
            lines.append(
                f"  Note: {s['unrecorded_calls']} call(s) had no usage metadata "
                "and are not counted above."
            )
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
