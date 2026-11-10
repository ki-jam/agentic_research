"""
Shared LLM response helpers — JSON extraction from model output.
"""

from __future__ import annotations

import json
from typing import Any

from utils.cost import get_active_tracker


def extract_json(text: str) -> str:
    """Extract the JSON payload from LLM text that may contain markdown
    fences or surrounding prose.

    Considers both array and object containers and picks the earliest
    opening bracket whose matching closing bracket is the last one in
    the text (handles both `[...]` and `{"...": [...]}` shapes).
    """
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]).strip()

    candidates = []
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            candidates.append((start, text[start : end + 1]))
    if candidates:
        candidates.sort(key=lambda c: c[0])
        return candidates[0][1]
    return text


def parse_json(text: str) -> Any:
    """extract_json + json.loads in one step. Raises on invalid JSON."""
    return json.loads(extract_json(text))


def invoke_with_cost(model, prompt, task: str = "unknown") -> Any:
    """Invoke an LLM model and record cost metrics.

    Wraps model.invoke() to automatically track token usage and API calls.
    Inspired by open_deep_research's cost transparency feature.

    Args:
        model: LangChain chat model instance.
        prompt: Prompt string to send to the model.
        task: Task name for cost tracking (e.g., "planning", "extraction", "report").

    Returns:
        The model's response object.
    """
    response = model.invoke(prompt)

    # Extract token counts from response metadata
    input_tokens = 0
    output_tokens = 0
    model_name = getattr(model, "model_name", getattr(model, "model", "unknown"))

    if hasattr(response, "usage_metadata"):
        meta = response.usage_metadata
        input_tokens = meta.get("input_tokens", 0)
        output_tokens = meta.get("output_tokens", 0)
    elif hasattr(response, "additional_kwargs"):
        extra = response.additional_kwargs
        if "usage" in extra:
            usage = extra["usage"]
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

    if input_tokens or output_tokens:
        tracker = get_active_tracker()
        if tracker is not None:
            tracker.record_call(task, model_name, input_tokens, output_tokens)

    return response
