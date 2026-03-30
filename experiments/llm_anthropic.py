"""Anthropic client for LLM calls with token & cost tracking.

Supports: claude-haiku-4-5, claude-sonnet-4-6
Requires: ANTHROPIC_API_KEY environment variable
"""

from anthropic import Anthropic
from experiments.baselines_config import LLM_CONFIG


def call_anthropic(prompt: str, model: str) -> dict:
    """Call Anthropic model and return answer + tokens + cost.

    Args:
        prompt: Full prompt text
        model: Model name (e.g., "claude-sonnet-4-6")

    Returns:
        {
            "answer": str,
            "input_tokens": int,
            "output_tokens": int,
            "cost_usd": float
        }

    Raises:
        KeyError: If model not in LLM_CONFIG
        Exception: If API call fails (auth, rate limit, etc.)
    """
    if model not in LLM_CONFIG:
        raise KeyError(f"Model {model} not in LLM_CONFIG")

    client = Anthropic()

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cache_read_tokens = getattr(response.usage, "cache_read_input_tokens", 0)

    # Calculate cost using pricing from config
    config = LLM_CONFIG[model]
    cost = (
        (input_tokens * config["input_price_per_1m"] / 1_000_000) +
        (output_tokens * config["output_price_per_1m"] / 1_000_000) +
        (cache_read_tokens * config.get("cache_read_price_per_1m", 0) / 1_000_000)
    )

    return {
        "answer": answer,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
    }
