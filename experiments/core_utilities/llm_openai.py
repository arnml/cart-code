"""OpenAI client for LLM calls with token & cost tracking.

Supports: gpt-4o-mini, gpt-5.4-mini
Requires: OPENAI_API_KEY environment variable
"""

from openai import OpenAI
from experiments.core_utilities.baselines_config import LLM_CONFIG


def call_openai(prompt: str, model: str) -> dict:
    """Call OpenAI model and return answer + tokens + cost.

    Args:
        prompt: Full prompt text
        model: Model name (e.g., "gpt-4o-mini")

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

    client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.choices[0].message.content
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    # Calculate cost using pricing from config
    config = LLM_CONFIG[model]
    cost = (
        (input_tokens * config["input_price_per_1m"] / 1_000_000) +
        (output_tokens * config["output_price_per_1m"] / 1_000_000)
    )

    return {
        "answer": answer,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
    }
