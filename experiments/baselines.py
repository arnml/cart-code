"""Shared helpers for HotpotQA baseline evaluation.

Includes:
- flatten_context / build_always_think_prompt / build_retrieval_prompt:
  Preprocessing and prompt templates
- call_llm: Provider routing and model invocation
"""

from typing import Any

from experiments.preprocessing import (
    build_always_think_prompt as _build_always_think_prompt,
)
from experiments.preprocessing import (
    build_retrieval_prompt as _build_retrieval_prompt,
)
from experiments.preprocessing import (
    flatten_context as _flatten_context,
)


def flatten_context(context: dict[str, Any]) -> list[str]:
    """Backward-compatible wrapper for preprocessing.flatten_context."""
    return _flatten_context(context)


def build_always_think_prompt(question: str) -> str:
    """Backward-compatible wrapper for preprocessing.build_always_think_prompt."""
    return _build_always_think_prompt(question)


def build_retrieval_prompt(question: str, paragraphs: list[str]) -> str:
    """Backward-compatible wrapper for preprocessing.build_retrieval_prompt."""
    return _build_retrieval_prompt(question, paragraphs)


def call_llm(prompt: str, model: str) -> tuple[str, int, int, float]:
    """Call LLM and return answer + token counts + cost.

    Routes to OpenAI or Anthropic based on model configuration.

    Args:
        prompt: Full prompt to send to LLM
        model: Model name (e.g., "gpt-4o-mini", "claude-sonnet-4-6")

    Returns:
        Tuple of (answer, input_tokens, output_tokens, cost_usd)

    Raises:
        KeyError: If model not recognized
        Exception: If API call fails (auth, rate limit, etc.)
    """
    from experiments.baselines_config import LLM_CONFIG

    if model not in LLM_CONFIG:
        raise KeyError(f"Unknown model: {model}")

    config = LLM_CONFIG[model]
    provider = config["provider"]

    if provider == "openai":
        from experiments.llm_openai import call_openai
        result = call_openai(prompt, model)
    elif provider == "anthropic":
        from experiments.llm_anthropic import call_anthropic
        result = call_anthropic(prompt, model)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return (
        result["answer"],
        result["input_tokens"],
        result["output_tokens"],
        result["cost_usd"],
    )
