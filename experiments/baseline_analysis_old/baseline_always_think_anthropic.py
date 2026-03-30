"""Baseline: Always-Think (chain-of-thought, no retrieval) — Anthropic Claude version."""

from anthropic import Anthropic
from anthropic.types import TextBlock
from .eval_utils import extract_answer

client = Anthropic()

_PROMPT = (
    "Think step by step, then provide a concise answer.\n\n"
    "Format your response as:\n"
    "Reasoning: [your reasoning steps]\n"
    "Answer: [the final answer]"
)


def always_think(question: str, model: str = "claude-haiku-4-5") -> dict:
    """Pure reasoning baseline: answer using chain-of-thought without retrieval."""
    message = client.messages.create(
        model=model,
        max_tokens=300,
        system="You are a helpful assistant. Answer questions using step-by-step reasoning.",
        messages=[{"role": "user", "content": f"Question: {question}\n\n{_PROMPT}"}],
    )

    return {
        "method": "always_think",
        "answer": extract_answer(
            message.content[0].text if isinstance(message.content[0], TextBlock) else ""
        ),
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "total_tokens": message.usage.input_tokens + message.usage.output_tokens,
        "llm_calls": 1,
    }
