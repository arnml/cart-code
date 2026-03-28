"""Baseline: Always-Think (chain-of-thought, no retrieval)."""

from eval_utils import extract_answer
from openai import OpenAI

client = OpenAI()

_PROMPT = (
    "Think step by step, then provide a concise answer.\n\n"
    "Format your response as:\n"
    "Reasoning: [your reasoning steps]\n"
    "Answer: [the final answer]"
)


def always_think(question: str, model: str = "gpt-4o-mini") -> dict:
    """Pure reasoning baseline: answer using chain-of-thought without retrieval."""
    _system = "You are a helpful assistant. Answer questions using step-by-step reasoning."
    messages = [
        {"role": "system", "content": _system},
        {"role": "user", "content": f"Question: {question}\n\n{_PROMPT}"},
    ]

    kwargs: dict = {"model": model, "messages": messages, "temperature": 0}
    # gpt-5+ uses max_completion_tokens instead of max_tokens
    if model.startswith("gpt-5"):
        kwargs["max_completion_tokens"] = 300
    else:
        kwargs["max_tokens"] = 300

    response = client.chat.completions.create(**kwargs)

    return {
        "method": "always_think",
        "answer": extract_answer(response.choices[0].message.content),
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "llm_calls": 1,
    }
