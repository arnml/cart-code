"""Baseline 2: Always-Think (chain-of-thought, no retrieval) — Anthropic Claude version."""

from anthropic import Anthropic

client = Anthropic()


def always_think(question: str, model: str = "claude-haiku-4-5") -> dict:
    """
    Pure reasoning baseline: answer using chain-of-thought without retrieval.

    Args:
        question: The question to answer
        model: Claude model to use (e.g., "claude-3-5-haiku-20241022")

    Returns:
        Dict with method, answer, token counts, etc.
    """
    message = client.messages.create(
        model=model,
        max_tokens=300,
        system="You are a helpful assistant. Answer questions using step-by-step reasoning.",
        messages=[
            {
                "role": "user",
                "content": f"""Question: {question}

Think step by step, then provide a concise answer.

Format your response as:
Reasoning: [your reasoning steps]
Answer: [the final answer]"""
            }
        ]
    )

    content = message.content[0].text
    # Extract answer after "Answer:" label
    answer = content.split("Answer:")[-1].strip() if "Answer:" in content else content.strip()

    return {
        "method": "always_think",
        "answer": answer,
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "total_tokens": message.usage.input_tokens + message.usage.output_tokens,
        "llm_calls": 1
    }
