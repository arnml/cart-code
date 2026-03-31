"""LLM call helpers for generation and chain-of-thought thinking."""

from openai import OpenAI

client = OpenAI()

_GENERATE_SYSTEM = "Answer based on context. 1-5 words only."
_THINK_SYSTEM = "Answer concisely. 1-5 words."


def generate(question: str, context: str, model: str) -> tuple[str, int, int]:
    """Generate an answer grounded in retrieved context."""
    r = client.chat.completions.create(
        model=model,
        temperature=0,
        max_completion_tokens=50,
        messages=[
            {"role": "system", "content": _GENERATE_SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
        ],
    )
    msg = r.choices[0].message.content.strip()
    return msg, r.usage.prompt_tokens, r.usage.completion_tokens


def think(question: str, model: str) -> tuple[str, int, int]:
    """Answer directly from parametric knowledge (no retrieval)."""
    r = client.chat.completions.create(
        model=model,
        temperature=0,
        max_completion_tokens=50,
        messages=[
            {"role": "system", "content": _THINK_SYSTEM},
            {"role": "user", "content": f"Question: {question}\nAnswer:"},
        ],
    )
    msg = r.choices[0].message.content.strip()
    return msg, r.usage.prompt_tokens, r.usage.completion_tokens
