"""Anthropic Claude call helpers for generation and chain-of-thought thinking."""

import os
from anthropic import Anthropic
from anthropic.types import TextBlock

# The client will automatically use the ANTHROPIC_API_KEY environment variable.
client = Anthropic()

_GENERATE_SYSTEM = "Answer based on context. 1-5 words only."
_THINK_SYSTEM = "Answer concisely. 1-5 words."


def generate(question: str, context: str, model: str) -> tuple[str, int, int]:
    """Generate an answer grounded in retrieved context using Claude."""
    message = client.messages.create(
        model=model,
        max_tokens=50,
        temperature=0,
        system=_GENERATE_SYSTEM,
        messages=[
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
        ],
    )
    msg = message.content[0].text if isinstance(message.content[0], TextBlock) else ""
    return msg.strip(), message.usage.input_tokens, message.usage.output_tokens


def think(question: str, model: str) -> tuple[str, int, int]:
    """Answer directly from parametric knowledge (no retrieval) using Claude."""
    message = client.messages.create(
        model=model,
        max_tokens=50,
        temperature=0,
        system=_THINK_SYSTEM,
        messages=[
            {"role": "user", "content": f"Question: {question}\nAnswer:"},
        ],
    )
    msg = message.content[0].text if isinstance(message.content[0], TextBlock) else ""
    return msg.strip(), message.usage.input_tokens, message.usage.output_tokens
