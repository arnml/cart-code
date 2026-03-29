"""Baseline: Always-Retrieve (fixed top-k documents) — Anthropic Claude version."""

from anthropic import Anthropic
from anthropic.types import TextBlock
from .embedding_utils import retrieve_top_k
from .eval_utils import extract_answer

client = Anthropic()

_PROMPT = (
    "Think step by step, then provide a concise answer.\n\n"
    "Format your response as:\n"
    "Reasoning: [your reasoning steps]\n"
    "Answer: [the final answer]"
)


def always_retrieve(
    question: str, paragraphs: list[str], k: int = 5, model: str = "claude-haiku-4-5"
) -> dict:
    """RAG baseline: retrieve top-k documents then answer."""
    docs, scores = retrieve_top_k(question, paragraphs, k=k)
    context = "\n\n".join(docs)

    message = client.messages.create(
        model=model,
        max_tokens=300,
        system="You are a helpful assistant. Answer the question based on the provided context.",
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}\n\n{_PROMPT}",
            }
        ],
    )

    return {
        "method": f"always_retrieve_k{k}",
        "answer": extract_answer(
            message.content[0].text if isinstance(message.content[0], TextBlock) else ""
        ),
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "total_tokens": message.usage.input_tokens + message.usage.output_tokens,
        "llm_calls": 1,
        "docs_retrieved": k,
        "avg_similarity": round(sum(scores) / len(scores), 4),
    }
