"""Baseline: Always-Retrieve (fixed top-k documents)."""

from .embedding_utils import retrieve_top_k
from .eval_utils import extract_answer
from openai import OpenAI

client = OpenAI()

_PROMPT = (
    "Think step by step, then provide a concise answer.\n\n"
    "Format your response as:\n"
    "Reasoning: [your reasoning steps]\n"
    "Answer: [the final answer]"
)


def always_retrieve(
    question: str, paragraphs: list[str], k: int = 5, model: str = "gpt-4o-mini"
) -> dict:
    """RAG baseline: retrieve top-k documents then answer."""
    docs, scores = retrieve_top_k(question, paragraphs, k=k)
    context = "\n\n".join(docs)

    _system = "You are a helpful assistant. Answer the question based on the provided context."
    messages = [
        {"role": "system", "content": _system},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\n\n{_PROMPT}",
        },
    ]

    kwargs: dict = {"model": model, "messages": messages, "temperature": 0}
    # gpt-5+ uses max_completion_tokens instead of max_tokens
    if model.startswith("gpt-5"):
        kwargs["max_completion_tokens"] = 300
    else:
        kwargs["max_tokens"] = 300

    response = client.chat.completions.create(**kwargs)

    return {
        "method": f"always_retrieve_k{k}",
        "answer": extract_answer(response.choices[0].message.content),
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "llm_calls": 1,
        "docs_retrieved": k,
        "avg_similarity": round(sum(scores) / len(scores), 4),
    }
