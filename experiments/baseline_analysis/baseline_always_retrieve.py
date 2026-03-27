"""Baseline 1: Always-Retrieve (fixed top-k documents)."""

import numpy as np
from openai import OpenAI

client = OpenAI()


def get_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    """Get embedding for text using OpenAI API."""
    # Truncate to 8000 chars to avoid token limits
    text = text[:8000]
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding


def retrieve_top_k(question: str, paragraphs: list[str], k: int = 5) -> tuple:
    """
    Retrieve top-k paragraphs by similarity to question.

    Args:
        question: The question
        paragraphs: List of paragraph strings
        k: Number of documents to retrieve

    Returns:
        (top_k_paragraphs, similarity_scores)
    """
    # Get question embedding
    q_emb = np.array(get_embedding(question)).reshape(1, -1)

    # Get embeddings for all paragraphs
    p_embs = []
    for p in paragraphs:
        p_embs.append(get_embedding(p))
    p_embs = np.array(p_embs)

    # Calculate cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(q_emb, p_embs)[0]

    # Get top-k indices
    top_idx = np.argsort(sims)[::-1][:k]
    top_paragraphs = [paragraphs[i] for i in top_idx]
    top_sims = sims[top_idx].tolist()

    return top_paragraphs, top_sims


def always_retrieve(question: str, paragraphs: list[str], k: int = 5) -> dict:
    """
    RAG baseline: retrieve top-k documents, then answer.

    Args:
        question: The question
        paragraphs: All available paragraphs
        k: Number of documents to retrieve (default 5)

    Returns:
        Dict with method, answer, token counts, etc.
    """
    docs, scores = retrieve_top_k(question, paragraphs, k=k)
    context = '\n\n'.join(docs)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the question based on the provided context."
        },
        {
            "role": "user",
            "content": f"""Context:
{context}

Question: {question}

Answer:"""
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
        max_tokens=100
    )

    answer = response.choices[0].message.content.strip()

    return {
        "method": f"always_retrieve_k{k}",
        "answer": answer,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "llm_calls": 1,
        "docs_retrieved": k,
        "avg_similarity": round(sum(scores) / len(scores), 4)
    }
