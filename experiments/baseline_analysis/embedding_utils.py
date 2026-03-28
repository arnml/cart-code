"""Shared embedding and retrieval utilities for baseline experiments."""

import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI()


def get_embeddings_batch(
    texts: list[str], model: str = "text-embedding-3-small"
) -> list[list[float]]:
    """Fetch embeddings for all texts in a single API call."""
    truncated = [t[:8000] for t in texts]
    response = client.embeddings.create(model=model, input=truncated)
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def retrieve_top_k(
    question: str, paragraphs: list[str], k: int = 5
) -> tuple[list[str], list[float]]:
    """
    Retrieve top-k paragraphs by cosine similarity to the question.

    Fetches all embeddings (question + paragraphs) in one batched API call.
    """
    embeddings = get_embeddings_batch([question, *paragraphs])
    q_emb = np.array(embeddings[0]).reshape(1, -1)
    p_embs = np.array(embeddings[1:])
    sims = cosine_similarity(q_emb, p_embs)[0]
    top_idx = np.argsort(sims)[::-1][:k]
    return [paragraphs[i] for i in top_idx], sims[top_idx].tolist()
