"""Embedding utilities with token-aware truncation.

Supports multiple providers (OpenAI, Anthropic via Voyage AI) with a unified interface.
No caching - embeddings are cheap and caching causes race conditions with parallel runs.
"""

import numpy as np


def cosine_similarities(query_embedding: list[float], embeddings: list[list[float]]) -> np.ndarray:
    """Compute cosine similarity between one query embedding and many embeddings."""
    query_arr = np.asarray(query_embedding, dtype=np.float64)
    embedding_arr = np.asarray(embeddings, dtype=np.float64)

    query_norm = np.linalg.norm(query_arr)
    embedding_norms = np.linalg.norm(embedding_arr, axis=1)
    denom = query_norm * embedding_norms

    similarities = np.zeros(len(embedding_arr), dtype=np.float64)
    nonzero = denom > 0
    similarities[nonzero] = embedding_arr[nonzero] @ query_arr / denom[nonzero]
    return similarities


def _get_encoding(model: str):
    """Get tiktoken encoder for the given model.

    For OpenAI models, uses the model-specific encoding.
    For Anthropic/Voyage models, falls back to cl100k_base.

    Args:
        model: Model name for encoding lookup
    """
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Voyage AI and other models use cl100k_base as a reasonable approximation
        return tiktoken.get_encoding("cl100k_base")


def _truncate(text: str, model: str, token_budget: int) -> str:
    """Truncate text to token budget using the model's tokenizer.

    Args:
        text: Raw text to truncate
        model: Model name (e.g., "text-embedding-3-small", "voyage-3")
        token_budget: Maximum tokens to keep

    Returns:
        Truncated text (or original if within budget)
    """
    if token_budget <= 0:
        raise ValueError(f"token_budget must be > 0, got {token_budget}")

    enc = _get_encoding(model)
    tokens = enc.encode(text)

    if len(tokens) <= token_budget:
        return text

    return enc.decode(tokens[:token_budget])




def embed_text(
    text: str,
    provider: str,
    embedding_model: str,
    token_budget: int,
) -> list[float]:
    """Embed text using specified provider.

    Args:
        text: Text to embed
        provider: "openai" or "anthropic"
        embedding_model: Model name (e.g., "text-embedding-3-small", "voyage-3")
        token_budget: Maximum tokens to truncate text to

    Returns:
        Embedding vector (list of floats)

    Raises:
        ValueError: If provider or model invalid
    """
    # Truncate to budget
    truncated = _truncate(text, embedding_model, token_budget)

    # Fetch embedding
    if provider == "openai":
        return _embed_openai(truncated, embedding_model)
    elif provider == "anthropic":
        return _embed_anthropic(truncated, embedding_model)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


def _embed_openai(text: str, model: str) -> list[float]:
    """Embed text using OpenAI API."""
    from openai import OpenAI

    client = OpenAI()
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def _embed_anthropic(text: str, model: str) -> list[float]:
    """Embed text using Anthropic/Voyage API."""
    import requests

    # Use Voyage API (third-party service for embeddings)
    api_key = __import__("os").getenv("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError(
            "VOYAGE_API_KEY not set. Required for Anthropic embedding model: " + model
        )

    url = "https://api.voyageai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {"input": text, "model": model}

    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()

    return response.json()["data"][0]["embedding"]


def retrieve_top_k(
    question: str,
    paragraphs: list[str],
    k: int,
    provider: str,
    embedding_model: str,
    token_budget: int,
) -> tuple[list[str], list[float]]:
    """Retrieve top-k paragraphs most similar to question using embeddings.

    Args:
        question: Query text
        paragraphs: List of paragraph texts to search
        k: Number of top results to return
        provider: "openai" or "anthropic"
        embedding_model: Model name
        token_budget: Max tokens for truncation

    Returns:
        Tuple of (top_k_paragraphs, similarity_scores)
    """
    # Embed question
    question_emb = embed_text(question, provider, embedding_model, token_budget)

    # Embed all paragraphs
    para_embs = [
        embed_text(para, provider, embedding_model, token_budget)
        for para in paragraphs
    ]

    # Compute similarity
    similarities = cosine_similarities(question_emb, para_embs)

    # Get top-k
    top_indices = np.argsort(similarities)[::-1][:k]
    top_paragraphs = [paragraphs[i] for i in top_indices]
    top_scores = [float(similarities[i]) for i in top_indices]

    return top_paragraphs, top_scores
