"""Embedding utilities with token-aware truncation and persistent caching.

Supports multiple providers (OpenAI, Anthropic via Voyage AI) with a unified interface.
Automatically caches embeddings to avoid re-computing identical texts.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Cache root — created lazily at first use
_CACHE_ROOT = Path(__file__).parent / "cache" / "embeddings_cache"


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


def _text_hash(text: str) -> str:
    """Create stable hash of text.

    Hash is computed on truncated text, so different token budgets
    produce different cache entries for the same raw input.
    """
    return hashlib.sha256(text.encode()).hexdigest()


def _cache_path(provider: str, token_budget: int) -> Path:
    """Return path to the cache JSON file for this provider and token budget."""
    return _CACHE_ROOT / provider / str(token_budget) / "embeddings.json"


def _load_cache(provider: str, token_budget: int) -> dict:
    """Load existing cache or create empty cache structure.

    Args:
        provider: "openai" or "anthropic"
        token_budget: Token budget for this cache

    Returns:
        Dictionary mapping text hash → embedding vector (list of floats)
    """
    cache_file = _cache_path(provider, token_budget)

    if cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
        return data.get("embeddings", {})

    return {}


def _save_cache(provider: str, token_budget: int, cache: dict) -> None:
    """Save embeddings cache to disk.

    Args:
        provider: "openai" or "anthropic"
        token_budget: Token budget for this cache
        cache: Dictionary mapping text hash → embedding vector
    """
    cache_file = _cache_path(provider, token_budget)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "token_budget": token_budget,
        "embeddings": cache,
    }

    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)


def embed_text(
    text: str,
    provider: str,
    embedding_model: str,
    token_budget: int,
) -> list[float]:
    """Embed text using specified provider, with caching.

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
    text_hash = _text_hash(truncated)

    # Check cache
    cache = _load_cache(provider, token_budget)
    if text_hash in cache:
        return cache[text_hash]

    # Fetch embedding
    if provider == "openai":
        embedding = _embed_openai(truncated, embedding_model)
    elif provider == "anthropic":
        embedding = _embed_anthropic(truncated, embedding_model)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

    # Cache and return
    cache[text_hash] = embedding
    _save_cache(provider, token_budget, cache)
    return embedding


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
    question_arr = np.array(question_emb).reshape(1, -1)
    para_arr = np.array(para_embs)
    similarities = cosine_similarity(question_arr, para_arr)[0]

    # Get top-k
    top_indices = np.argsort(similarities)[::-1][:k]
    top_paragraphs = [paragraphs[i] for i in top_indices]
    top_scores = [float(similarities[i]) for i in top_indices]

    return top_paragraphs, top_scores
