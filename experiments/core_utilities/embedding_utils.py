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
_CACHE_ROOT = Path(__file__).parent / "cache"


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


def _cache_path(provider: str, model: str, token_budget: int) -> Path:
    """Return path to the cache JSON file for this config."""
    return _CACHE_ROOT / provider / model / str(token_budget) / "embeddings.json"


def _load_cache(provider: str, model: str, token_budget: int) -> dict:
    """Load existing cache or create empty cache structure.

    Args:
        provider: "openai" or "anthropic"
        model: Model name
        token_budget: Token budget for this cache

    Cache format:
    {
        "metadata": {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "token_budget": 8000,
            "updated_at": "2026-03-29T12:00:00"
        },
        "entries": {
            "<sha256_of_truncated_text>": {
                "embedding": [...],
                "tokens_used": 245,
                "text_preview": "first 80 chars of original..."
            }
        }
    }
    """
    path = _cache_path(provider, model, token_budget)

    if path.exists():
        with open(path) as f:
            return json.load(f)

    return {
        "metadata": {
            "provider": provider,
            "model": model,
            "token_budget": token_budget,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        "entries": {},
    }


def _save_cache(cache: dict, provider: str, model: str, token_budget: int) -> None:
    """Write cache to disk, creating directories as needed.

    Args:
        cache: Cache dict to save
        provider: "openai" or "anthropic"
        model: Model name
        token_budget: Token budget for this cache
    """
    path = _cache_path(provider, model, token_budget)
    path.parent.mkdir(parents=True, exist_ok=True)

    cache["metadata"]["updated_at"] = datetime.now(timezone.utc).isoformat()

    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def _fetch_embeddings_openai(texts: list[str], model: str) -> list[list[float]]:
    """Fetch embeddings from OpenAI."""
    from openai import OpenAI

    client = OpenAI()
    response = client.embeddings.create(model=model, input=texts)
    # Sort by index to guarantee order matches input
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def _fetch_embeddings_voyage(texts: list[str], model: str) -> list[list[float]]:
    """Fetch embeddings from Voyage AI (Anthropic's embedding provider)."""
    import voyageai

    client = voyageai.Client()
    response = client.embed(texts, model=model)
    return response.embeddings


def _fetch_embeddings(
    texts: list[str], provider: str, model: str
) -> list[list[float]]:
    """Route to the appropriate embedding provider."""
    if provider.lower() == "openai":
        return _fetch_embeddings_openai(texts, model)
    elif provider.lower() == "anthropic":
        return _fetch_embeddings_voyage(texts, model)
    else:
        raise ValueError(
            f"Unknown provider: {provider!r}. Must be 'openai' or 'anthropic'."
        )


def get_embeddings(
    texts: list[str],
    provider: str,
    model: str,
    token_budget: int = 8000,
) -> list[list[float]]:
    """Get embeddings for texts, using cache when available.

    Queries the cache first, fetches only new texts, then updates cache.

    Args:
        texts: List of strings to embed (may contain empty/whitespace)
        provider: "openai" or "anthropic"
        model: Model name (e.g., "text-embedding-3-small", "voyage-3")
        token_budget: Max tokens per text (default 8000)

    Returns:
        List of embeddings (one per input text). Empty/whitespace texts
        return empty embeddings [].

    Raises:
        ValueError: If token_budget <= 0 or provider not recognized
    """
    # Edge case: empty input
    if not texts:
        return []

    # Normalize: strip whitespace, but remember originals for cache key
    stripped = [t.strip() for t in texts]

    # Edge case: all whitespace
    if not any(stripped):
        return [[] for _ in texts]

    # Token-truncate all texts
    truncated = [
        _truncate(t, model, token_budget) for t in stripped
    ]

    # Load cache
    cache = _load_cache(provider, model, token_budget)

    # Track which texts need fetching
    results = [None] * len(texts)
    miss_indices = []  # (index, original_text, truncated_text, hash)

    for i, (orig, trunc) in enumerate(zip(stripped, truncated)):
        key = _text_hash(trunc)
        if key in cache["entries"]:
            results[i] = cache["entries"][key]["embedding"]
        else:
            miss_indices.append((i, orig, trunc, key))

    # Fetch only cache misses in one batched call
    if miss_indices:
        miss_texts = [trunc for _, _, trunc, _ in miss_indices]
        new_embeddings = _fetch_embeddings(miss_texts, provider, model)

        # Update cache with new embeddings
        enc = _get_encoding(model)
        for (idx, orig, trunc, key), emb in zip(miss_indices, new_embeddings):
            results[idx] = emb
            cache["entries"][key] = {
                "embedding": emb,
                "tokens_used": len(enc.encode(trunc)),
                "text_preview": orig[:80],
            }

        _save_cache(cache, provider, model, token_budget)

    return results


def retrieve_top_k(
    question: str,
    paragraphs: list[str],
    k: int,
    provider: str,
    model: str,
    token_budget: int = 8000,
) -> tuple[list[str], list[float]]:
    """Retrieve top-k paragraphs by cosine similarity to question.

    Embeds question + paragraphs in one batched call (respecting cache).

    Args:
        question: Query text
        paragraphs: List of candidate paragraphs
        k: Number of top results to return
        provider: "openai" or "anthropic"
        model: Model name
        token_budget: Max tokens per text

    Returns:
        (top_k_paragraphs, similarity_scores) where both are lists of length min(k, len(paragraphs))

    Raises:
        ValueError: If k <= 0
    """
    # Edge case: k must be positive
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")

    # Edge case: no paragraphs
    if not paragraphs:
        return [], []

    # Fetch embeddings for question + all paragraphs (cached)
    all_texts = [question, *paragraphs]
    embeddings = get_embeddings(all_texts, provider, model, token_budget)

    # Handle edge case: all-whitespace texts
    if not embeddings[0] or not embeddings[1:]:
        return [], []

    q_emb = np.array(embeddings[0]).reshape(1, -1)
    p_embs = np.array(embeddings[1:])

    # Compute cosine similarity
    sims = cosine_similarity(q_emb, p_embs)[0]

    # Return top-k (clamped to available results)
    k_actual = min(k, len(paragraphs))
    top_idx = np.argsort(sims)[::-1][:k_actual]

    return [paragraphs[i] for i in top_idx], sims[top_idx].tolist()


if __name__ == "__main__":
    """Test embeddings for both providers with different token budgets."""

    # Use a longer text so truncation at budget 100 actually happens
    test_text = (
        "The quick brown fox jumps over the lazy dog. "
        "This is a sample text used to test the embedding utilities. "
        "We are testing token-aware truncation and caching functionality. "
        "The embedding system supports multiple providers including OpenAI and Anthropic. "
        "When using Anthropic, embeddings are powered by Voyage AI, which provides "
        "high-quality embedding models optimized for various use cases. "
        "Token-aware truncation ensures that we respect model limits while preserving "
        "as much information as possible from the original text. "
        "Caching prevents redundant API calls, improving both speed and cost efficiency. "
        "The cache is organized by provider, model, and token budget for flexibility. "
        "Different token budgets create separate cache entries, allowing you to experiment "
        "with different truncation strategies without invalidating cached results."
    )

    print("=" * 70)
    print("EMBEDDING UTILITIES TEST")
    print("=" * 70)

    # Test OpenAI with two different token budgets
    print("\n[OpenAI] text-embedding-3-small")
    print("-" * 70)

    print("\n1. Token budget: 8000 (default)")
    embs_openai_8000 = get_embeddings(
        [test_text],
        provider="openai",
        model="text-embedding-3-small",
        token_budget=8000,
    )
    print(f"   Embedding dim: {len(embs_openai_8000[0])}")
    print(f"   First 5 values: {embs_openai_8000[0][:5]}")

    print("\n2. Token budget: 100 (truncated)")
    embs_openai_100 = get_embeddings(
        [test_text],
        provider="openai",
        model="text-embedding-3-small",
        token_budget=100,
    )
    print(f"   Embedding dim: {len(embs_openai_100[0])}")
    print(f"   First 5 values: {embs_openai_100[0][:5]}")

    # Check if embeddings differ (should due to truncation)
    diff = sum(abs(a - b) for a, b in zip(embs_openai_8000[0], embs_openai_100[0]))
    print(f"   L1 difference: {diff:.4f} (should be > 0 if text was truncated)")

    # Test Anthropic (Voyage AI) with two different token budgets
    print("\n[Anthropic] voyage-3")
    print("-" * 70)
    print("Note: Anthropic embeddings use Voyage AI")
    print("Set VOYAGE_API_KEY from https://dash.voyageai.com to enable\n")

    try:
        print("1. Token budget: 8000 (default)")
        embs_voyage_8000 = get_embeddings(
            [test_text],
            provider="anthropic",
            model="voyage-3",
            token_budget=8000,
        )
        print(f"   Embedding dim: {len(embs_voyage_8000[0])}")
        print(f"   First 5 values: {embs_voyage_8000[0][:5]}")

        print("\n2. Token budget: 100 (truncated)")
        embs_voyage_100 = get_embeddings(
            [test_text],
            provider="anthropic",
            model="voyage-3",
            token_budget=100,
        )
        print(f"   Embedding dim: {len(embs_voyage_100[0])}")
        print(f"   First 5 values: {embs_voyage_100[0][:5]}")

        diff = sum(abs(a - b) for a, b in zip(embs_voyage_8000[0], embs_voyage_100[0]))
        print(f"   L1 difference: {diff:.4f} (should be > 0 if text was truncated)")
    except Exception as e:
        if "No API key" in str(e) or "VOYAGE_API_KEY" in str(e):
            print("   VOYAGE_API_KEY not set")
            print("   Set: export VOYAGE_API_KEY=<your-key>")
        else:
            print(f"   Error: {e}")

    # Test cache efficiency
    print("\n" + "=" * 70)
    print("CACHE TEST")
    print("=" * 70)

    print("\nFetching same text again (should hit cache)...")
    embs_cached = get_embeddings(
        [test_text],
        provider="openai",
        model="text-embedding-3-small",
        token_budget=8000,
    )
    print(f"Cache hit: {embs_cached[0] == embs_openai_8000[0]}")

    # Show cache directory structure
    cache_dir = _CACHE_ROOT
    if cache_dir.exists():
        print(f"\nCache location: {cache_dir}")
        print("Cache structure:")
        for provider_dir in sorted(cache_dir.iterdir()):
            if provider_dir.is_dir():
                print(f"  {provider_dir.name}/")
                for model_dir in sorted(provider_dir.iterdir()):
                    if model_dir.is_dir():
                        print(f"    {model_dir.name}/")
                        for budget_dir in sorted(model_dir.iterdir()):
                            if budget_dir.is_dir():
                                cache_file = budget_dir / "embeddings.json"
                                if cache_file.exists():
                                    with open(cache_file) as f:
                                        cache_data = json.load(f)
                                        num_entries = len(cache_data.get("entries", {}))
                                    print(
                                        f"      {budget_dir.name}/ ({num_entries} entries)"
                                    )

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
