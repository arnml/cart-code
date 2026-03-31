"""Embedding-based retrieval with adaptive-k and noise gate filtering."""

import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from .similarity import jaccard

client = OpenAI()

_EMBED_MODEL = "text-embedding-3-small"


def get_embeddings_batch(texts: list[str]) -> np.ndarray:
    """Embed a list of texts in a single API call."""
    truncated = [t[:8000] for t in texts]
    r = client.embeddings.create(model=_EMBED_MODEL, input=truncated)
    return np.array([item.embedding for item in r.data])


def get_embedding(text: str) -> list[float]:
    return get_embeddings_batch([text])[0].tolist()


def adaptive_k(scores: list[float], gap_threshold: float = 0.08) -> int:
    """
    Implements Taguchi et al. 2025 (EMNLP) similarity gap formula.
    k* = argmax_i [s_i - s_{i+1}]
    """
    if len(scores) <= 1:
        return len(scores)
    gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
    max_gap = max(gaps)
    return gaps.index(max_gap) + 1 if max_gap > gap_threshold else min(5, len(scores))


def noise_gate(
    docs: list[str],
    scores: list[float],
    sim_thr: float = 0.35,
    jac_thr: float = 0.65,
) -> tuple[list[str], list[float]]:
    """Filters low-similarity and redundant documents. Returns (docs, scores)."""
    out, out_scores, seen = [], [], []
    for doc, score in zip(docs, scores, strict=True):
        if score < sim_thr:
            continue
        if any(jaccard(doc, s) > jac_thr for s in seen):
            continue
        out.append(doc)
        out_scores.append(score)
        seen.append(doc)
    return out, out_scores


def embed_and_rank(
    question: str, paragraphs: list[str], top_n: int = 10
) -> tuple[list[str], list[float]]:
    """Embed question and paragraphs in two API calls, return top-N by cosine similarity."""
    all_embs = get_embeddings_batch([question] + paragraphs)
    q_emb, p_embs = all_embs[:1], all_embs[1:]
    sims = cosine_similarity(q_emb, p_embs)[0]
    top_idx = np.argsort(sims)[::-1][:top_n]
    docs = [paragraphs[i] for i in top_idx]
    scores = sims[top_idx].tolist()
    return docs, scores


def retrieve_and_filter(
    question: str, paragraphs: list[str]
) -> tuple[list[str], list[float]]:
    """Embed question, rank paragraphs, apply adaptive-k and noise gate."""
    docs, scores = embed_and_rank(question, paragraphs)
    k = adaptive_k(scores)
    return noise_gate(docs[: k + 2], scores[: k + 2])
