"""Noise-gate evaluation script for CART paper.

Implements the CART noise-gate ablation: score every paragraph against the
question, then apply similarity and redundancy filtering before generation.

Usage from root:
    uv run python -m experiments.run_noise_gate gpt-4o-mini 2
    uv run python -m experiments.run_noise_gate claude-sonnet-4-6 100
    uv run python -m experiments.run_noise_gate claude-sonnet-4-6 100 20
"""

from __future__ import annotations

import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from experiments.baselines_config import DATASET_CONFIG, LLM_CONFIG, LLM_TO_EMBEDDING, MODELS
from experiments.cache_dataset import load_dataset_cached
from experiments.embedding_utils import embed_text
from experiments.eval_utils import evaluate_sample
from experiments.llm_anthropic import call_anthropic
from experiments.llm_openai import call_openai
from experiments.preprocessing import (
    build_always_think_prompt,
    build_retrieval_prompt,
    flatten_context,
)

RESULTS_DIR = Path(__file__).parent / "results" / "cart"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

DEFAULT_MAX_WORKERS = 20
NOISE_GATE_SIM_THRESHOLDS = (0.2, 0.3, 0.5)
NOISE_GATE_REDUNDANCY_THRESHOLD = 0.65
NOISE_GATE_METHOD = "noise_gate"

ResultDict = dict[str, str | int | float]


def call_llm(prompt: str, model: str) -> tuple[str, int, int, float]:
    """Call the configured provider and return answer + token counts + cost."""
    if model not in LLM_CONFIG:
        raise KeyError(f"Unknown model: {model}")

    provider = LLM_CONFIG[model]["provider"]
    if provider == "openai":
        result = call_openai(prompt, model)
    elif provider == "anthropic":
        result = call_anthropic(prompt, model)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return (
        result["answer"],
        result["input_tokens"],
        result["output_tokens"],
        result["cost_usd"],
    )


def _jaccard(text_a: str, text_b: str) -> float:
    """Compute a simple word-set Jaccard similarity."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _apply_noise_gate(
    docs: list[str],
    scores: list[float],
    sim_threshold: float,
    jac_threshold: float,
) -> list[str]:
    """Filter low-similarity and redundant documents.

    The gate keeps the first document that clears the similarity threshold and
    drops later documents that are too similar to anything already kept.
    """
    filtered_docs: list[str] = []
    seen_docs: list[str] = []

    for doc, score in zip(docs, scores, strict=True):
        if score < sim_threshold:
            continue
        if any(_jaccard(doc, prev_doc) > jac_threshold for prev_doc in seen_docs):
            continue
        filtered_docs.append(doc)
        seen_docs.append(doc)

    return filtered_docs


def noise_gate_select(
    question: str,
    paragraphs: list[str],
    provider: str,
    embedding_model: str,
    token_budget: int,
    sim_threshold: float,
    jac_threshold: float = NOISE_GATE_REDUNDANCY_THRESHOLD,
) -> list[str]:
    """Select paragraphs using cosine similarity and noise-gate filtering.

    This scores every paragraph against the question, ranks all paragraphs by
    similarity, then keeps only those above the similarity threshold while
    dropping highly redundant context.
    """
    if not paragraphs:
        return []

    question_embedding = embed_text(question, provider, embedding_model, token_budget)
    paragraph_embeddings = [
        embed_text(paragraph, provider, embedding_model, token_budget)
        for paragraph in paragraphs
    ]

    question_arr = np.array(question_embedding).reshape(1, -1)
    paragraph_arr = np.array(paragraph_embeddings)
    similarities = cosine_similarity(question_arr, paragraph_arr)[0]

    ranked_indices = np.argsort(similarities)[::-1]
    ranked_docs = [paragraphs[i] for i in ranked_indices]
    ranked_scores = [float(similarities[i]) for i in ranked_indices]
    return _apply_noise_gate(ranked_docs, ranked_scores, sim_threshold, jac_threshold)


def _build_result(
    sample: dict[str, Any],
    sim_threshold: float,
    pred: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> ResultDict:
    """Build a result row and attach evaluation metrics."""
    result: ResultDict = {
        "question_id": sample["id"],
        "method": NOISE_GATE_METHOD,
        "threshold": sim_threshold,
        "answer_pred": pred,
        "answer_gt": sample["answer"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
    }

    metrics = evaluate_sample(pred, sample["answer"])
    result.update(
        {
            "em": metrics.em,
            "f1": metrics.f1,
            "precision": metrics.precision,
            "recall": metrics.recall,
        }
    )
    return result


def save_csv(model: str, results: list[ResultDict]) -> None:
    """Save per-record results to a CSV file."""
    csv_path = RESULTS_DIR / f"results_noise_gate_{model}.csv"

    fieldnames = [
        "question_id",
        "method",
        "threshold",
        "answer_pred",
        "answer_gt",
        "em",
        "f1",
        "precision",
        "recall",
        "input_tokens",
        "output_tokens",
        "cost_usd",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def _run_noise_gate_method(
    sample: dict[str, Any],
    model: str,
    sim_threshold: float,
) -> tuple[str, int, int, float]:
    """Run noise-gate retrieval and generation for one threshold."""
    question = sample["question"]

    if model not in LLM_TO_EMBEDDING:
        raise KeyError(f"Unknown model for embeddings: {model}")

    paragraphs = flatten_context(sample["context"])
    emb_config = LLM_TO_EMBEDDING[model]
    selected_docs = noise_gate_select(
        question=question,
        paragraphs=paragraphs,
        provider=emb_config["provider"],
        embedding_model=emb_config["embedding_model"],
        token_budget=emb_config["max_tokens"],
        sim_threshold=sim_threshold,
        jac_threshold=NOISE_GATE_REDUNDANCY_THRESHOLD,
    )

    if selected_docs:
        prompt = build_retrieval_prompt(question, selected_docs)
    else:
        prompt = build_always_think_prompt(question)

    answer, input_tokens, output_tokens, cost_usd = call_llm(prompt, model)
    return answer, input_tokens, output_tokens, cost_usd


def _process_sample(
    sample_idx: int,
    sample: dict[str, Any],
    model: str,
) -> tuple[int, str, list[ResultDict], list[str]]:
    """Process all noise-gate thresholds for a single sample."""
    qid = sample["id"]
    sample_results: list[ResultDict] = []
    statuses: list[str] = []

    for sim_threshold in NOISE_GATE_SIM_THRESHOLDS:
        try:
            pred, input_tokens, output_tokens, cost_usd = _run_noise_gate_method(
                sample=sample,
                model=model,
                sim_threshold=sim_threshold,
            )
            sample_results.append(
                _build_result(
                    sample=sample,
                    sim_threshold=sim_threshold,
                    pred=pred,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                )
            )
            statuses.append(f"{NOISE_GATE_METHOD}(th={sim_threshold:g})=OK")
        except Exception as e:
            statuses.append(f"{NOISE_GATE_METHOD}(th={sim_threshold:g})=ERROR: {e}")

    return sample_idx, qid, sample_results, statuses


def run_noise_gate(
    model: str,
    n_rows: int,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> None:
    """Run the noise-gate ablation for a model."""
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    print("\n" + "=" * 70)
    print(f"Running noise-gate: {model} (n={n_rows}, workers={max_workers})")
    print(
        "Thresholds: "
        f"{', '.join(str(t) for t in NOISE_GATE_SIM_THRESHOLDS)} "
        f"(Jaccard={NOISE_GATE_REDUNDANCY_THRESHOLD})"
    )
    print("=" * 70)

    print("Loading HotpotQA...")
    ds = load_dataset_cached(
        dataset_name=DATASET_CONFIG["dataset_name"],
        subset=DATASET_CONFIG["subset"],
        split=DATASET_CONFIG["split"],
        n_rows=n_rows,
    )
    print(f"Loaded {len(ds)} samples")

    csv_path = RESULTS_DIR / f"results_noise_gate_{model}.csv"
    print("Rewriting results CSV from scratch for this run")

    worker_count = min(max_workers, len(ds)) if ds else 1
    print(f"Processing samples with {worker_count} threads...")

    results_by_sample: list[list[ResultDict] | None] = [None] * len(ds)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(_process_sample, i, sample, model)
            for i, sample in enumerate(ds)
        ]

        for future in as_completed(futures):
            sample_idx, qid, sample_results, statuses = future.result()
            results_by_sample[sample_idx] = sample_results
            print(f"  [{sample_idx + 1}/{len(ds)}] {qid}: {', '.join(statuses)}")

    all_results: list[ResultDict] = []
    for sample_results in results_by_sample:
        if sample_results:
            all_results.extend(sample_results)

    print(f"\nSaving CSV: {csv_path}")
    save_csv(model, all_results)

    print("\n" + "=" * 70)
    print(f"Complete! Results: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: uv run python -m experiments.run_noise_gate "
            "<model> <n_rows> [max_workers]"
        )
        print(f"\nAvailable models: {', '.join(MODELS)}")
        sys.exit(1)

    model = sys.argv[1]
    n_rows = int(sys.argv[2])
    max_workers = int(sys.argv[3]) if len(sys.argv) >= 4 else DEFAULT_MAX_WORKERS

    if model not in MODELS:
        print(f"Unknown model: {model}")
        print(f"Available: {MODELS}")
        sys.exit(1)

    run_noise_gate(model, n_rows, max_workers=max_workers)
