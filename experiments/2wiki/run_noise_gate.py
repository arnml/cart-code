"""2WikiMultiHopQA noise-gate evaluation.

Usage from root:
    uv run python -m experiments.2wiki.run_noise_gate 10
    uv run python -m experiments.2wiki.run_noise_gate 10 5
    uv run python -m experiments.2wiki.run_noise_gate 10 5 --split validation
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_from_disk
from sklearn.metrics.pairwise import cosine_similarity

from experiments.baselines import call_llm
from experiments.baselines_config import LLM_TO_EMBEDDING, MODELS
from experiments.embedding_utils import embed_text
from experiments.eval_utils import evaluate_sample

from .utils import (
    CACHE_DIR,
    build_always_think_prompt,
    build_retrieval_prompt,
    flatten_context,
    get_answer,
    get_raw_context,
    get_question,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "noise_gate"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_MAX_WORKERS = 20
DEFAULT_SPLIT = "validation"
NOISE_GATE_SIM_THRESHOLDS = (0.2, 0.25, 0.3, 0.35, 0.5)
NOISE_GATE_REDUNDANCY_THRESHOLD = 0.65
NOISE_GATE_METHOD = "noise_gate"

ResultDict = dict[str, str | int | float]


def _sample_rows(ds: Any, n_rows: int | None) -> list[dict[str, Any]]:
    if n_rows is None or n_rows <= 0:
        return [dict(row) for row in ds]

    n_rows = min(n_rows, len(ds))
    rng = random.Random(42)
    indices = rng.sample(range(len(ds)), n_rows)
    return [dict(row) for row in ds.select(indices)]


def _load_dataset_cached(split: str, n_rows: int | None) -> list[dict[str, Any]]:
    cache_dir = Path(CACHE_DIR) / split
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Local dataset snapshot not found at {cache_dir}. "
            f"Run `uv run python -m experiments.2wiki.download_dataset --split {split}` first."
        )

    ds = load_from_disk(cache_dir)
    return _sample_rows(ds, n_rows)


def _jaccard(text_a: str, text_b: str) -> float:
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


def _evaluate_prediction(prediction: str, sample: dict[str, Any]) -> Any:
    return evaluate_sample(prediction, get_answer(sample))


def _build_result(
    sample: dict[str, Any],
    sim_threshold: float,
    pred: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> ResultDict:
    result: ResultDict = {
        "question_id": sample["id"],
        "method": NOISE_GATE_METHOD,
        "threshold": sim_threshold,
        "answer_pred": pred,
        "answer_gt": get_answer(sample),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
    }

    metrics = _evaluate_prediction(pred, sample)
    result.update(
        {
            "em": metrics.em,
            "f1": metrics.f1,
            "precision": metrics.precision,
            "recall": metrics.recall,
        }
    )
    return result


def save_csv(model: str, split: str, results: list[ResultDict]) -> None:
    csv_path = RESULTS_DIR / f"results_noise_gate_{model}_{split}.csv"
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
    question = get_question(sample)

    if model not in LLM_TO_EMBEDDING:
        raise KeyError(f"Unknown model for embeddings: {model}")

    paragraphs = flatten_context(get_raw_context(sample))
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

    return call_llm(prompt, model)


def _process_sample(
    sample_idx: int,
    sample: dict[str, Any],
    model: str,
) -> tuple[int, str, list[ResultDict], list[str]]:
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
    split: str = DEFAULT_SPLIT,
) -> None:
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    if split not in {"train", "validation"}:
        raise ValueError("split must be either 'train' or 'validation'")

    print("\n" + "=" * 70)
    print(
        f"Running 2Wiki noise-gate: {model} (n={n_rows}, workers={max_workers}, split={split})"
    )
    print(
        "Thresholds: "
        f"{', '.join(str(t) for t in NOISE_GATE_SIM_THRESHOLDS)} "
        f"(Jaccard={NOISE_GATE_REDUNDANCY_THRESHOLD})"
    )
    print("=" * 70)

    print("Loading 2WikiMultiHopQA...")
    ds = _load_dataset_cached(split=split, n_rows=n_rows)
    print(f"Loaded {len(ds)} samples")

    csv_path = RESULTS_DIR / f"results_noise_gate_{model}_{split}.csv"
    print(f"Rewriting results CSV: {csv_path}")

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
    save_csv(model, split, all_results)

    print("\n" + "=" * 70)
    print(f"Complete! Results: {RESULTS_DIR}")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 2WikiMultiHopQA noise gate.")
    parser.add_argument("n_rows", type=int, help="Number of rows to sample.")
    parser.add_argument(
        "max_workers",
        type=int,
        nargs="?",
        default=DEFAULT_MAX_WORKERS,
        help="Maximum number of worker threads (default: 20).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=MODELS,
        help="Model to evaluate (default: gpt-5.4-mini).",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        choices=["train", "validation"],
        help="Dataset split to use (default: validation).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_noise_gate(
        model=args.model,
        n_rows=args.n_rows,
        max_workers=args.max_workers,
        split=args.split,
    )


if __name__ == "__main__":
    main()
