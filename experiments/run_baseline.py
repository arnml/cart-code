"""Main baseline evaluation script for CART paper.

Evaluates LLM performance on HotpotQA using different retrieval strategies.

Usage from root:
    uv run python -m experiments.run_baseline gpt-4o-mini 2
    uv run python -m experiments.run_baseline claude-sonnet-4-6 100
    uv run python -m experiments.run_baseline claude-sonnet-4-6 100 20
"""

import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from experiments.cache_dataset import load_dataset_cached
from experiments.eval_utils import evaluate_sample
from experiments.baselines_config import (
    MODELS,
    METHODS,
    DATASET_CONFIG,
    LLM_TO_EMBEDDING,
)
from experiments.baselines import (
    flatten_context,
    build_always_think_prompt,
    build_retrieval_prompt,
    call_llm,
)
from experiments.embedding_utils import retrieve_top_k

RESULTS_DIR = Path(__file__).parent / "results" / "baseline"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

DEFAULT_MAX_WORKERS = 20
RETRIEVAL_METHOD_TO_K = {
    "retrieval_k3": 3,
    "retrieval_k5": 5,
    "retrieval_k10": 10,
}

ResultDict = dict[str, str | int | float]


def save_csv(model: str, results: list[ResultDict]) -> None:
    """Save per-record results to CSV.

    Args:
        model: Model name
        results: List of result dictionaries
    """
    csv_path = RESULTS_DIR / f"baseline_{model}.csv"

    keys = [
        "question_id",
        "method",
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
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def _build_result(
    sample: dict[str, Any],
    method_name: str,
    pred: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> ResultDict:
    """Build a result row and attach evaluation metrics."""
    result: ResultDict = {
        "question_id": sample["id"],
        "method": method_name,
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


def _process_sample(
    sample_idx: int,
    sample: dict[str, Any],
    model: str,
) -> tuple[int, str, list[ResultDict], list[str]]:
    """Process all requested methods for a single sample.

    The retrieval rank for this sample is computed once and reused for the
    retrieval_k3/k5/k10 baselines.
    """
    qid = sample["id"]
    question = sample["question"]
    sample_results: list[ResultDict] = []
    statuses: list[str] = []

    top_paragraphs = None
    if any(method_name in RETRIEVAL_METHOD_TO_K for method_name in METHODS):
        try:
            paragraphs = flatten_context(sample["context"])
            emb_config = LLM_TO_EMBEDDING[model]
            top_paragraphs, _ = retrieve_top_k(
                question=question,
                paragraphs=paragraphs,
                k=10,
                provider=emb_config["provider"],
                embedding_model=emb_config["embedding_model"],
                token_budget=emb_config["max_tokens"],
            )
        except Exception as e:
            statuses.append(f"retrieval_setup=ERROR: {e}")
            top_paragraphs = None

    for method_name in METHODS:
        try:
            if method_name == "always_think":
                prompt = build_always_think_prompt(question)
                pred, input_tokens, output_tokens, cost_usd = call_llm(prompt, model)
            elif method_name in RETRIEVAL_METHOD_TO_K:
                if top_paragraphs is None:
                    raise RuntimeError("retrieval ranking failed for this sample")
                k = RETRIEVAL_METHOD_TO_K[method_name]
                prompt = build_retrieval_prompt(question, top_paragraphs[:k])
                pred, input_tokens, output_tokens, cost_usd = call_llm(prompt, model)
            else:
                raise ValueError(f"Unknown baseline method: {method_name}")

            sample_results.append(
                _build_result(
                    sample=sample,
                    method_name=method_name,
                    pred=pred,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                )
            )
            statuses.append(f"{method_name}=OK")
        except Exception as e:
            statuses.append(f"{method_name}=ERROR: {e}")

    return sample_idx, qid, sample_results, statuses


def run_baseline(
    model: str,
    n_rows: int,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> None:
    """Run baseline evaluation for a model.

    Args:
        model: Model name
        n_rows: Number of rows to evaluate
        max_workers: Number of threads used to process samples in parallel
    """
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    print("\n" + "="*70)
    print(f"Running baseline: {model} (n={n_rows}, workers={max_workers})")
    print("="*70)

    # Load the local dataset snapshot.
    print("Loading HotpotQA...")
    ds = load_dataset_cached(
        dataset_name=DATASET_CONFIG["dataset_name"],
        subset=DATASET_CONFIG["subset"],
        split=DATASET_CONFIG["split"],
        n_rows=n_rows,
    )
    print(f"Loaded {len(ds)} samples")

    csv_path = RESULTS_DIR / f"baseline_{model}.csv"
    print("Rewriting results CSV from scratch for this run")

    # Process samples in parallel. Each worker handles the full method path for
    # one sample so retrieval work can be reused within that sample.
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
            print(f"  [{sample_idx+1}/{len(ds)}] {qid}: {', '.join(statuses)}")

    all_results: list[ResultDict] = []
    for sample_results in results_by_sample:
        if sample_results:
            all_results.extend(sample_results)

    # Save CSV
    print(f"\nSaving CSV: {csv_path}")
    save_csv(model, all_results)

    print("\n" + "="*70)
    print(f"Complete! Results: {RESULTS_DIR}")
    print("="*70)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: uv run python -m experiments.run_baseline "
            "<model> <n_rows> [max_workers]"
        )
        print(f"\nAvailable models: {', '.join(MODELS)}")
        print(f"Available methods: {', '.join(METHODS)}")
        sys.exit(1)

    model = sys.argv[1]
    n_rows = int(sys.argv[2])
    max_workers = int(sys.argv[3]) if len(sys.argv) >= 4 else DEFAULT_MAX_WORKERS

    if model not in MODELS:
        print(f"Unknown model: {model}")
        print(f"Available: {MODELS}")
        sys.exit(1)

    run_baseline(model, n_rows, max_workers=max_workers)
