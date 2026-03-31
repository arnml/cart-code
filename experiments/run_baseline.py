"""
Main baseline evaluation script for CART paper.

Evaluates LLM performance on HotpotQA using different retrieval strategies.

Usage from root:
    uv run python -m experiments.run_baseline gpt-4o-mini 2
    uv run python -m experiments.run_baseline claude-sonnet-4-6 100
"""

import csv
import sys
from pathlib import Path

from experiments.cache_dataset import load_dataset_cached
from experiments.eval_utils import evaluate_sample
from experiments.baselines_config import (
    MODELS,
    METHODS,
    DATASET_CONFIG,
)
from experiments.baselines import get_method

RESULTS_DIR = Path(__file__).parent / "results" / "baseline"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


def load_or_create_csv(model: str) -> tuple[dict[tuple[str, str], dict[str, str | int | float]], Path]:
    """Load existing CSV or create structure. Returns (cache_dict, path).

    Args:
        model: Model name

    Returns:
        Tuple of (cache dict keyed by (question_id, method), csv path)
    """
    csv_path = RESULTS_DIR / f"baseline_{model}.csv"
    cache = {}

    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["question_id"], row["method"])
                cache[key] = row

    return cache, csv_path


def save_csv(model: str, results: list[dict[str, str | int | float]]) -> None:
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

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def _process_method(
    method_name: str,
    method_fn: any,
    ds: list[dict[str, any]],
    model: str,
    cache: dict[tuple[str, str], dict[str, str | int | float]],
) -> list[dict[str, str | int | float]]:
    """Process a single baseline method across all samples.

    Args:
        method_name: Name of the method
        method_fn: Method function (e.g., always_think, retrieval_k5)
        ds: Dataset samples
        model: Model name
        cache: Existing result cache

    Returns:
        List of result dictionaries
    """
    print(f"\nMethod: {method_name}")
    new_results = []

    for i, sample in enumerate(ds):
        qid = sample["id"]
        key = (qid, method_name)

        if key in cache:
            result = cache[key]
            print(f"  [{i+1}/{len(ds)}] {qid} (cached)")
        else:
            try:
                pred, input_tokens, output_tokens, cost_usd = method_fn(sample, model)
                result = {
                    "question_id": qid,
                    "method": method_name,
                    "answer_pred": pred,
                    "answer_gt": sample["answer"],
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": cost_usd,
                }

                # Evaluate
                metrics = evaluate_sample(
                    pred,
                    sample["answer"],
                )
                result.update(
                    {
                        "em": metrics.em,
                        "f1": metrics.f1,
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                    }
                )

                print(f"  [{i+1}/{len(ds)}] {qid} OK")

            except Exception as e:
                print(f"  [{i+1}/{len(ds)}] {qid} ERROR: {e}")
                continue

        new_results.append(result)

    return new_results


def run_baseline(model: str, n_rows: int, methods: list[str] | None = None) -> None:
    """Run baseline evaluation for a model.

    Args:
        model: Model name
        n_rows: Number of rows to evaluate
        methods: Optional list of methods to run (defaults to all)
    """
    if methods is None:
        methods = METHODS

    print("\n" + "="*70)
    print(f"Running baseline: {model} (n={n_rows})")
    print("="*70)

    # Load dataset using cache
    print("Loading HotpotQA...")
    ds = load_dataset_cached(
        dataset_name=DATASET_CONFIG["dataset_name"],
        subset=DATASET_CONFIG["subset"],
        split=DATASET_CONFIG["split"],
        n_rows=n_rows,
    )
    print(f"Loaded {len(ds)} samples")

    # Load cache
    cache, csv_path = load_or_create_csv(model)
    print(f"Cache: {len(cache)} existing results")

    # Collect all results (initialized as empty, will be populated by _process_method)
    all_results = []

    # Process each method
    for method_name in methods:
        method_fn = get_method(method_name)
        new_results = _process_method(
            method_name, method_fn, ds, model, cache
        )
        all_results.extend(new_results)

    # Save CSV
    print(f"\nSaving CSV: {csv_path}")
    save_csv(model, all_results)

    print("\n" + "="*70)
    print(f"Complete! Results: {RESULTS_DIR}")
    print("="*70)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: uv run python -m experiments.run_baseline <model> <n_rows>")
        print(f"\nAvailable models: {', '.join(MODELS)}")
        print(f"Available methods: {', '.join(METHODS)}")
        sys.exit(1)

    model = sys.argv[1]
    n_rows = int(sys.argv[2])

    if model not in MODELS:
        print(f"Unknown model: {model}")
        print(f"Available: {MODELS}")
        sys.exit(1)

    run_baseline(model, n_rows)
