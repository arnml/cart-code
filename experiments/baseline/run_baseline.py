"""
Main baseline evaluation script for CART paper.

Evaluates LLM performance on HotpotQA using different retrieval strategies.

Usage from root:
    python -m experiments.baseline.run_baseline gpt-4o-mini 2
    python -m experiments.baseline.run_baseline claude-sonnet-4-6 100
"""

import csv
import sys
from pathlib import Path

from experiments.core_utilities.cache_dataset import load_dataset_cached
from experiments.core_utilities.eval_utils import (
    evaluate_sample,
    aggregate_metrics,
)
from experiments.core_utilities.baselines_config import (
    MODELS,
    METHODS,
    DATASET_CONFIG,
)
from experiments.core_utilities.baselines import get_method

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_or_create_csv(model: str) -> tuple[dict, Path]:
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


def save_csv(model: str, results: list[dict]):
    """Save per-record results to CSV.

    Args:
        model: Model name
        results: List of result dictionaries
    """
    csv_path = RESULTS_DIR / f"baseline_{model}.csv"

    keys = [
        "question_id",
        "method",
        "n_total",
        "answer_pred",
        "answer_gt",
        "em",
        "f1",
        "precision",
        "recall",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "token_efficiency",
        "cost_efficiency",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def save_md(model: str, agg_by_method: dict, n: int):
    """Save summary report to MD.

    Args:
        model: Model name
        agg_by_method: Dict mapping method name to aggregated metrics
        n: Number of rows evaluated
    """
    md_path = RESULTS_DIR / f"baseline_{model}.md"

    lines = [
        f"# Baseline Results: {model}",
        f"\n**Evaluation Setup**: n={n}, dataset=HotpotQA (distractor, validation split)\n",
        "## Aggregate Metrics\n",
        "| Method | Count | EM | F1 | Precision | Recall | Token-Eff | Cost-Eff |",
        "|--------|-------|----|----|-----------|--------|-----------|----------|",
    ]

    for method, metrics in agg_by_method.items():
        em = metrics.get("em_mean", 0)
        f1 = metrics.get("f1_mean", 0)
        prec = metrics.get("precision_mean", 0)
        rec = metrics.get("recall_mean", 0)
        tok_eff = metrics.get("token_efficiency_mean", 0)
        cost_eff = metrics.get("cost_efficiency_mean", 0)
        count = metrics.get("count", 0)

        lines.append(
            f"| {method} | {count} | {em:.3f} | {f1:.3f} | {prec:.3f} | {rec:.3f} | {tok_eff:.3f} | {cost_eff:.2f} |"
        )

    md_path.write_text("\n".join(lines) + "\n")


def _process_method(
    method_name: str,
    method_fn,
    ds: list,
    model: str,
    n_rows: int,
    cache: dict,
) -> tuple[list[dict], list[dict]]:
    """Process a single baseline method across all samples.

    Extracted as separate function to reduce cognitive complexity of run_baseline.

    Args:
        method_name: Name of the method
        method_fn: Method function (e.g., always_think, retrieval_k5)
        ds: Dataset samples
        model: Model name
        n_rows: Total number of rows
        cache: Existing result cache

    Returns:
        Tuple of (new_results, method_results_for_aggregation)
    """
    print(f"\nMethod: {method_name}")
    new_results = []
    method_results = []
    
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
                    "n_total": n_rows,
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
                    input_tokens,
                    output_tokens,
                    cost_usd,
                )
                result.update(
                    {
                        "em": metrics.em,
                        "f1": metrics.f1,
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "token_efficiency": metrics.token_efficiency or 0,
                        "cost_efficiency": metrics.cost_efficiency or 0,
                    }
                )

                print(f"  [{i+1}/{len(ds)}] {qid} ✓")

            except Exception as e:
                print(f"  [{i+1}/{len(ds)}] {qid} ✗ Error: {e}")
                continue

        new_results.append(result)
        method_results.append(result)

    return new_results, method_results


def run_baseline(model: str, n_rows: int, methods: list[str] = None):
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

    # Collect all results (including cached)
    all_results = list(cache.values()) if cache else []
    agg_by_method = {m: [] for m in methods}

    # Process each method
    for method_name in methods:
        method_fn = get_method(method_name)
        new_results, method_results = _process_method(
            method_name, method_fn, ds, model, n_rows, cache
        )
        all_results.extend(new_results)
        agg_by_method[method_name] = method_results

    # Save CSV
    print(f"\nSaving CSV: {csv_path}")
    save_csv(model, all_results)

    # Aggregate and save MD
    print("Saving MD summary...")
    agg_by_method_metrics = {}
    for method_name, results in agg_by_method.items():
        if results:
            agg = aggregate_metrics(
                [r["answer_pred"] for r in results],
                [r["answer_gt"] for r in results],
                [int(r["input_tokens"]) for r in results],
                [int(r["output_tokens"]) for r in results],
                [float(r["cost_usd"]) for r in results],
            )
            agg_by_method_metrics[method_name] = agg

    save_md(model, agg_by_method_metrics, n_rows)

    print("\n" + "="*70)
    print(f"Complete! Results: {RESULTS_DIR}")
    print("="*70)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m experiments.baseline.run_baseline <model> <n_rows>")
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
