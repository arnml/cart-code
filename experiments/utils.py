"""Shared utilities for HotpotQA experiments.

Provides:
- Evaluation metrics (HotpotQA standard: EM, F1, precision, recall)
- CSV loading and aggregation
- Result building and saving
- Parallel sample processing with ThreadPoolExecutor
- Cost calculation utilities
"""

import csv
import re
import statistics
import string
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


# ============================================================================
# Evaluation Metrics (HotpotQA v1 standard)
# ============================================================================

@dataclass
class EvalMetrics:
    """Container for evaluation results."""
    em: float  # Exact match
    f1: float  # F1 score
    precision: float  # Token-level precision
    recall: float  # Token-level recall


def normalize_answer(s: str) -> str:
    """Normalize answer for evaluation (HotpotQA standard).

    Following official HotpotQA v1 evaluation script exactly.
    """
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """Compute token-level F1, precision, recall (HotpotQA standard).

    Returns:
        (f1, precision, recall) - Tuple of three floats [0, 1]
        Returns (0, 0, 0) if yes/no/noanswer mismatch (strict evaluation)

    Special handling:
        - If ground_truth is "yes"/"no"/"noanswer", prediction must match exactly
        - Handles empty predictions gracefully
    """
    normalized_pred = normalize_answer(prediction)
    normalized_gt = normalize_answer(ground_truth)

    # HotpotQA special rule: yes/no/noanswer must match exactly
    if normalized_gt in ['yes', 'no', 'noanswer']:
        if normalized_pred != normalized_gt:
            return 0.0, 0.0, 0.0

    # Tokenize
    pred_tokens = normalized_pred.split()
    gt_tokens = normalized_gt.split()

    # Handle empty predictions
    if not pred_tokens or not gt_tokens:
        return 0.0, 0.0, 0.0

    # Compute overlap using multiset (Counter)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0, 0.0, 0.0

    # Precision & Recall
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)

    # F1
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Compute exact match (0 or 1) after normalization.

    Returns:
        True if normalized prediction == normalized ground_truth, False otherwise
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def evaluate_sample(
    prediction: str,
    ground_truth: str,
) -> EvalMetrics:
    """Evaluate a single prediction-ground_truth pair.

    Args:
        prediction: Generated answer
        ground_truth: Reference answer

    Returns:
        EvalMetrics dataclass with EM, F1, precision, recall
    """
    # Exact Match
    em = float(exact_match_score(prediction, ground_truth))

    # F1, Precision, Recall
    f1, prec, recall = f1_score(prediction, ground_truth)

    return EvalMetrics(
        em=em,
        f1=f1,
        precision=prec,
        recall=recall,
    )


_MODEL_PRICING = {
    "gpt-5.4-mini": (0.00075, 0.0045),
    "gpt-4o-mini": (0.00015, 0.0006),
    "claude-haiku": (0.001, 0.005),
    "claude-sonnet": (0.003, 0.015),
    "claude-opus": (0.015, 0.045),
}


def cost_usd(
    input_tokens: int,
    output_tokens: int,
    model: str = "claude-haiku",
) -> float:
    """Estimate LLM generation cost in USD based on 2026 pricing tiers.

    Note: Only includes LLM API costs (input + output tokens).
    Embedding costs (if using retrieval) are cached and calculated separately.

    Args:
        input_tokens: Number of input tokens to the LLM
        output_tokens: Number of output tokens from the LLM
        model: Model name (claude-haiku, claude-sonnet, gpt-4o-mini, etc.)

    Returns:
        Total LLM cost in USD
    """
    model_lower = model.lower()

    # Find matching pricing
    in_rate, out_rate = _MODEL_PRICING.get("claude-haiku", (0.001, 0.005))

    for model_key in _MODEL_PRICING:
        if model_key in model_lower:
            in_rate, out_rate = _MODEL_PRICING[model_key]
            break

    return (input_tokens * in_rate + output_tokens * out_rate) / 1000


# ============================================================================
# CSV Loading & Aggregation (used by analyse_* scripts)
# ============================================================================

def load_csv(csv_path: Path) -> list[dict[str, str]]:
    """Load results from CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of result dictionaries

    Raises:
        FileNotFoundError: If CSV doesn't exist
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")

    results = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)

    return results


def aggregate_metrics(
    results: list[dict[str, str]],
    include_k_star: bool = False,
) -> dict[str, Any]:
    """Aggregate metrics across results.

    Computes mean values for all metrics.

    Args:
        results: List of result dictionaries (from CSV)
        include_k_star: If True, include k_star distribution stats

    Returns:
        Dict with aggregated metrics:
        - count, input_mean, output_mean, total_tokens_mean, cost_mean
        - em_mean, f1_mean, precision_mean, recall_mean
        - (optional) k_star_mean, k_star_median, k_star_stdev
    """
    n = len(results)
    if n == 0:
        raise ValueError("No results to aggregate")

    input_tokens = [int(r["input_tokens"]) for r in results]
    output_tokens = [int(r["output_tokens"]) for r in results]
    total_tokens = [i + o for i, o in zip(input_tokens, output_tokens)]
    costs = [float(r["cost_usd"]) for r in results]
    ems = [float(r["em"]) for r in results]
    f1s = [float(r["f1"]) for r in results]
    precisions = [float(r["precision"]) for r in results]
    recalls = [float(r["recall"]) for r in results]

    f1_mean = sum(f1s) / n
    total_tokens_mean = sum(total_tokens) / n
    cost_mean = sum(costs) / n

    aggregated = {
        "count": n,
        "input_mean": sum(input_tokens) / n,
        "output_mean": sum(output_tokens) / n,
        "total_tokens_mean": total_tokens_mean,
        "cost_mean": cost_mean,
        "em_mean": sum(ems) / n,
        "f1_mean": f1_mean,
        "precision_mean": sum(precisions) / n,
        "recall_mean": sum(recalls) / n,
    }

    # Optional k_star statistics
    if include_k_star:
        k_stars = []
        for r in results:
            k_star_val = r.get("k_star", "").strip()
            if k_star_val:
                k_stars.append(int(k_star_val))

        if k_stars:
            aggregated.update({
                "k_star_mean": sum(k_stars) / len(k_stars),
                "k_star_median": statistics.median(k_stars),
                "k_star_stdev": statistics.stdev(k_stars) if len(k_stars) > 1 else 0.0,
            })

    return aggregated


def aggregate_by_method(
    results: list[dict[str, str]],
) -> dict[str, dict[str, float]]:
    """Aggregate metrics grouped by method.

    Args:
        results: List of result dictionaries

    Returns:
        Dict mapping method name to aggregated metrics

    Raises:
        ValueError: If duplicate (question_id, method) pairs exist
    """
    by_method = defaultdict(list)
    by_method_unique = defaultdict(set)
    seen_pairs = set()

    for row in results:
        method = row["method"]
        question_id = row["question_id"]
        pair = (question_id, method)

        if pair in seen_pairs:
            raise ValueError(
                f"Duplicate entry: question_id={question_id}, method={method}"
            )
        seen_pairs.add(pair)

        by_method[method].append(row)
        by_method_unique[method].add(question_id)

    # Aggregate each method separately
    aggregated = {}
    for method, rows in by_method.items():
        n = len(by_method_unique[method])

        input_tokens = [int(r["input_tokens"]) for r in rows]
        output_tokens = [int(r["output_tokens"]) for r in rows]
        total_tokens = [i + o for i, o in zip(input_tokens, output_tokens)]
        costs = [float(r["cost_usd"]) for r in rows]
        ems = [float(r["em"]) for r in rows]
        f1s = [float(r["f1"]) for r in rows]
        precisions = [float(r["precision"]) for r in rows]
        recalls = [float(r["recall"]) for r in rows]

        f1_mean = sum(f1s) / n
        total_tokens_mean = sum(total_tokens) / n
        cost_mean = sum(costs) / n

        aggregated[method] = {
            "count": n,
            "input_mean": sum(input_tokens) / n,
            "output_mean": sum(output_tokens) / n,
            "total_tokens_mean": total_tokens_mean,
            "cost_mean": cost_mean,
            "em_mean": sum(ems) / n,
            "f1_mean": f1_mean,
            "precision_mean": sum(precisions) / n,
            "recall_mean": sum(recalls) / n,
        }

    return aggregated


def aggregate_by_k(
    results: list[dict[str, str]],
) -> dict[int, dict[str, float]]:
    """Aggregate metrics grouped by k value.

    Args:
        results: List of result dictionaries (must have 'k' column)

    Returns:
        Dict mapping k (int) to aggregated metrics:
        - count, input_mean, output_mean, total_tokens_mean, cost_mean
        - em_mean, f1_mean, precision_mean, recall_mean
    """
    by_k = defaultdict(list)

    for row in results:
        k = int(row["k"])
        by_k[k].append(row)

    # Aggregate each k separately
    aggregated = {}
    for k in sorted(by_k.keys()):
        rows = by_k[k]
        n = len(rows)

        input_tokens = [int(r["input_tokens"]) for r in rows]
        output_tokens = [int(r["output_tokens"]) for r in rows]
        total_tokens = [i + o for i, o in zip(input_tokens, output_tokens)]
        costs = [float(r["cost_usd"]) for r in rows]
        ems = [float(r["em"]) for r in rows]
        f1s = [float(r["f1"]) for r in rows]
        precisions = [float(r["precision"]) for r in rows]
        recalls = [float(r["recall"]) for r in rows]

        f1_mean = sum(f1s) / n
        total_tokens_mean = sum(total_tokens) / n
        cost_mean = sum(costs) / n

        aggregated[k] = {
            "count": n,
            "input_mean": sum(input_tokens) / n,
            "output_mean": sum(output_tokens) / n,
            "total_tokens_mean": total_tokens_mean,
            "cost_mean": cost_mean,
            "em_mean": sum(ems) / n,
            "f1_mean": f1_mean,
            "precision_mean": sum(precisions) / n,
            "recall_mean": sum(recalls) / n,
        }

    return aggregated


# ============================================================================
# Result Building & Saving (used by run_* scripts)
# ============================================================================

def build_result(
    sample: dict[str, Any],
    prediction: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    extra_fields: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a result row with evaluation metrics.

    Args:
        sample: HotpotQA sample dict with "id", "answer" keys
        prediction: Predicted answer string
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens generated
        cost_usd: Cost in USD
        extra_fields: Optional dict of additional fields to include

    Returns:
        Result dict with evaluation metrics (em, f1, precision, recall)
    """
    result = {
        "question_id": sample["id"],
        "answer_pred": prediction,
        "answer_gt": sample["answer"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
    }

    # Add evaluation metrics
    metrics = evaluate_sample(prediction, sample["answer"])
    result.update({
        "em": metrics.em,
        "f1": metrics.f1,
        "precision": metrics.precision,
        "recall": metrics.recall,
    })

    # Add extra fields (method, threshold, k_star, etc.)
    if extra_fields:
        result.update(extra_fields)

    return result


def save_results_csv(
    csv_path: Path,
    results: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    """Save results to CSV file.

    Args:
        csv_path: Output CSV path
        results: List of result dictionaries
        fieldnames: CSV column names (in desired order)
    """
    csv_path.parent.mkdir(exist_ok=True, parents=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


# ============================================================================
# Parallel Sample Processing (used by run_* scripts)
# ============================================================================

def process_samples_parallel(
    dataset: list[dict[str, Any]],
    process_fn: Callable[[int, dict[str, Any]], tuple[int, str, Any, list[str]]],
    max_workers: int = 20,
) -> tuple[list[Any], list[str]]:
    """Process dataset samples in parallel using ThreadPoolExecutor.

    Args:
        dataset: List of samples to process
        process_fn: Function(sample_idx, sample) -> (sample_idx, qid, result, statuses)
        max_workers: Number of worker threads

    Returns:
        Tuple of (results_by_sample, all_statuses)
        - results_by_sample: List indexed by sample position (None if failed)
        - all_statuses: Flat list of status strings for logging
    """
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    worker_count = min(max_workers, len(dataset)) if dataset else 1
    results_by_sample: list[Any] = [None] * len(dataset)
    all_statuses: list[str] = []

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(process_fn, i, sample)
            for i, sample in enumerate(dataset)
        ]

        for future in as_completed(futures):
            sample_idx, qid, result, statuses = future.result()
            results_by_sample[sample_idx] = result
            all_statuses.extend(statuses)
            print(f"  [{sample_idx+1}/{len(dataset)}] {qid}: {', '.join(statuses)}")

    return results_by_sample, all_statuses


def flatten_results(
    results_by_sample: list[Any],
    is_list: bool = False,
) -> list[Any]:
    """Flatten results_by_sample into a single list.

    Args:
        results_by_sample: List where each element is a result (or list of results, or None)
        is_list: If True, each element is a list (extend); else treat as single item (append)

    Returns:
        Flat list of all results
    """
    all_results = []
    for result in results_by_sample:
        if result is not None:
            if is_list:
                all_results.extend(result)
            else:
                all_results.append(result)
    return all_results


# ============================================================================
# Experiment Runner Template
# ============================================================================

def run_experiment(
    model: str,
    n_rows: int,
    max_workers: int,
    dataset_config: dict[str, str],
    process_fn: Callable[[int, dict[str, Any], str], tuple[int, str, Any, list[str]]],
    save_fn: Callable[[Path, list[Any]], None],
    results_path: Path,
    experiment_name: str,
) -> None:
    """Run a standard experiment pipeline.

    Orchestrates: dataset loading -> parallel processing -> result saving.

    Args:
        model: Model name to test
        n_rows: Number of samples to process
        max_workers: Number of parallel threads
        dataset_config: Dict with dataset_name, subset, split
        process_fn: Function(sample_idx, sample, model) -> (idx, qid, result, statuses)
        save_fn: Function(results_path, results) to save results
        results_path: Where to save results CSV
        experiment_name: Name for logging (e.g. "baseline", "adaptive-k")
    """
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    print("\n" + "="*70)
    print(f"Running {experiment_name}: {model} (n={n_rows}, workers={max_workers})")
    print("="*70)

    # Load dataset
    print("Loading HotpotQA...")
    from experiments.cache_dataset import load_dataset_cached
    ds = load_dataset_cached(
        dataset_name=dataset_config["dataset_name"],
        subset=dataset_config["subset"],
        split=dataset_config["split"],
        n_rows=n_rows,
    )
    print(f"Loaded {len(ds)} samples")

    # Process in parallel
    print("Rewriting results CSV from scratch for this run")
    worker_count = min(max_workers, len(ds)) if ds else 1
    print(f"Processing samples with {worker_count} threads...")

    results_by_sample: list[Any] = [None] * len(ds)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(process_fn, i, sample, model)
            for i, sample in enumerate(ds)
        ]

        for future in as_completed(futures):
            sample_idx, qid, result, statuses = future.result()
            results_by_sample[sample_idx] = result
            print(f"  [{sample_idx+1}/{len(ds)}] {qid}: {', '.join(statuses)}")

    # Save results
    print(f"\nSaving CSV: {results_path}")
    save_fn(results_path, flatten_results(results_by_sample, is_list=True))

    print("\n" + "="*70)
    print(f"Complete! Results: {results_path.parent}")
    print("="*70)
