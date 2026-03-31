"""Adaptive-K selection evaluation script for CART paper.

Implements Stage 2 of CART: find natural evidence cluster cutpoint.

Usage from root:
    uv run python -m experiments.run_adaptive_k gpt-4o-mini 2
    uv run python -m experiments.run_adaptive_k claude-sonnet-4-6 100
    uv run python -m experiments.run_adaptive_k claude-sonnet-4-6 100 20
"""

import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments.cache_dataset import load_dataset_cached
from experiments.eval_utils import evaluate_sample
from experiments.baselines_config import MODELS, DATASET_CONFIG
from experiments.baselines import (
    flatten_context,
    build_retrieval_prompt,
    call_llm,
)
from experiments.embedding_utils import retrieve_top_k

RESULTS_DIR = Path(__file__).parent / "results" / "cart"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
DEFAULT_MAX_WORKERS = 20
METHOD_NAME = "adaptive_k"


def adaptive_k_select(
    scores: list[float],
    delta: float = 0.08,
) -> int:
    """Find natural cutpoint using largest adjacent gap in similarity scores.

    Implements Adaptive-k from Taguchi et al. EMNLP 2025.

    Args:
        scores: Sorted similarity scores (descending order)
        delta: Threshold for gap significance (default 0.08)

    Returns:
        Selected k* (1-indexed, capped at min(5, len(scores)) if no significant gap)
    """
    if len(scores) <= 1:
        return 1

    # Find largest gap between consecutive scores
    max_gap = 0
    max_gap_idx = 1  # Default to k*=1 if no gap found

    for i in range(len(scores) - 1):
        gap = scores[i] - scores[i + 1]
        if gap > max_gap:
            max_gap = gap
            max_gap_idx = i + 1  # k* is the count of documents up to this index

    # If no significant gap, default to min(5, N)
    if max_gap < delta:
        return min(5, len(scores))

    return max_gap_idx


ResultDict = dict[str, str | int | float]


def _build_result(
    sample: dict,
    pred: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    k_star: int,
) -> ResultDict:
    """Build a result row and attach evaluation metrics."""
    result: ResultDict = {
        "question_id": sample["id"],
        "method": METHOD_NAME,
        "answer_pred": pred,
        "answer_gt": sample["answer"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "k_star": k_star,
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


def save_csv(
    model: str,
    results: list[ResultDict],
) -> None:
    """Save per-record results to CSV.

    Args:
        model: Model name
        results: List of result dictionaries
    """
    csv_path = RESULTS_DIR / f"results_adaptive_k_{model}.csv"

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
        "k_star",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def run_adaptive_k_method(
    sample: dict,
    model: str,
) -> tuple[str, int, int, float, int]:
    """Implement adaptive-k: retrieve top-10, select k* via gap, then ask LLM.

    Args:
        sample: HotpotQA sample with "question" and "context" keys
        model: LLM model name

    Returns:
        Tuple of (answer, input_tokens, output_tokens, cost_usd, k_star)
    """
    question = sample["question"]
    context = sample["context"]

    from experiments.baselines_config import LLM_TO_EMBEDDING

    # Flatten context to list of paragraph strings
    paragraphs = flatten_context(context)

    # Get embedding config for this LLM model
    emb_config = LLM_TO_EMBEDDING[model]

    # Stage 1: Retrieve top-N=10 candidates by embedding similarity
    top_paragraphs, top_scores = retrieve_top_k(
        question=question,
        paragraphs=paragraphs,
        k=10,
        provider=emb_config["provider"],
        embedding_model=emb_config["embedding_model"],
        token_budget=emb_config["max_tokens"],
    )

    # Stage 2: Adaptive-K selection — find natural cutpoint via largest gap
    k_star = adaptive_k_select(top_scores, delta=0.08)

    # Select top-k* documents
    selected_paragraphs = top_paragraphs[:k_star]

    # Build prompt with selected context
    prompt = build_retrieval_prompt(question, selected_paragraphs)

    # Call LLM and get answer + tokens + cost
    answer, input_tokens, output_tokens, cost_usd = call_llm(prompt, model)
    return answer, input_tokens, output_tokens, cost_usd, k_star


def _process_sample(
    sample_idx: int,
    sample: dict,
    model: str,
) -> tuple[int, str, ResultDict | None, list[str]]:
    """Process one sample end-to-end.

    The adaptive-k retrieval and generation path runs once per sample so
    thread workers can keep the result join logic simple.
    """
    qid = sample["id"]

    try:
        pred, input_tokens, output_tokens, cost_usd, k_star = run_adaptive_k_method(
            sample, model
        )
        result = _build_result(
            sample=sample,
            pred=pred,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            k_star=k_star,
        )
        return sample_idx, qid, result, [f"{METHOD_NAME}=OK"]
    except Exception as e:
        return sample_idx, qid, None, [f"{METHOD_NAME}=ERROR: {e}"]


def run_adaptive_k(
    model: str,
    n_rows: int,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> None:
    """Run adaptive-k evaluation for a model.

    Args:
        model: Model name
        n_rows: Number of rows to evaluate
        max_workers: Number of threads used to process samples in parallel
    """
    print("\n" + "="*70)
    print(f"Running adaptive-k: {model} (n={n_rows}, workers={max_workers})")
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

    csv_path = RESULTS_DIR / f"results_adaptive_k_{model}.csv"
    print("Rewriting results CSV from scratch for this run")

    worker_count = min(max_workers, len(ds)) if ds else 1
    print(f"Processing samples with {worker_count} threads...")

    results_by_sample: list[ResultDict | None] = [None] * len(ds)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(_process_sample, i, sample, model)
            for i, sample in enumerate(ds)
        ]

        for future in as_completed(futures):
            sample_idx, qid, result, statuses = future.result()
            if result is not None:
                results_by_sample[sample_idx] = result
            print(f"  [{sample_idx+1}/{len(ds)}] {qid}: {', '.join(statuses)}")

    all_results: list[ResultDict] = []
    for result in results_by_sample:
        if result is not None:
            all_results.append(result)

    # Save CSV
    print(f"\nSaving CSV: {csv_path}")
    save_csv(model, all_results)

    print("\n" + "="*70)
    print(f"Complete! Results: {RESULTS_DIR}")
    print("="*70)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: uv run python -m experiments.run_adaptive_k "
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

    run_adaptive_k(model, n_rows, max_workers=max_workers)
