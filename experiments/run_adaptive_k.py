"""Adaptive-K selection evaluation script for CART paper.

Implements Stage 2 of CART: find natural evidence cluster cutpoint.

Usage from root:
    uv run python -m experiments.run_adaptive_k gpt-4o-mini 2
    uv run python -m experiments.run_adaptive_k claude-sonnet-4-6 100
"""

import csv
import sys
from pathlib import Path

from experiments.cache_dataset import load_dataset_cached
from experiments.eval_utils import evaluate_sample
from experiments.baselines_config import MODELS, DATASET_CONFIG
from experiments.baselines import _flatten_context, call_llm
from experiments.embedding_utils import retrieve_top_k

RESULTS_DIR = Path(__file__).parent / "results" / "cart"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


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


def load_or_create_csv(model: str) -> tuple[dict[str, dict], Path]:
    """Load existing CSV or create structure. Returns (cache_dict, path).

    Args:
        model: Model name

    Returns:
        Tuple of (cache dict keyed by question_id, csv path)
    """
    csv_path = RESULTS_DIR / f"results_adaptive_k_{model}.csv"
    cache = {}

    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                cache[row["question_id"]] = row

    return cache, csv_path


def save_csv(
    model: str,
    results: list[dict[str, str | int | float]],
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

    with open(csv_path, "w", newline="") as f:
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
    paragraphs = _flatten_context(context)

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
    context_str = "\n\n".join(
        [f"[{i+1}] {p}" for i, p in enumerate(selected_paragraphs)]
    )
    prompt = f"""Answer the HotpotQA question using only the provided context.

Rules:
- Output only the final answer.
- If the answer is yes or no, output exactly: yes or no.
- Otherwise output a short span or entity name only.
- Do not include any explanation.
- Do not repeat the question.
- If multiple positions/titles are mentioned, output only the one that directly answers the question.

Context:
{context_str}

Question: {question}

Answer:"""

    # Call LLM and get answer + tokens + cost
    answer, input_tokens, output_tokens, cost_usd = call_llm(prompt, model)
    return answer, input_tokens, output_tokens, cost_usd, k_star


def run_adaptive_k(model: str, n_rows: int) -> None:
    """Run adaptive-k evaluation for a model.

    Args:
        model: Model name
        n_rows: Number of rows to evaluate
    """
    print("\n" + "="*70)
    print(f"Running adaptive-k: {model} (n={n_rows})")
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

    # Process all samples
    print("\nProcessing samples...")
    all_results = []

    for i, sample in enumerate(ds):
        qid = sample["id"]

        if qid in cache:
            result = cache[qid]
            print(f"  [{i+1}/{len(ds)}] {qid} (cached)")
        else:
            try:
                pred, input_tokens, output_tokens, cost_usd, k_star = run_adaptive_k_method(
                    sample, model
                )
                result = {
                    "question_id": qid,
                    "method": "adaptive_k",
                    "answer_pred": pred,
                    "answer_gt": sample["answer"],
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": cost_usd,
                    "k_star": k_star,
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

        all_results.append(result)

    # Save CSV
    print(f"\nSaving CSV: {csv_path}")
    save_csv(model, all_results)

    print("\n" + "="*70)
    print(f"Complete! Results: {RESULTS_DIR}")
    print("="*70)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: uv run python -m experiments.run_adaptive_k <model> <n_rows>")
        print(f"\nAvailable models: {', '.join(MODELS)}")
        sys.exit(1)

    model = sys.argv[1]
    n_rows = int(sys.argv[2])

    if model not in MODELS:
        print(f"Unknown model: {model}")
        print(f"Available: {MODELS}")
        sys.exit(1)

    run_adaptive_k(model, n_rows)
