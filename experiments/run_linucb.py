"""LinUCB reranker evaluation script.

Loads a trained LinUCB model and evaluates on HotpotQA validation split.
Tests multiple k values (2, 3, 5) in a single run, producing one CSV row per k per question.

Usage from root:
    uv run python -m experiments.run_linucb gpt-4o-mini 100
    uv run python -m experiments.run_linucb claude-sonnet-4-6 100 20
"""

from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from experiments.baselines_config import DATASET_CONFIG, LLM_CONFIG, MODELS
from experiments.cache_dataset import load_dataset_cached
from experiments.eval_utils import evaluate_sample
from experiments.linucb import LinUCBReranker
from experiments.llm_anthropic import call_anthropic
from experiments.llm_openai import call_openai
from experiments.preprocessing import (
    build_retrieval_prompt,
    flatten_context,
)

RESULTS_DIR = Path(__file__).parent / "results" / "linucb"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

MODEL_PATH = Path(__file__).parent / "cache" / "linucb_model.json"
LINUCB_K_VALUES = (2, 3, 5)
LINUCB_METHOD = "linucb"
DEFAULT_MAX_WORKERS = 20

# CSV fieldnames
FIELDNAMES = [
    "question_id",
    "method",
    "k",
    "answer_pred",
    "answer_gt",
    "em",
    "f1",
    "precision",
    "recall",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "selected_titles",
]

ResultDict = dict[str, str | int | float | bool]


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


def _build_result(
    sample: dict[str, Any],
    k: int,
    pred: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    selected_titles: list[str],
) -> ResultDict:
    """Build a result row and attach evaluation metrics."""
    metrics = evaluate_sample(pred, sample["answer"])
    result: ResultDict = {
        "question_id": sample["id"],
        "method": LINUCB_METHOD,
        "k": k,
        "answer_pred": pred,
        "answer_gt": sample["answer"],
        "em": metrics.em,
        "f1": metrics.f1,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "selected_titles": json.dumps(selected_titles),
    }
    return result


def save_csv(model: str, results: list[ResultDict]) -> None:
    """Save per-record results to a CSV file."""
    csv_path = RESULTS_DIR / f"linucb_{model}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)


def _run_linucb_inference(
    sample: dict[str, Any],
    model: str,
    reranker: LinUCBReranker,
    k: int,
) -> tuple[str, int, int, float, list[str]]:
    """Run LinUCB selection and LLM generation for one k value."""
    question = sample["question"]
    titles = sample["context"]["title"]
    sentences_list = sample["context"]["sentences"]
    paragraphs = flatten_context(sample["context"])

    # Select top-k by LinUCB score
    selected_titles = reranker.select(question, titles, sentences_list, k)

    # Build context from selected titles
    if selected_titles:
        selected_paragraphs = [
            paragraphs[titles.index(t)] for t in selected_titles
        ]
        prompt = build_retrieval_prompt(question, selected_paragraphs)
    else:
        # Fallback: no context selected (shouldn't happen)
        prompt = f"""Answer the HotpotQA question.

Rules:
- Output only the final answer.
- If the answer is yes, no or noanswer, output exactly: yes or no or noanswer.
- Otherwise output a short span or entity name only.
- Do not include any explanation.
- Do not repeat the question.

Question: {question}

Answer:"""

    answer, input_tokens, output_tokens, cost_usd = call_llm(prompt, model)
    return answer, input_tokens, output_tokens, cost_usd, selected_titles


def _process_sample(
    sample_idx: int,
    sample: dict[str, Any],
    model: str,
    reranker: LinUCBReranker,
) -> tuple[int, str, list[ResultDict], list[str]]:
    """Process all k values for a single sample."""
    qid = sample["id"]
    sample_results: list[ResultDict] = []
    statuses: list[str] = []

    for k in LINUCB_K_VALUES:
        try:
            (
                pred,
                input_tokens,
                output_tokens,
                cost_usd,
                selected_titles,
            ) = _run_linucb_inference(sample, model, reranker, k)

            sample_results.append(
                _build_result(
                    sample=sample,
                    k=k,
                    pred=pred,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    selected_titles=selected_titles,
                )
            )
            statuses.append(f"{LINUCB_METHOD}(k={k})=OK")
        except Exception as e:
            statuses.append(f"{LINUCB_METHOD}(k={k})=ERROR: {e}")

    return sample_idx, qid, sample_results, statuses


def run_linucb(
    model: str,
    n_rows: int,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> None:
    """Run the LinUCB reranker evaluation for a model."""
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    print("\n" + "=" * 70)
    print(f"Running LinUCB: {model} (n={n_rows}, workers={max_workers})")
    print(f"K values: {LINUCB_K_VALUES}")
    print("=" * 70)

    # Load model
    print("\nLoading LinUCB model...")
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train the reranker first:")
        print("  uv run python -m experiments.train_linucb")
        sys.exit(1)

    reranker = LinUCBReranker()
    reranker.load(str(MODEL_PATH))
    print(f"Loaded model: {reranker.t:,} training observations, d={reranker.d}")

    # Load validation dataset
    print("\nLoading HotpotQA validation split...")
    ds = load_dataset_cached(
        dataset_name=DATASET_CONFIG["dataset_name"],
        subset=DATASET_CONFIG["subset"],
        split="validation",
        n_rows=n_rows,
    )
    print(f"Loaded {len(ds)} validation samples")

    csv_path = RESULTS_DIR / f"linucb_{model}.csv"
    print("Rewriting results CSV from scratch for this run")

    worker_count = min(max_workers, len(ds)) if ds else 1
    print(f"Processing samples with {worker_count} threads...")

    results_by_sample: list[list[ResultDict] | None] = [None] * len(ds)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(_process_sample, i, sample, model, reranker)
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
    print(
        f"Total rows: {len(all_results)} "
        f"({len(all_results) // len(LINUCB_K_VALUES)} samples × {len(LINUCB_K_VALUES)} k values)"
    )
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: uv run python -m experiments.run_linucb "
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

    run_linucb(model, n_rows, max_workers=max_workers)
