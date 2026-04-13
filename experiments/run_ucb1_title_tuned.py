"""Legacy UCB1-TUNED title-based reranker evaluation script.

This version preserves the older BM25 fallback behavior that was used in the
title-based UCB1 analysis and its `unseen_fallback` diagnostic.
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
from experiments.llm_anthropic import call_anthropic
from experiments.llm_openai import call_openai
from experiments.preprocessing import build_retrieval_prompt, flatten_context
from experiments.ucb1_title_tuned import UCB1TitleTunedReranker

RESULTS_DIR = Path(__file__).parent / "results" / "ucb1_title"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

SCOREBOARD_PATH = Path(__file__).parent / "cache" / "ucb1_title_scoreboard.json"
UCB1_K_VALUES = (2, 3, 5)
UCB1_METHOD = "ucb1_title"
DEFAULT_MAX_WORKERS = 20

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
    "unseen_fallback",
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
    unseen_fallback: bool,
) -> ResultDict:
    """Build a result row and attach evaluation metrics."""
    metrics = evaluate_sample(pred, sample["answer"])
    return {
        "question_id": sample["id"],
        "method": UCB1_METHOD,
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
        "unseen_fallback": unseen_fallback,
    }


def save_csv(model: str, results: list[ResultDict]) -> None:
    """Save per-record results to a CSV file."""
    csv_path = RESULTS_DIR / f"ucb1_title_{model}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)


def _run_ucb1_inference(
    sample: dict[str, Any],
    model: str,
    reranker: UCB1TitleTunedReranker,
    k: int,
) -> tuple[str, int, int, float, list[str], bool]:
    """Run UCB1 selection and LLM generation for one k value."""
    question = sample["question"]
    titles = sample["context"]["title"]
    paragraphs = flatten_context(sample["context"])

    reranker.set_bm25_fallback(question, titles)
    selected_titles, had_unseen = reranker.select(titles, k, question=question)

    if selected_titles:
        selected_indices = [titles.index(t) for t in selected_titles]
        selected_paragraphs = [paragraphs[i] for i in selected_indices]
        prompt = build_retrieval_prompt(question, selected_paragraphs)
    else:
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
    return answer, input_tokens, output_tokens, cost_usd, selected_titles, had_unseen


def _process_sample(
    sample_idx: int,
    sample: dict[str, Any],
    model: str,
    reranker: UCB1TitleTunedReranker,
) -> tuple[int, str, list[ResultDict], list[str]]:
    """Process all k values for a single sample."""
    qid = sample["id"]
    sample_results: list[ResultDict] = []
    statuses: list[str] = []

    for k in UCB1_K_VALUES:
        try:
            (
                pred,
                input_tokens,
                output_tokens,
                cost_usd,
                selected_titles,
                had_unseen,
            ) = _run_ucb1_inference(sample, model, reranker, k)

            sample_results.append(
                _build_result(
                    sample=sample,
                    k=k,
                    pred=pred,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    selected_titles=selected_titles,
                    unseen_fallback=had_unseen,
                )
            )
            statuses.append(f"{UCB1_METHOD}(k={k})=OK")
        except Exception as e:
            statuses.append(f"{UCB1_METHOD}(k={k})=ERROR: {e}")

    return sample_idx, qid, sample_results, statuses


def run_ucb1_title_tuned(
    model: str,
    n_rows: int,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> None:
    """Run the legacy title-based UCB1-TUNED evaluation for a model."""
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    print("\n" + "=" * 70)
    print(f"Running legacy UCB1-TUNED (title-based): {model} (n={n_rows}, workers={max_workers})")
    print(f"K values: {UCB1_K_VALUES}")
    print("=" * 70)

    print("\nLoading scoreboard...")
    if not SCOREBOARD_PATH.exists():
        print(f"Error: Scoreboard not found at {SCOREBOARD_PATH}")
        print("Please train the reranker first:")
        print("  uv run python -m experiments.train_ucb1_title_tuned")
        sys.exit(1)

    reranker = UCB1TitleTunedReranker()
    reranker.load(str(SCOREBOARD_PATH))
    print(
        f"Loaded scoreboard: {reranker.t:,} observations, "
        f"{len(reranker.scoreboard):,} unique titles"
    )

    print("\nLoading HotpotQA validation split...")
    ds = load_dataset_cached(
        dataset_name=DATASET_CONFIG["dataset_name"],
        subset=DATASET_CONFIG["subset"],
        split=DATASET_CONFIG["split"],
        n_rows=n_rows,
    )
    print(f"Loaded {len(ds)} validation samples")

    csv_path = RESULTS_DIR / f"ucb1_title_{model}.csv"
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
        f"({len(all_results) // len(UCB1_K_VALUES)} samples × {len(UCB1_K_VALUES)} k values)"
    )
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: uv run python -m experiments.run_ucb1_title_tuned "
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

    run_ucb1_title_tuned(model, n_rows, max_workers=max_workers)
