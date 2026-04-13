"""2WikiMultiHopQA baseline evaluation.

Usage from root:
    uv run python -m experiments.2wiki.run_baseline 10
    uv run python -m experiments.2wiki.run_baseline 10 8
    uv run python -m experiments.2wiki.run_baseline 10 --split validation
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from datasets import load_from_disk

from experiments.baselines import call_llm
from experiments.baselines_config import LLM_TO_EMBEDDING, MODELS
from experiments.embedding_utils import retrieve_top_k
from experiments.eval_utils import evaluate_sample

from .utils import (
    CACHE_DIR,
    build_always_think_prompt,
    build_retrieval_prompt,
    flatten_context,
    get_answer,
    get_question,
    get_raw_context,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "baseline"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_MAX_WORKERS = 20
DEFAULT_SPLIT = "validation"

METHODS = (
    "always_think",
    "retrieval_k3",
    "retrieval_k5",
    "retrieval_k10",
)

RETRIEVAL_METHOD_TO_K = {
    "retrieval_k3": 3,
    "retrieval_k5": 5,
    "retrieval_k10": 10,
}

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


def _build_result(
    sample: dict[str, Any],
    method_name: str,
    pred: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> ResultDict:
    result: ResultDict = {
        "question_id": sample["id"],
        "method": method_name,
        "answer_pred": pred,
        "answer_gt": get_answer(sample),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
    }

    metrics = evaluate_sample(pred, get_answer(sample))
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
    qid = sample["id"]
    question = get_question(sample)
    sample_results: list[ResultDict] = []
    statuses: list[str] = []

    top_paragraphs: list[str] | None = None
    try:
        paragraphs = flatten_context(get_raw_context(sample))
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
            else:
                if top_paragraphs is None:
                    raise RuntimeError("retrieval ranking failed for this sample")
                k = RETRIEVAL_METHOD_TO_K[method_name]
                prompt = build_retrieval_prompt(question, top_paragraphs[:k])

            pred, input_tokens, output_tokens, cost_usd = call_llm(prompt, model)
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


def save_csv(model: str, split: str, results: list[ResultDict]) -> None:
    csv_path = RESULTS_DIR / f"baseline_{model}_2wiki_{split}.csv"

    fieldnames = [
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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def run_baseline(
    n_rows: int,
    max_workers: int = DEFAULT_MAX_WORKERS,
    model: str = DEFAULT_MODEL,
    split: str = DEFAULT_SPLIT,
) -> None:
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    if split not in {"train", "validation"}:
        raise ValueError("split must be either 'train' or 'validation'")

    print("\n" + "=" * 70)
    print(f"Running 2Wiki baseline: {model} (n={n_rows}, workers={max_workers}, split={split})")
    print("=" * 70)

    print("Loading 2WikiMultiHopQA...")
    ds = _load_dataset_cached(split=split, n_rows=n_rows)
    print(f"Loaded {len(ds)} samples")

    csv_path = RESULTS_DIR / f"baseline_{model}_2wiki_{split}.csv"
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
    parser = argparse.ArgumentParser(description="Run the 2WikiMultiHopQA baseline.")
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
    run_baseline(
        n_rows=args.n_rows,
        max_workers=args.max_workers,
        model=args.model,
        split=args.split,
    )


if __name__ == "__main__":
    main()
