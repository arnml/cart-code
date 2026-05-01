"""Noise-gate ablation runner for CART paper.

Runs the noise gate over a grid of cosine-similarity thresholds and Jaccard
redundancy thresholds, then writes per-sample results to CSV. The regular
noise-gate run already covers the base Jaccard threshold 0.65, so this runner
only evaluates additional Jaccard settings.

Usage from root:
    uv run python -m experiments.run_noise_gate_ablation gpt-5.4-mini 100
    uv run python -m experiments.run_noise_gate_ablation gpt-5.4-mini 100 20
"""

from __future__ import annotations

import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from experiments.baselines_config import DATASET_CONFIG, LLM_TO_EMBEDDING, MODELS
from experiments.cache_dataset import load_dataset_cached
from experiments.eval_utils import evaluate_sample
from experiments.preprocessing import (
    build_always_think_prompt,
    build_retrieval_prompt,
    flatten_context,
)
from experiments.run_noise_gate import (
    NOISE_GATE_METHOD,
    NOISE_GATE_SIM_THRESHOLDS,
    call_llm,
    noise_gate_select,
)

RESULTS_DIR = Path(__file__).parent / "results" / "cart"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

DEFAULT_MAX_WORKERS = 20
NOISE_GATE_JACCARD_THRESHOLDS = (0.05, 0.20, 0.35, 0.50, 0.80, 0.95)

ResultDict = dict[str, str | int | float]

FIELDNAMES = [
    "question_id",
    "method",
    "threshold",
    "jaccard",
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


def _build_result(
    sample: dict[str, Any],
    sim_threshold: float,
    jaccard_threshold: float,
    pred: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> ResultDict:
    """Build a result row and attach evaluation metrics."""
    result: ResultDict = {
        "question_id": sample["id"],
        "method": NOISE_GATE_METHOD,
        "threshold": sim_threshold,
        "jaccard": jaccard_threshold,
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


def save_csv(model: str, results: list[ResultDict]) -> None:
    """Save per-record results to a CSV file."""
    csv_path = RESULTS_DIR / f"results_ablation_noise_gate_{model}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)


def _load_completed_keys(csv_path: Path) -> set[tuple[str, float, float]]:
    """Load completed (question, tau, rho) keys for resumable runs."""
    if not csv_path.exists():
        return set()

    completed: set[tuple[str, float, float]] = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                completed.add(
                    (
                        row["question_id"],
                        float(row["threshold"]),
                        float(row["jaccard"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return completed


def _append_results(csv_path: Path, results: list[ResultDict], lock: Lock) -> None:
    """Append result rows, creating the CSV header if needed."""
    if not results:
        return

    with lock:
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerows(results)


def _run_noise_gate_ablation_method(
    sample: dict[str, Any],
    model: str,
    sim_threshold: float,
    jaccard_threshold: float,
) -> tuple[str, int, int, float]:
    """Run noise-gate retrieval and generation for one threshold pair."""
    question = sample["question"]

    if model not in LLM_TO_EMBEDDING:
        raise KeyError(f"Unknown model for embeddings: {model}")

    paragraphs = flatten_context(sample["context"])
    emb_config = LLM_TO_EMBEDDING[model]
    selected_docs = noise_gate_select(
        question=question,
        paragraphs=paragraphs,
        provider=emb_config["provider"],
        embedding_model=emb_config["embedding_model"],
        token_budget=emb_config["max_tokens"],
        sim_threshold=sim_threshold,
        jac_threshold=jaccard_threshold,
    )

    if selected_docs:
        prompt = build_retrieval_prompt(question, selected_docs)
    else:
        prompt = build_always_think_prompt(question)

    answer, input_tokens, output_tokens, cost_usd = call_llm(prompt, model)
    return answer, input_tokens, output_tokens, cost_usd


def _build_prompt_for_threshold_pair(
    sample: dict[str, Any],
    model: str,
    sim_threshold: float,
    jaccard_threshold: float,
) -> str:
    """Build the prompt for one threshold pair without calling the LLM."""
    question = sample["question"]

    if model not in LLM_TO_EMBEDDING:
        raise KeyError(f"Unknown model for embeddings: {model}")

    paragraphs = flatten_context(sample["context"])
    emb_config = LLM_TO_EMBEDDING[model]
    selected_docs = noise_gate_select(
        question=question,
        paragraphs=paragraphs,
        provider=emb_config["provider"],
        embedding_model=emb_config["embedding_model"],
        token_budget=emb_config["max_tokens"],
        sim_threshold=sim_threshold,
        jac_threshold=jaccard_threshold,
    )

    if selected_docs:
        return build_retrieval_prompt(question, selected_docs)
    return build_always_think_prompt(question)


def _process_sample(
    sample_idx: int,
    sample: dict[str, Any],
    model: str,
    pending_pairs: list[tuple[float, float]],
) -> tuple[int, str, list[ResultDict], list[str]]:
    """Process pending threshold pairs for one sample.

    Several Jaccard thresholds can produce the same selected context. The
    prompt-level cache avoids repeated LLM calls for those duplicate prompts.
    """
    qid = sample["id"]
    prompt_to_pairs: dict[str, list[tuple[float, float]]] = {}
    statuses: list[str] = []

    for sim_threshold, jaccard_threshold in pending_pairs:
        try:
            prompt = _build_prompt_for_threshold_pair(
                sample=sample,
                model=model,
                sim_threshold=sim_threshold,
                jaccard_threshold=jaccard_threshold,
            )
            prompt_to_pairs.setdefault(prompt, []).append((sim_threshold, jaccard_threshold))
        except Exception as e:
            statuses.append(
                f"{NOISE_GATE_METHOD}(th={sim_threshold:g}, "
                f"jac={jaccard_threshold:g})=ERROR: {e}"
            )

    sample_results: list[ResultDict] = []
    llm_calls = 0
    reused_rows = 0
    for prompt, pairs in prompt_to_pairs.items():
        try:
            answer, input_tokens, output_tokens, cost_usd = call_llm(prompt, model)
            llm_calls += 1
            reused_rows += max(0, len(pairs) - 1)
            for sim_threshold, jaccard_threshold in pairs:
                sample_results.append(
                    _build_result(
                        sample=sample,
                        sim_threshold=sim_threshold,
                        jaccard_threshold=jaccard_threshold,
                        pred=answer,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost_usd=cost_usd,
                    )
                )
        except Exception as e:
            pair_text = ", ".join(
                f"th={sim_threshold:g}/jac={jaccard_threshold:g}"
                for sim_threshold, jaccard_threshold in pairs
            )
            statuses.append(f"{NOISE_GATE_METHOD}({pair_text})=ERROR: {e}")

    statuses.append(
        f"saved={len(sample_results)}, llm_calls={llm_calls}, reused={reused_rows}"
    )
    return sample_idx, qid, sample_results, statuses


def run_noise_gate_ablation(
    model: str,
    n_rows: int,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> None:
    """Run the noise-gate ablation for a model."""
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    print("\n" + "=" * 70)
    print(f"Running noise-gate ablation: {model} (n={n_rows}, workers={max_workers})")
    print(
        "Thresholds: "
        f"{', '.join(str(t) for t in NOISE_GATE_SIM_THRESHOLDS)}; "
        f"Jaccard: {', '.join(str(t) for t in NOISE_GATE_JACCARD_THRESHOLDS)}"
    )
    print("=" * 70)

    print("Loading HotpotQA...")
    ds = load_dataset_cached(
        dataset_name=DATASET_CONFIG["dataset_name"],
        subset=DATASET_CONFIG["subset"],
        split=DATASET_CONFIG["split"],
        n_rows=n_rows,
    )
    print(f"Loaded {len(ds)} samples")

    csv_path = RESULTS_DIR / f"results_ablation_noise_gate_{model}.csv"
    completed = _load_completed_keys(csv_path)
    if completed:
        print(f"Resuming from existing CSV with {len(completed)} completed rows")
    else:
        print("Starting a new incremental results CSV")

    tasks: list[tuple[int, dict[str, Any], list[tuple[float, float]]]] = []
    for sample_idx, sample in enumerate(ds):
        qid = sample["id"]
        pending_pairs: list[tuple[float, float]] = []
        for jaccard_threshold in NOISE_GATE_JACCARD_THRESHOLDS:
            for sim_threshold in NOISE_GATE_SIM_THRESHOLDS:
                key = (qid, float(sim_threshold), float(jaccard_threshold))
                if key not in completed:
                    pending_pairs.append((sim_threshold, jaccard_threshold))
        if pending_pairs:
            tasks.append((sample_idx, sample, pending_pairs))

    pending_rows = sum(len(pending_pairs) for _, _, pending_pairs in tasks)
    if pending_rows == 0:
        print("No pending rows; CSV is already complete for this grid")
        return

    worker_count = min(max_workers, len(tasks))
    print(
        f"Processing {pending_rows} pending rows across {len(tasks)} samples "
        f"with {worker_count} threads..."
    )

    csv_lock = Lock()
    completed_count = len(completed)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = []
        for sample_idx, sample, pending_pairs in tasks:
            futures.append(
                executor.submit(
                    _process_sample,
                    sample_idx,
                    sample,
                    model,
                    pending_pairs,
                )
            )

        for future in as_completed(futures):
            sample_idx, qid, sample_results, statuses = future.result()
            if sample_results:
                _append_results(csv_path, sample_results, csv_lock)
                completed_count += len(sample_results)
            print(
                f"  [{sample_idx + 1}/{len(ds)}] {qid}: "
                f"{', '.join(statuses)}"
            )

    print("\n" + "=" * 70)
    print(f"Complete! Saved {completed_count} rows to: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: uv run python -m experiments.run_noise_gate_ablation "
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

    run_noise_gate_ablation(model, n_rows, max_workers=max_workers)
