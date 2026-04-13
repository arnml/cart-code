"""Analyze and summarize MuSiQue-Ans baseline results.

Reads CSV from ``experiments/musique_ans/results/baseline/`` and writes a
dataset-local markdown summary next to it.

Usage from root:
    uv run python -m experiments.musique_ans.analyse_baseline gpt-5.4-mini
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parent / "results" / "baseline"


def load_csv(model: str) -> tuple[list[dict[str, str]], Path]:
    csv_path = RESULTS_DIR / f"baseline_{model}_musique_ans_validation.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {csv_path}\n"
            f"Run: uv run python -m experiments.musique_ans.run_baseline <n_rows> [max_workers]"
        )

    results: list[dict[str, str]] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)

    return results, csv_path


def aggregate_by_method(results: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    by_method = defaultdict(list)
    by_method_unique = defaultdict(set)
    seen_pairs = set()

    for row in results:
        method = row["method"]
        question_id = row["question_id"]
        pair = (question_id, method)

        if pair in seen_pairs:
            raise ValueError(
                f"Duplicate entry found: question_id={question_id}, method={method}. "
                "Each question/method combination should appear exactly once."
            )
        seen_pairs.add(pair)

        by_method[method].append(row)
        by_method_unique[method].add(question_id)

    aggregated: dict[str, dict[str, float]] = {}
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


def generate_summary(model: str, aggregated: dict[str, dict[str, float]]) -> str:
    lines = [
        f"# MuSiQue-Ans Baseline Analysis: {model}\n",
        "## Summary by Method\n",
        "| Method | Count | input | output | total token | cost_usd | EM | F1 | Precision | Recall |",
        "|--------|-------|-------|--------|-------------|----------|----|----|-----------|--------|",
    ]

    for method in sorted(aggregated.keys()):
        metrics = aggregated[method]
        lines.append(
            f"| {method} | {metrics['count']} | "
            f"{metrics['input_mean']:.1f} | {metrics['output_mean']:.1f} | "
            f"{metrics['total_tokens_mean']:.1f} | {metrics['cost_mean']:.2e} | "
            f"{metrics['em_mean']:.3f} | {metrics['f1_mean']:.3f} | "
            f"{metrics['precision_mean']:.3f} | {metrics['recall_mean']:.3f} |"
        )

    return "\n".join(lines) + "\n"


def save_summary(model: str, summary: str) -> None:
    md_path = RESULTS_DIR / f"analysis_{model}_musique_ans_validation.md"
    md_path.write_text(summary, encoding="utf-8")
    print(f"Summary saved: {md_path}")


def analyse_baseline(model: str) -> None:
    print(f"Analyzing MuSiQue-Ans baseline results for: {model}")
    results, csv_path = load_csv(model)
    print(f"Loaded {len(results)} results from {csv_path}")
    aggregated = aggregate_by_method(results)
    print(f"Found {len(aggregated)} methods")
    summary = generate_summary(model, aggregated)
    print("\n" + summary)
    save_summary(model, summary)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m experiments.musique_ans.analyse_baseline <model>")
        print("\nExample:")
        print("  uv run python -m experiments.musique_ans.analyse_baseline gpt-5.4-mini")
        sys.exit(1)

    analyse_baseline(sys.argv[1])
