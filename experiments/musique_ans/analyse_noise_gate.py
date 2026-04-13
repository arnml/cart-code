"""Analyze and summarize MuSiQue-Ans noise-gate results.

Reads CSV from ``experiments/musique_ans/results/noise_gate/`` and produces a
summary report with aggregated metrics by threshold.

Usage from root:
    uv run python -m experiments.musique_ans.analyse_noise_gate gpt-5.4-mini
    uv run python -m experiments.musique_ans.analyse_noise_gate gpt-5.4-mini validation
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parent / "results" / "noise_gate"


def load_csv(model: str, split: str) -> tuple[list[dict[str, str]], Path]:
    """Load noise-gate results CSV."""
    csv_path = RESULTS_DIR / f"results_noise_gate_{model}_{split}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {csv_path}\n"
            f"Run: uv run python -m experiments.musique_ans.run_noise_gate {model} <n_rows> [max_workers]"
        )

    results: list[dict[str, str]] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)

    return results, csv_path


def aggregate_by_threshold(results: list[dict[str, str]]) -> dict[float, dict[str, float]]:
    """Aggregate metrics by threshold."""
    by_threshold: dict[float, list[dict[str, str]]] = defaultdict(list)

    for row in results:
        threshold_raw = row.get("threshold", "").strip()
        if not threshold_raw:
            raise ValueError("Missing threshold field in results CSV")
        by_threshold[float(threshold_raw)].append(row)

    aggregated: dict[float, dict[str, float]] = {}
    for threshold, rows in sorted(by_threshold.items()):
        n = len(rows)
        if n == 0:
            continue

        input_tokens = [int(r["input_tokens"]) for r in rows]
        output_tokens = [int(r["output_tokens"]) for r in rows]
        total_tokens = [i + o for i, o in zip(input_tokens, output_tokens, strict=True)]
        costs = [float(r["cost_usd"]) for r in rows]
        ems = [float(r["em"]) for r in rows]
        f1s = [float(r["f1"]) for r in rows]
        precisions = [float(r["precision"]) for r in rows]
        recalls = [float(r["recall"]) for r in rows]

        aggregated[threshold] = {
            "count": n,
            "input_mean": sum(input_tokens) / n,
            "output_mean": sum(output_tokens) / n,
            "total_tokens_mean": sum(total_tokens) / n,
            "cost_mean": sum(costs) / n,
            "em_mean": sum(ems) / n,
            "f1_mean": sum(f1s) / n,
            "precision_mean": sum(precisions) / n,
            "recall_mean": sum(recalls) / n,
        }

    if not aggregated:
        raise ValueError("No threshold groups found in noise-gate results")

    return aggregated


def generate_summary(model: str, split: str, aggregated: dict[float, dict[str, float]]) -> str:
    """Generate a single markdown table with thresholds as columns."""
    thresholds = sorted(aggregated.keys())
    header_cells = ["Metric"] + [f"{threshold:g}" for threshold in thresholds]
    lines = [
        f"# MuSiQue-Ans Noise-Gate Analysis: {model} ({split})\n",
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join("---" for _ in header_cells) + " |",
    ]

    metric_rows = [
        ("Count", lambda metrics: f"{int(metrics['count'])}"),
        ("EM", lambda metrics: f"{metrics['em_mean']:.3f}"),
        ("F1", lambda metrics: f"{metrics['f1_mean']:.3f}"),
        ("Precision", lambda metrics: f"{metrics['precision_mean']:.3f}"),
        ("Recall", lambda metrics: f"{metrics['recall_mean']:.3f}"),
        ("Input Tokens (mean)", lambda metrics: f"{metrics['input_mean']:.1f}"),
        ("Output Tokens (mean)", lambda metrics: f"{metrics['output_mean']:.1f}"),
        ("Total Tokens (mean)", lambda metrics: f"{metrics['total_tokens_mean']:.1f}"),
        ("Cost (mean, USD)", lambda metrics: f"{metrics['cost_mean']:.2e}"),
    ]

    for label, formatter in metric_rows:
        row = [label]
        for threshold in thresholds:
            row.append(formatter(aggregated[threshold]))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def save_summary(model: str, split: str, summary: str) -> None:
    """Save summary to markdown file."""
    md_path = RESULTS_DIR / f"analysis_noise_gate_{model}_{split}.md"
    md_path.write_text(summary, encoding="utf-8")
    print(f"Summary saved: {md_path}")


def analyse_noise_gate(model: str, split: str = "validation") -> None:
    """Main analysis function."""
    print(f"Analyzing MuSiQue-Ans noise-gate results for: {model} ({split})")

    results, csv_path = load_csv(model, split)
    print(f"Loaded {len(results)} results from {csv_path}")

    aggregated = aggregate_by_threshold(results)
    print(f"Found {len(aggregated)} threshold groups")

    summary = generate_summary(model, split, aggregated)
    print("\n" + summary)

    save_summary(model, split, summary)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: uv run python -m experiments.musique_ans.analyse_noise_gate <model> [split]"
        )
        print("\nExample:")
        print("  uv run python -m experiments.musique_ans.analyse_noise_gate gpt-5.4-mini")
        print(
            "  uv run python -m experiments.musique_ans.analyse_noise_gate gpt-5.4-mini validation"
        )
        sys.exit(1)

    model = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) >= 3 else "validation"

    try:
        analyse_noise_gate(model, split=split)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
