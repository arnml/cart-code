"""Analyze and summarize noise-gate results.

Reads CSV from experiments/results/cart/results_noise_gate_MODEL.csv and
produces a summary report with aggregated metrics by threshold.

Usage from root:
    uv run python -m experiments.analyse_noise_gate gpt-5.4-mini
    uv run python -m experiments.analyse_noise_gate claude-sonnet-4-6
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


def load_csv(model: str) -> tuple[list[dict[str, str]], Path]:
    """Load noise-gate results CSV."""
    results_dir = Path(__file__).parent / "results" / "cart"
    csv_path = results_dir / f"results_noise_gate_{model}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {csv_path}\n"
            f"Run: uv run python -m experiments.run_noise_gate {model} <n_rows>"
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


def generate_summary(model: str, aggregated: dict[float, dict[str, float]]) -> str:
    """Generate a single markdown table with thresholds as columns."""
    thresholds = sorted(aggregated.keys())
    header_cells = ["Metric"] + [f"{threshold:g}" for threshold in thresholds]
    lines = [
        f"# Noise-Gate Analysis: {model}\n",
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


def save_summary(model: str, summary: str) -> None:
    """Save summary to markdown file."""
    results_dir = Path(__file__).parent / "results" / "cart"
    md_path = results_dir / f"analysis_noise_gate_{model}.md"

    md_path.write_text(summary, encoding="utf-8")
    print(f"Summary saved: {md_path}")


def analyse_noise_gate(model: str) -> None:
    """Main analysis function."""
    print(f"Analyzing noise-gate results for: {model}")

    results, csv_path = load_csv(model)
    print(f"Loaded {len(results)} results from {csv_path}")

    aggregated = aggregate_by_threshold(results)
    print(f"Found {len(aggregated)} threshold groups")

    summary = generate_summary(model, aggregated)
    print("\n" + summary)

    save_summary(model, summary)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m experiments.analyse_noise_gate <model>")
        print("\nExample:")
        print("  uv run python -m experiments.analyse_noise_gate gpt-5.4-mini")
        print("  uv run python -m experiments.analyse_noise_gate claude-sonnet-4-6")
        sys.exit(1)

    model = sys.argv[1]

    try:
        analyse_noise_gate(model)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
