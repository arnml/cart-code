"""Analyze and summarize noise-gate ablation results.

Reads CSV from experiments/results/cart/results_ablation_noise_gate_MODEL.csv
and produces a markdown report with F1 values arranged by Jaccard threshold
versus similarity threshold.

Usage from root:
    uv run python -m experiments.analyse_ablation_noise_gate gpt-5.4-mini
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


def load_csv(model: str) -> tuple[list[dict[str, str]], Path]:
    """Load noise-gate ablation results CSV."""
    results_dir = Path(__file__).parent / "results" / "cart"
    csv_path = results_dir / f"results_ablation_noise_gate_{model}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {csv_path}\n"
            f"Run: uv run python -m experiments.run_noise_gate_ablation {model} <n_rows>"
        )

    results: list[dict[str, str]] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)

    return results, csv_path


def aggregate_f1_by_jaccard_and_threshold(
    results: list[dict[str, str]],
) -> dict[float, dict[float, float]]:
    """Aggregate mean F1 by Jaccard threshold and similarity threshold."""
    grouped: dict[float, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in results:
        jaccard_raw = row.get("jaccard", "").strip()
        threshold_raw = row.get("threshold", "").strip()
        if not jaccard_raw or not threshold_raw:
            raise ValueError("Missing jaccard or threshold field in results CSV")

        grouped[float(jaccard_raw)][float(threshold_raw)].append(float(row["f1"]))

    aggregated: dict[float, dict[float, float]] = {}
    for jaccard, threshold_map in grouped.items():
        aggregated[jaccard] = {}
        for threshold, f1_values in threshold_map.items():
            if not f1_values:
                continue
            aggregated[jaccard][threshold] = sum(f1_values) / len(f1_values)

    if not aggregated:
        raise ValueError("No jaccard/threshold groups found in noise-gate ablation results")

    return aggregated


def generate_summary(model: str, aggregated: dict[float, dict[float, float]]) -> str:
    """Generate a single markdown table with Jaccard as rows and thresholds as columns."""
    jaccard_values = sorted(aggregated.keys())
    threshold_values = sorted({threshold for row in aggregated.values() for threshold in row})

    header_cells = ["Jaccard \\ Threshold"] + [f"{threshold:g}" for threshold in threshold_values]
    lines = [
        f"# Noise-Gate Ablation Analysis: {model}\n",
        "F1 values averaged across samples.",
        "",
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join("---" for _ in header_cells) + " |",
    ]

    for jaccard in jaccard_values:
        row = [f"{jaccard:g}"]
        for threshold in threshold_values:
            value = aggregated[jaccard].get(threshold)
            row.append(f"{value:.3f}" if value is not None else "n/a")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def save_summary(model: str, summary: str) -> None:
    """Save summary to markdown file."""
    results_dir = Path(__file__).parent / "results" / "cart"
    md_path = results_dir / f"analysis_ablation_noise_gate_{model}.md"

    md_path.write_text(summary, encoding="utf-8")
    print(f"Summary saved: {md_path}")


def analyse_ablation_noise_gate(model: str) -> None:
    """Main analysis function."""
    print(f"Analyzing noise-gate ablation results for: {model}")

    results, csv_path = load_csv(model)
    print(f"Loaded {len(results)} results from {csv_path}")

    aggregated = aggregate_f1_by_jaccard_and_threshold(results)
    print(f"Found {len(aggregated)} jaccard groups")

    summary = generate_summary(model, aggregated)
    print("\n" + summary)

    save_summary(model, summary)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m experiments.analyse_ablation_noise_gate <model>")
        print("\nExample:")
        print("  uv run python -m experiments.analyse_ablation_noise_gate gpt-5.4-mini")
        print("  uv run python -m experiments.analyse_ablation_noise_gate claude-sonnet-4-6")
        sys.exit(1)

    model = sys.argv[1]

    try:
        analyse_ablation_noise_gate(model)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
