"""Analyze and summarize noise-gate ablation results.

Reads the regular noise-gate CSV for the base Jaccard threshold (0.65) and the
ablation CSV for the tighter Jaccard thresholds (0.55 and 0.50), then produces
two markdown tables:

- F1 by Jaccard threshold versus similarity threshold
- Mean total tokens by Jaccard threshold versus similarity threshold

Usage from root:
    uv run python -m experiments.analyse_ablation_noise_gate gpt-5.4-mini
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

BASE_JACCARD_THRESHOLD = 0.65


def _read_csv(csv_path: Path, default_jaccard: float | None = None) -> list[dict[str, str]]:
    """Read a results CSV and ensure each row has a jaccard value."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")

    results: list[dict[str, str]] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            threshold_raw = row.get("threshold", "").strip()
            if not threshold_raw:
                raise ValueError(f"Missing threshold field in {csv_path}")

            if default_jaccard is not None:
                row["jaccard"] = row.get("jaccard", "").strip() or f"{default_jaccard:g}"
            else:
                jaccard_raw = row.get("jaccard", "").strip()
                if not jaccard_raw:
                    raise ValueError(f"Missing jaccard field in {csv_path}")
                row["jaccard"] = jaccard_raw

            results.append(row)

    return results


def load_results(model: str) -> tuple[list[dict[str, str]], tuple[Path, Path]]:
    """Load the base noise-gate CSV plus the ablation CSV."""
    results_dir = Path(__file__).parent / "results" / "cart"
    base_csv_path = results_dir / f"results_noise_gate_{model}.csv"
    ablation_csv_path = results_dir / f"results_ablation_noise_gate_{model}.csv"

    missing = [path for path in (base_csv_path, ablation_csv_path) if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Results file(s) not found:\n{missing_text}\n"
            f"Run: uv run python -m experiments.run_noise_gate {model} <n_rows>\n"
            f"Run: uv run python -m experiments.run_noise_gate_ablation {model} <n_rows>"
        )

    results = _read_csv(base_csv_path, default_jaccard=BASE_JACCARD_THRESHOLD)
    results.extend(_read_csv(ablation_csv_path))
    return results, (base_csv_path, ablation_csv_path)


def aggregate_results(results: list[dict[str, str]]) -> dict[float, dict[float, dict[str, float]]]:
    """Aggregate the key metrics by Jaccard and similarity threshold."""
    grouped: dict[float, dict[float, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))

    for row in results:
        jaccard_raw = row.get("jaccard", "").strip()
        threshold_raw = row.get("threshold", "").strip()
        if not jaccard_raw or not threshold_raw:
            raise ValueError("Missing jaccard or threshold field in results CSV")

        grouped[float(jaccard_raw)][float(threshold_raw)].append(row)

    aggregated: dict[float, dict[float, dict[str, float]]] = {}
    for jaccard, threshold_map in grouped.items():
        aggregated[jaccard] = {}
        for threshold, rows in threshold_map.items():
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

            aggregated[jaccard][threshold] = {
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
        raise ValueError("No jaccard/threshold groups found in noise-gate results")

    return aggregated


def _render_table(
    aggregated: dict[float, dict[float, dict[str, float]]],
    metric_key: str,
    section_title: str,
    description: str,
    value_format: str,
) -> list[str]:
    """Render one Jaccard-vs-threshold table."""
    jaccard_values = sorted(aggregated.keys())
    threshold_values = sorted({threshold for row in aggregated.values() for threshold in row})

    header_cells = ["Jaccard \\ Threshold"] + [f"{threshold:g}" for threshold in threshold_values]
    lines = [
        f"## {section_title}",
        "",
        description,
        "",
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join("---" for _ in header_cells) + " |",
    ]

    for jaccard in jaccard_values:
        row = [f"{jaccard:g}"]
        for threshold in threshold_values:
            metrics = aggregated[jaccard].get(threshold)
            if metrics is None:
                row.append("n/a")
            else:
                row.append(value_format.format(value=metrics[metric_key]))
        lines.append("| " + " | ".join(row) + " |")

    return lines


def generate_summary(model: str, aggregated: dict[float, dict[float, dict[str, float]]]) -> str:
    """Generate the markdown report."""
    lines = [f"# Noise-Gate Ablation Analysis: {model}", ""]

    lines.extend(
        _render_table(
            aggregated=aggregated,
            metric_key="f1_mean",
            section_title="F1",
            description="F1 values averaged across samples.",
            value_format="{value:.3f}",
        )
    )
    lines.append("")
    lines.extend(
        _render_table(
            aggregated=aggregated,
            metric_key="total_tokens_mean",
            section_title="Total Tokens (mean)",
            description="Mean total tokens (input + output) averaged across samples.",
            value_format="{value:.1f}",
        )
    )

    return "\n".join(lines).rstrip() + "\n"


def save_summary(model: str, summary: str) -> None:
    """Save summary to markdown file."""
    results_dir = Path(__file__).parent / "results" / "cart"
    md_path = results_dir / f"analysis_ablation_noise_gate_{model}.md"

    md_path.write_text(summary, encoding="utf-8")
    print(f"Summary saved: {md_path}")


def analyse_ablation_noise_gate(model: str) -> None:
    """Main analysis function."""
    print(f"Analyzing noise-gate ablation results for: {model}")

    results, csv_paths = load_results(model)
    print(f"Loaded {len(results)} combined results from:")
    for csv_path in csv_paths:
        print(f"  - {csv_path}")

    aggregated = aggregate_results(results)
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
