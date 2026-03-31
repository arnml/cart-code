"""
Analyze and summarize adaptive-k results.

Reads CSV from experiments/results/cart/results_adaptive_k_MODEL.csv and produces
a summary report with aggregated metrics and k_star distribution analysis.

Usage from root:
    uv run python -m experiments.analyse_adaptive_k gpt-4o-mini
    uv run python -m experiments.analyse_adaptive_k claude-sonnet-4-6
"""

import csv
import sys
import math
import statistics
from pathlib import Path


def load_csv(model: str) -> tuple[list[dict[str, str]], Path]:
    """Load adaptive-k results CSV.

    Args:
        model: Model name

    Returns:
        Tuple of (list of result dicts, csv path)

    Raises:
        FileNotFoundError: If CSV doesn't exist
    """
    results_dir = Path(__file__).parent / "results" / "cart"
    csv_path = results_dir / f"results_adaptive_k_{model}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {csv_path}\n"
            f"Run: uv run python -m experiments.run_adaptive_k {model} <n_rows>"
        )

    results = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)

    return results, csv_path


def aggregate_metrics(results: list[dict[str, str]]) -> dict[str, float]:
    """Aggregate all metrics across all results.

    Args:
        results: List of result dictionaries from CSV

    Returns:
        Dict of aggregated metrics

    Raises:
        ValueError: If k_star field is missing or invalid
    """
    n = len(results)

    input_tokens = [int(r["input_tokens"]) for r in results]
    output_tokens = [int(r["output_tokens"]) for r in results]
    total_tokens = [i + o for i, o in zip(input_tokens, output_tokens)]
    costs = [float(r["cost_usd"]) for r in results]
    ems = [float(r["em"]) for r in results]
    f1s = [float(r["f1"]) for r in results]
    precisions = [float(r["precision"]) for r in results]
    recalls = [float(r["recall"]) for r in results]

    # Parse k_star values (may be missing for old results)
    k_stars = []
    for r in results:
        k_star_val = r.get("k_star", "").strip()
        if k_star_val:
            k_stars.append(int(k_star_val))

    # Calculate derived metrics: token_efficiency and cost_efficiency
    f1_mean = sum(f1s) / n
    total_tokens_mean = sum(total_tokens) / n
    cost_mean = sum(costs) / n

    # Token Efficiency: F1 / log(1 + total_tokens)
    token_eff = f1_mean / math.log(1 + total_tokens_mean) if total_tokens_mean >= 0 else 0

    # Cost Efficiency: F1 / cost_usd
    cost_eff = f1_mean / cost_mean if cost_mean > 0 else 0

    aggregated = {
        "count": n,
        "input_mean": sum(input_tokens) / n,
        "output_mean": sum(output_tokens) / n,
        "total_tokens_mean": total_tokens_mean,
        "cost_mean": cost_mean,
        "em_mean": sum(ems) / n,
        "f1_mean": f1_mean,
        "precision_mean": sum(precisions) / n,
        "recall_mean": sum(recalls) / n,
        "token_eff_mean": token_eff,
        "cost_eff_mean": cost_eff,
    }

    # Add k_star statistics if available
    if k_stars:
        aggregated["k_star_count"] = len(k_stars)
        aggregated["k_star_min"] = min(k_stars)
        aggregated["k_star_max"] = max(k_stars)
        aggregated["k_star_mean"] = statistics.mean(k_stars)
        aggregated["k_star_median"] = statistics.median(k_stars)
        if len(k_stars) > 1:
            aggregated["k_star_stdev"] = statistics.stdev(k_stars)
        else:
            aggregated["k_star_stdev"] = 0.0

    return aggregated


def generate_summary(model: str, aggregated: dict[str, float]) -> str:
    """Generate markdown summary.

    Args:
        model: Model name
        aggregated: Dict of aggregated metrics

    Returns:
        Markdown formatted summary
    """
    lines = [
        f"# Adaptive-K Analysis: {model}\n",
        "## Overall Metrics\n",
        "| Metric | Value |",
        "|--------|-------|",
    ]

    # Performance metrics
    lines.append(f"| Count | {int(aggregated['count'])} |")
    lines.append(f"| EM | {aggregated['em_mean']:.3f} |")
    lines.append(f"| F1 | {aggregated['f1_mean']:.3f} |")
    lines.append(f"| Precision | {aggregated['precision_mean']:.3f} |")
    lines.append(f"| Recall | {aggregated['recall_mean']:.3f} |")

    # Token metrics
    lines.append(f"| Input Tokens (mean) | {aggregated['input_mean']:.1f} |")
    lines.append(f"| Output Tokens (mean) | {aggregated['output_mean']:.1f} |")
    lines.append(f"| Total Tokens (mean) | {aggregated['total_tokens_mean']:.1f} |")

    # Cost metrics
    lines.append(f"| Cost (mean, USD) | {aggregated['cost_mean']:.2e} |")

    # Efficiency metrics
    lines.append(f"| Token Efficiency (F1/log tokens) | {aggregated['token_eff_mean']:.4f} |")
    lines.append(f"| Cost Efficiency (F1/cost) | {aggregated['cost_eff_mean']:.2f} |")

    # K* statistics
    if "k_star_mean" in aggregated:
        lines.append("\n## Adaptive-K Selection (k*) Statistics\n")
        lines.append("| Statistic | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| Count | {int(aggregated['k_star_count'])} |")
        lines.append(f"| Min | {int(aggregated['k_star_min'])} |")
        lines.append(f"| Max | {int(aggregated['k_star_max'])} |")
        lines.append(f"| Mean | {aggregated['k_star_mean']:.2f} |")
        lines.append(f"| Median | {int(aggregated['k_star_median'])} |")
        lines.append(f"| Stdev | {aggregated['k_star_stdev']:.2f} |")

    return "\n".join(lines) + "\n"


def save_summary(model: str, summary: str) -> None:
    """Save summary to markdown file.

    Args:
        model: Model name
        summary: Markdown summary text
    """
    results_dir = Path(__file__).parent / "results" / "cart"
    md_path = results_dir / f"analysis_adaptive_k_{model}.md"

    md_path.write_text(summary, encoding="utf-8")
    print(f"Summary saved: {md_path}")


def analyse_adaptive_k(model: str) -> None:
    """Main analysis function.

    Args:
        model: Model name

    Raises:
        FileNotFoundError: If CSV doesn't exist
    """
    print(f"Analyzing adaptive-k results for: {model}")

    # Load CSV
    results, csv_path = load_csv(model)
    print(f"Loaded {len(results)} results from {csv_path}")

    # Aggregate metrics
    aggregated = aggregate_metrics(results)
    print(f"Aggregated metrics for {aggregated['count']} samples")

    # Generate summary
    summary = generate_summary(model, aggregated)

    # Print summary
    print("\n" + summary)

    # Save summary
    save_summary(model, summary)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m experiments.analyse_adaptive_k <model>")
        print("\nExample:")
        print("  uv run python -m experiments.analyse_adaptive_k gpt-4o-mini")
        print("  uv run python -m experiments.analyse_adaptive_k claude-sonnet-4-6")
        sys.exit(1)

    model = sys.argv[1]

    try:
        analyse_adaptive_k(model)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
