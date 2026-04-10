"""Analyze and summarize LinUCB reranker results.

Reads CSV from experiments/results/linucb/linucb_{MODEL}.csv and produces
a summary report with metrics by k value and overall performance.

Usage from root:
    uv run python -m experiments.analyse_linucb gpt-4o-mini
    uv run python -m experiments.analyse_linucb claude-sonnet-4-6
"""

import sys
from pathlib import Path

from experiments.utils import load_csv, aggregate_by_k, aggregate_metrics


def generate_summary(
    model: str,
    results: list[dict[str, str]],
    by_k: dict[int, dict[str, float]],
) -> str:
    """Generate markdown summary report.

    Args:
        model: Model name
        results: All result rows
        by_k: Aggregated metrics by k

    Returns:
        Markdown summary string
    """
    # Compute overall metrics
    overall = aggregate_metrics(results)

    lines = [
        f"# LinUCB Reranker Analysis: {model}",
        "",
        f"**Total samples:** {overall['count']}",
        "",
    ]

    # Section 1: K-value Performance Table
    lines.extend([
        "## Performance by K Value",
        "",
        "| K | Samples | EM | F1 | Precision | Recall | Avg Tokens |",
        "|---|---------|----|----|-----------|--------|------------|",
    ])

    for k in sorted(by_k.keys()):
        metrics = by_k[k]
        lines.append(
            f"| {k} | {metrics['count']:.0f} | "
            f"{metrics['em_mean']:.3f} | {metrics['f1_mean']:.3f} | "
            f"{metrics['precision_mean']:.3f} | {metrics['recall_mean']:.3f} | "
            f"{metrics['total_tokens_mean']:.0f} |"
        )

    lines.append("")

    # Section 2: Overall Metrics
    lines.extend([
        "## Overall Metrics",
        "",
        f"**Exact Match (EM):** {overall['em_mean']:.3f}",
        f"**F1 Score:** {overall['f1_mean']:.3f}",
        f"**Precision:** {overall['precision_mean']:.3f}",
        f"**Recall:** {overall['recall_mean']:.3f}",
        "",
        f"**Avg Input Tokens:** {overall['input_mean']:.1f}",
        f"**Avg Output Tokens:** {overall['output_mean']:.1f}",
        f"**Avg Total Tokens:** {overall['total_tokens_mean']:.1f}",
        f"**Avg Cost:** ${overall['cost_mean']:.4f}",
        "",
    ])

    return "\n".join(lines)


def save_summary(output_path: Path, summary: str) -> None:
    """Save markdown summary to file.

    Args:
        output_path: Path to save summary
        summary: Markdown content
    """
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)


def analyze_linucb(model: str) -> None:
    """Load CSV, aggregate by k, and generate summary report.

    Args:
        model: Model name (e.g., 'gpt-4o-mini', 'claude-sonnet-4-6')
    """
    results_dir = Path(__file__).parent / "results" / "linucb"
    csv_path = results_dir / f"linucb_{model}.csv"

    print(f"Loading LinUCB results for {model}...")
    try:
        results = load_csv(csv_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Run: uv run python -m experiments.run_linucb {model} <n_rows>")
        sys.exit(1)

    print(f"Loaded {len(results)} result rows")

    # Aggregate by k
    by_k = aggregate_by_k(results)

    # Generate summary
    summary = generate_summary(model, results, by_k)

    # Save summary
    output_path = results_dir / f"analysis_linucb_{model}.md"
    print(f"Saving analysis to {output_path}...")
    save_summary(output_path, summary)

    print("\nAnalysis complete!")
    print(summary)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m experiments.analyse_linucb <MODEL>")
        sys.exit(1)

    model = sys.argv[1]
    analyze_linucb(model)
