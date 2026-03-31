"""
Analyze and summarize baseline results.

Reads CSV from experiments/results/baseline/baseline_MODEL.csv and produces
a summary report with aggregated metrics by method.

Usage from root:
    uv run python -m experiments.analyse_baseline gpt-4o-mini
    uv run python -m experiments.analyse_baseline claude-sonnet-4-6
"""

import csv
import sys
import math
from pathlib import Path
from collections import defaultdict


def load_csv(model: str) -> tuple[list[dict[str, str]], Path]:
    """Load baseline results CSV.

    Args:
        model: Model name

    Returns:
        Tuple of (list of result dicts, csv path)

    Raises:
        FileNotFoundError: If CSV doesn't exist
    """
    results_dir = Path(__file__).parent / "results" / "baseline"
    csv_path = results_dir / f"baseline_{model}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {csv_path}\n"
            f"Run: uv run python -m experiments.run_baseline {model} <n_rows>"
        )

    results = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)

    return results, csv_path


def aggregate_by_method(results: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    """Aggregate metrics by method.

    Args:
        results: List of result dictionaries from CSV

    Returns:
        Dict mapping method name to aggregated metrics

    Raises:
        ValueError: If duplicate (question_id, method) pairs are found
    """
    by_method = defaultdict(list)
    by_method_unique = defaultdict(set)  # Track unique question_ids per method
    seen_pairs = set()

    for row in results:
        method = row["method"]
        question_id = row["question_id"]
        pair = (question_id, method)

        if pair in seen_pairs:
            raise ValueError(
                f"Duplicate entry found: question_id={question_id}, method={method}. "
                f"Each question/method combination should appear exactly once."
            )
        seen_pairs.add(pair)

        by_method[method].append(row)
        by_method_unique[method].add(question_id)

    aggregated = {}
    for method, rows in by_method.items():
        # Count unique question_ids, not total rows (in case of duplicates)
        n = len(by_method_unique[method])

        input_tokens = [int(r["input_tokens"]) for r in rows]
        output_tokens = [int(r["output_tokens"]) for r in rows]
        total_tokens = [i + o for i, o in zip(input_tokens, output_tokens)]
        costs = [float(r["cost_usd"]) for r in rows]
        ems = [float(r["em"]) for r in rows]
        f1s = [float(r["f1"]) for r in rows]
        precisions = [float(r["precision"]) for r in rows]
        recalls = [float(r["recall"]) for r in rows]

        # Calculate derived metrics: token_efficiency and cost_efficiency
        f1_mean = sum(f1s) / n
        total_tokens_mean = sum(total_tokens) / n
        cost_mean = sum(costs) / n

        # Token Efficiency: F1 / log(1 + total_tokens)
        token_eff = f1_mean / math.log(1 + total_tokens_mean) if total_tokens_mean >= 0 else 0

        # Cost Efficiency: F1 / cost_usd
        cost_eff = f1_mean / cost_mean if cost_mean > 0 else 0

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
            "token_eff_mean": token_eff,
            "cost_eff_mean": cost_eff,
        }

    return aggregated


def generate_summary(model: str, aggregated: dict[str, dict[str, float]]) -> str:
    """Generate markdown summary.

    Args:
        model: Model name
        aggregated: Dict of aggregated metrics by method

    Returns:
        Markdown formatted summary
    """
    lines = [
        f"# Baseline Analysis: {model}\n",
        "## Summary by Method\n",
        "| Method | Count | input | output | total token | cost_usd | EM | F1 | Precision | Recall | Token-Eff | Cost-Eff |",
        "|--------|-------|-------|--------|-------------|----------|----|----|-----------|--------|-----------|----------|",
    ]

    for method in sorted(aggregated.keys()):
        metrics = aggregated[method]
        line = (
            f"| {method} | {metrics['count']} | "
            f"{metrics['input_mean']:.1f} | {metrics['output_mean']:.1f} | "
            f"{metrics['total_tokens_mean']:.1f} | {metrics['cost_mean']:.2e} | "
            f"{metrics['em_mean']:.3f} | {metrics['f1_mean']:.3f} | "
            f"{metrics['precision_mean']:.3f} | {metrics['recall_mean']:.3f} | "
            f"{metrics['token_eff_mean']:.4f} | {metrics['cost_eff_mean']:.2f} |"
        )
        lines.append(line)

    return "\n".join(lines) + "\n"


def save_summary(model: str, summary: str) -> None:
    """Save summary to markdown file.

    Args:
        model: Model name
        summary: Markdown summary text
    """
    results_dir = Path(__file__).parent / "results" / "baseline"
    md_path = results_dir / f"analysis_{model}.md"

    md_path.write_text(summary)
    print(f"Summary saved: {md_path}")


def analyse_baseline(model: str) -> None:
    """Main analysis function.

    Args:
        model: Model name

    Raises:
        FileNotFoundError: If CSV doesn't exist
    """
    print(f"Analyzing baseline results for: {model}")

    # Load CSV
    results, csv_path = load_csv(model)
    print(f"Loaded {len(results)} results from {csv_path}")

    # Aggregate by method
    aggregated = aggregate_by_method(results)
    print(f"Found {len(aggregated)} methods")

    # Generate summary
    summary = generate_summary(model, aggregated)

    # Print summary
    print("\n" + summary)

    # Save summary
    save_summary(model, summary)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m experiments.analyse_baseline <model>")
        print("\nExample:")
        print("  uv run python -m experiments.analyse_baseline gpt-4o-mini")
        print("  uv run python -m experiments.analyse_baseline claude-sonnet-4-6")
        sys.exit(1)

    model = sys.argv[1]

    try:
        analyse_baseline(model)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
