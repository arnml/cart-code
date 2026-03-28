"""Shared utilities for running and summarizing baseline evaluations."""

import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def print_summary(results: list[dict[str, Any]], output_dir: Path, model: str) -> None:
    """Print and save summary statistics to a markdown file."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        grouped[r["method"]].append(r)

    print("📊 SUMMARY BY METHOD")
    print("=" * 95)
    print(f"{'Method':<25} {'Count':<7} {'F1':<8} {'EM':<8} {'Tokens':<10} {'Cost':<10} {'Eff':<8}")
    print("-" * 95)

    summary_data: list[dict[str, Any]] = []
    for method in sorted(grouped):
        rows = grouped[method]
        count = len(rows)

        # Single-pass aggregation over method rows
        total_f1 = total_em = total_tokens = total_cost = total_eff = 0.0
        for r in rows:
            total_f1 += r["f1_score"]
            total_em += r["exact_match"]
            total_tokens += r["total_tokens"]
            total_cost += r["cost_usd"]
            total_eff += r["efficiency"]

        avg_f1 = total_f1 / count
        avg_em = total_em / count
        avg_tokens = total_tokens / count
        avg_cost = total_cost / count
        avg_eff = total_eff / count

        print(
            f"{method:<25} {count:<7} "
            f"{avg_f1:<8.4f} {avg_em:<8.4f} {avg_tokens:<10.0f} "
            f"${avg_cost:<9.5f} {avg_eff:<8.5f}"
        )

        summary_data.append(
            {
                "method": method,
                "count": count,
                "avg_f1": avg_f1,
                "avg_exact_match": avg_em,
                "avg_tokens": avg_tokens,
                "avg_cost_usd": avg_cost,
                "avg_efficiency": avg_eff,
            }
        )

    md_path = output_dir / "summary.md"
    with open(md_path, "w") as f:
        f.write(f"# Day 2 Baseline Analysis Summary — {model}\n\n")
        f.write(f"**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Samples:** {len(results)} questions\n")
        f.write(f"**Model:** {model}\n\n")

        f.write("## Results by Method\n\n")
        f.write(
            "| Method | Count | F1 Score | Exact Match | Avg Tokens | Avg Cost | Efficiency |\n"
        )
        f.write("|---|---|---|---|---|---|---|\n")
        for row in summary_data:
            f.write(
                f"| {row['method']} | {row['count']} | {row['avg_f1']:.4f} | "
                f"{row['avg_exact_match']:.4f} | {row['avg_tokens']:.0f} | "
                f"${row['avg_cost_usd']:.5f} | {row['avg_efficiency']:.5f} |\n"
            )

        f.write("\n## Key Observations\n\n")
        f.write("- **F1 Score**: Measures answer correctness (0-1, higher is better)\n")
        f.write("- **Exact Match**: Binary perfect answer match (0 or 1)\n")
        f.write("- **Tokens**: Total input + output tokens (fewer = cheaper)\n")
        f.write("- **Cost**: Estimated USD cost for all queries\n")
        f.write("- **Efficiency**: F1 / log(1 + tokens) — quality per unit cost\n\n")

        f.write("## Interpretation\n\n")
        best_f1 = max(summary_data, key=lambda x: x["avg_f1"])
        cheapest = min(summary_data, key=lambda x: x["avg_tokens"])
        best_eff = max(summary_data, key=lambda x: x["avg_efficiency"])
        f.write(f"- **Best F1:** {best_f1['method']} ({best_f1['avg_f1']:.4f})\n")
        f.write(
            f"- **Cheapest (tokens):** {cheapest['method']} ({cheapest['avg_tokens']:.0f} tokens)\n"
        )
        f.write(f"- **Best Efficiency:** {best_eff['method']} ({best_eff['avg_efficiency']:.5f})\n")

    print(f"✓ Saved summary to {md_path}")
