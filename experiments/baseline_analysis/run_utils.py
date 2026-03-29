"""Shared utilities for running and summarizing baseline evaluations."""

import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .eval_utils import efficiency


def print_summary(results: list[dict[str, Any]], output_dir: Path, model: str) -> None:
    """Print and save summary statistics to a markdown file."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        grouped[r["method"]].append(r)

    # Determine baseline for token reduction (k5 is the standard)
    baseline_tokens = 0.0
    for method, rows in grouped.items():
        if "k5" in method.lower() or "always_retrieve_k5" in method:
            baseline_tokens = sum(r["total_tokens"] for r in rows) / len(rows)
            break

    print("📊 SUMMARY BY METHOD (Global Metrics)")
    print("=" * 115)
    print(f"{'Method':<25} {'Count':<7} {'F1':<8} {'EM':<8} {'Tokens':<10} {'Reduct%':<10} {'Eff_Global':<8}")
    print("-" * 115)

    summary_data: list[dict[str, Any]] = []
    for method in sorted(grouped):
        rows = grouped[method]
        count = len(rows)

        avg_f1 = sum(r["f1_score"] for r in rows) / count
        avg_em = sum(r["exact_match"] for r in rows) / count
        avg_tokens = sum(r["total_tokens"] for r in rows) / count
        avg_cost = sum(r["cost_usd"] for r in rows) / count
        
        # New Global Metrics
        reduction_pct = (1 - (avg_tokens / baseline_tokens)) * 100 if baseline_tokens > 0 else 0.0
        eff_global = efficiency(avg_f1, int(avg_tokens))

        print(
            f"{method:<25} {count:<7} "
            f"{avg_f1:<8.4f} {avg_em:<8.4f} {avg_tokens:<10.0f} "
            f"{reduction_pct:>8.1f}%   {eff_global:<8.5f}"
        )

        summary_data.append(
            {
                "method": method,
                "count": count,
                "avg_f1": avg_f1,
                "avg_exact_match": avg_em,
                "avg_tokens": avg_tokens,
                "reduction_pct": reduction_pct,
                "avg_cost_usd": avg_cost,
                "eff_global": eff_global,
            }
        )

    md_path = output_dir / "summary.md"
    with open(md_path, "w") as f:
        f.write(f"# Day 2 Baseline Analysis Summary — {model}\n\n")
        f.write(f"**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Samples:** {len(results)} questions\n")
        f.write(f"**Model:** {model}\n\n")

        f.write("## Results by Method (Global Statistics)\n\n")
        f.write(
            "| Method | Count | F1 Score | EM | Avg Tokens | Reduction % | Global Efficiency |\n"
        )
        f.write("|---|---|---|---|---|---|---|\n")
        for row in summary_data:
            f.write(
                f"| {row['method']} | {row['count']} | {row['avg_f1']:.4f} | "
                f"{row['avg_exact_match']:.4f} | {row['avg_tokens']:.0f} | "
                f"{row['reduction_pct']:.1f}% | {row['eff_global']:.5f} |\n"
            )

        f.write("\n## Metric Definitions\n\n")
        f.write("- **F1 Score**: Mean token-level accuracy (0-1).\n")
        f.write("- **Reduction %**: Token savings compared to `always_retrieve_k5` baseline.\n")
        f.write("- **Global Efficiency**: $\\eta = \\overline{F1} / \\ln(1 + \\overline{T})$. Measures quality per unit of 'log-effort'.\n\n")

        f.write("## Interpretation\n\n")
        best_f1 = max(summary_data, key=lambda x: x["avg_f1"])
        cheapest = min(summary_data, key=lambda x: x["avg_tokens"])
        best_eff = max(summary_data, key=lambda x: x["eff_global"])
        f.write(f"- **Best F1:** {best_f1['method']} ({best_f1['avg_f1']:.4f})\n")
        f.write(
            f"- **Most Efficient (Tokens):** {cheapest['method']} ({cheapest['avg_tokens']:.0f} tokens)\n"
        )
        f.write(f"- **Best Global Efficiency:** {best_eff['method']} ({best_eff['eff_global']:.5f})\n")

    print(f"✓ Saved global summary to {md_path}")
