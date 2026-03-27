"""
Day 2: Run baseline analysis and generate results.

This script:
1. Loads 50 random HotpotQA questions
2. Runs 3 methods: Always-Think, Always-Retrieve k=3, Always-Retrieve k=5
3. Measures F1, exact match, tokens, and cost
4. Saves results to CSV and summary markdown
"""

import csv
import time
import os
import sys
from collections import defaultdict
from pathlib import Path

from eval_utils import f1_score, exact_match, cost_usd, efficiency
from dataset_prep import get_sample, extract_paragraphs
from baseline_always_think import always_think
from baseline_always_retrieve import always_retrieve


def check_api_key():
    """Check that OPENAI_API_KEY is set before running."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ ERROR: OPENAI_API_KEY not found in environment variables")
        print("\nSet it with (Windows PowerShell):")
        print("  $env:OPENAI_API_KEY = 'sk-your-key-here'")
        print("\nOr (Windows CMD):")
        print("  set OPENAI_API_KEY=sk-your-key-here")
        print("\nGet your key from: https://platform.openai.com/account/api-keys\n")
        sys.exit(1)
    return api_key


def run_all_baselines(n_samples: int = 50, output_dir: str = "results"):
    """
    Run all baselines on HotpotQA samples.

    Args:
        n_samples: Number of questions to evaluate
        output_dir: Directory to save results
    """
    # Verify API key before starting
    api_key = check_api_key()
    print(f"✓ API key found ({api_key[:15]}...)\n")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Load dataset
    print(f"📥 Loading {n_samples} HotpotQA samples...")
    samples = get_sample(n=n_samples, seed=42)
    print(f"✓ Loaded {len(samples)} samples\n")

    results = []

    # Define methods to test
    methods = [
        ("Always-Think (CoT only)", always_think, {"question": None}),
        ("Always-Retrieve k=3", always_retrieve, {"question": None, "paragraphs": None, "k": 3}),
        ("Always-Retrieve k=5", always_retrieve, {"question": None, "paragraphs": None, "k": 5}),
    ]

    # Run evaluation
    print(f"{'#':<4} {'Question':<50} {'Method':<25} {'F1':<6} {'Tokens':<7}")
    print("=" * 98)

    for i, sample in enumerate(samples):
        question = sample['question']
        ground_truth = sample['answer']
        paragraphs = extract_paragraphs(sample)

        # Print progress
        q_short = question[:47] + "..." if len(question) > 50 else question
        print(f"{i+1:<4} {q_short:<50}", end="")

        for method_name, method_fn, kwargs in methods:
            try:
                # Fill in question/paragraphs
                filled_kwargs = kwargs.copy()
                filled_kwargs['question'] = question
                if 'paragraphs' in filled_kwargs:
                    filled_kwargs['paragraphs'] = paragraphs

                # Call method
                result = method_fn(**filled_kwargs)

                # Calculate metrics
                f1 = f1_score(result['answer'], ground_truth)
                em = exact_match(result['answer'], ground_truth)
                total_cost = cost_usd(result['input_tokens'], result['output_tokens'])
                effic = efficiency(f1, result['total_tokens'])

                # Store result
                results.append({
                    "question_id": i,
                    "question": question,
                    "ground_truth": ground_truth,
                    "method": result['method'],
                    "answer": result['answer'],
                    "f1_score": round(f1, 4),
                    "exact_match": em,
                    "input_tokens": result['input_tokens'],
                    "output_tokens": result['output_tokens'],
                    "total_tokens": result['total_tokens'],
                    "cost_usd": round(total_cost, 6),
                    "efficiency": round(effic, 5),
                    "llm_calls": result['llm_calls'],
                    **({
                        "docs_retrieved": result.get('docs_retrieved'),
                        "avg_similarity": result.get('avg_similarity')
                    } if 'docs_retrieved' in result else {})
                })

                print(f"\n{'':<54} {result['method']:<25} {f1:<6.3f} {result['total_tokens']:<7}",
                      end="")

            except Exception as e:
                print(f"\n{'':<54} ❌ ERROR: {str(e)[:40]}", end="")

        print()  # Newline after all methods for this question
        time.sleep(1)  # Rate limit

    # Save results
    print("\n" + "=" * 98)
    if results:
        csv_path = output_path / "results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"✓ Saved {len(results)} rows to {csv_path}\n")

        # Generate summary
        print_summary(results, output_path)


def print_summary(results: list[dict], output_dir: Path):
    """Print and save summary statistics."""
    # Group by method
    grouped = defaultdict(list)
    for r in results:
        grouped[r['method']].append(r)

    # Print table
    print("📊 SUMMARY BY METHOD")
    print("=" * 95)
    print(f"{'Method':<25} {'Count':<7} {'F1':<8} {'EM':<8} {'Tokens':<10} {'Cost':<10} {'Eff':<8}")
    print("-" * 95)

    summary_data = []
    for method in sorted(grouped.keys()):
        rows = grouped[method]
        count = len(rows)
        avg_f1 = sum(r['f1_score'] for r in rows) / count
        avg_em = sum(r['exact_match'] for r in rows) / count
        avg_tokens = sum(r['total_tokens'] for r in rows) / count
        avg_cost = sum(r['cost_usd'] for r in rows) / count
        avg_eff = sum(r['efficiency'] for r in rows) / count

        print(
            f"{method:<25} {count:<7} "
            f"{avg_f1:<8.4f} {avg_em:<8.4f} {avg_tokens:<10.0f} "
            f"${avg_cost:<9.5f} {avg_eff:<8.5f}"
        )

        summary_data.append({
            "method": method,
            "count": count,
            "avg_f1": avg_f1,
            "avg_exact_match": avg_em,
            "avg_tokens": avg_tokens,
            "avg_cost_usd": avg_cost,
            "avg_efficiency": avg_eff
        })

    # Save summary as markdown
    md_path = output_dir / "summary.md"
    with open(md_path, "w") as f:
        f.write("# Day 2 Baseline Analysis Summary\n\n")
        f.write(f"**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Samples:** {len(results)} questions\n\n")

        f.write("## Results by Method\n\n")
        f.write("| Method | Count | F1 Score | Exact Match | Avg Tokens | Avg Cost | Efficiency |\n")
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
        best_f1 = max(summary_data, key=lambda x: x['avg_f1'])
        cheapest = min(summary_data, key=lambda x: x['avg_tokens'])
        best_eff = max(summary_data, key=lambda x: x['avg_efficiency'])

        f.write(f"- **Best F1:** {best_f1['method']} ({best_f1['avg_f1']:.4f})\n")
        f.write(f"- **Cheapest (tokens):** {cheapest['method']} ({cheapest['avg_tokens']:.0f} tokens)\n")
        f.write(f"- **Best Efficiency:** {best_eff['method']} ({best_eff['avg_efficiency']:.5f})\n")

    print(f"\n✓ Saved summary to {md_path}")


if __name__ == "__main__":
    run_all_baselines(n_samples=50, output_dir="results")
