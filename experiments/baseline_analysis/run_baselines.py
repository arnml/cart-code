"""
Day 2: Run baseline analysis and generate results.

1. Loads 50 random HotpotQA questions
2. Runs 3 methods: Always-Think, Always-Retrieve k=3, Always-Retrieve k=5
3. Measures F1, exact match, tokens, and cost
4. Saves results to CSV and summary markdown (model name in directory)
"""

import csv
import os
import sys
import time
from pathlib import Path
from typing import Any

from baseline_always_retrieve import always_retrieve
from baseline_always_think import always_think
from dataset_prep import extract_paragraphs, get_sample
from eval_utils import cost_usd, efficiency, exact_match, f1_score
from run_utils import print_summary


def check_api_key() -> str:
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


def run_all_baselines(
    n_samples: int = 50, output_dir: str = "results", model: str = "gpt-4o-mini"
) -> None:
    """Run all baselines on HotpotQA samples."""
    api_key = check_api_key()
    print(f"✓ API key found ({api_key[:15]}...)\n")

    output_path = Path(f"{output_dir}_{model.replace('-', '_')}")

    print(f"📥 Loading {n_samples} HotpotQA samples...")
    samples = get_sample(n=n_samples, seed=42)
    print(f"✓ Loaded {len(samples)} samples")
    print(f"🤖 Model: {model}\n")

    results: list[dict] = []

    methods: list[tuple[str, Any, dict[str, Any]]] = [
        ("Always-Think (CoT only)", always_think, {"question": None, "model": model}),
        (
            "Always-Retrieve k=3",
            always_retrieve,
            {"question": None, "paragraphs": None, "k": 3, "model": model},
        ),
        (
            "Always-Retrieve k=5",
            always_retrieve,
            {"question": None, "paragraphs": None, "k": 5, "model": model},
        ),
    ]

    print(f"{'#':<4} {'Question':<50} {'Method':<25} {'F1':<6} {'Tokens':<7}")
    print("=" * 98)

    for i, sample in enumerate(samples):
        question = sample["question"]
        ground_truth = sample["answer"]
        paragraphs = extract_paragraphs(sample)

        q_short = question[:47] + "..." if len(question) > 50 else question
        print(f"{i + 1:<4} {q_short:<50}", end="")

        for _, method_fn, kwargs in methods:
            try:
                filled_kwargs = {**kwargs, "question": question}
                if "paragraphs" in filled_kwargs:
                    filled_kwargs["paragraphs"] = paragraphs

                result = method_fn(**filled_kwargs)

                f1 = f1_score(result["answer"], ground_truth)
                em = exact_match(result["answer"], ground_truth)
                total_cost = cost_usd(result["input_tokens"], result["output_tokens"], model=model)
                effic = efficiency(f1, result["total_tokens"])

                results.append(
                    {
                        "question_id": i,
                        "question": question,
                        "ground_truth": ground_truth,
                        "method": result["method"],
                        "answer": result["answer"],
                        "f1_score": round(f1, 4),
                        "exact_match": em,
                        "input_tokens": result["input_tokens"],
                        "output_tokens": result["output_tokens"],
                        "total_tokens": result["total_tokens"],
                        "cost_usd": round(total_cost, 6),
                        "efficiency": round(effic, 5),
                        "llm_calls": result["llm_calls"],
                        "docs_retrieved": result.get("docs_retrieved"),
                        "avg_similarity": result.get("avg_similarity"),
                    }
                )

                print(f"{result['method']:<25} {f1:<6.3f} {result['total_tokens']:<7}", end="")

            except Exception as e:
                print(f"\n{'':<54} ❌ ERROR: {str(e)[:40]}", end="")

        print()
        time.sleep(1)  # rate limit

    print("\n" + "=" * 98)
    if results:
        output_path.mkdir(parents=True, exist_ok=True)
        csv_path = output_path / "results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"✓ Saved {len(results)} rows to {csv_path}\n")
        print_summary(results, output_path, model)


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o-mini"
    run_all_baselines(n_samples=50, output_dir="results", model=model)
