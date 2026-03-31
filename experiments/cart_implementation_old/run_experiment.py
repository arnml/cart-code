"""Component ablation experiment (Table 5 in paper): cart_base / cart_noise / cart_full × 2 models, 50 samples."""

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add experiments directory to path to allow importing from baseline_analysis
sys.path.append(str(Path(__file__).parent.parent))

try:
    from baseline_analysis.dataset_prep import extract_paragraphs, get_sample
    from baseline_analysis.eval_utils import cost_usd, efficiency, exact_match, f1_score
except ImportError:
    # Fallback for different execution environments
    from experiments.baseline_analysis.dataset_prep import extract_paragraphs, get_sample
    from experiments.baseline_analysis.eval_utils import cost_usd, efficiency, exact_match, f1_score

from .cart import cart_base, cart_full, cart_noise
from .policy import UCBCostPolicy

RESULTS_DIR = Path(__file__).parent / "results"

MODELS: dict[str, str] = {
    "gpt4o_mini": "gpt-4o-mini",
    "gpt54_mini": "gpt-5.4-mini-2026-03-17",
}

METHODS = [
    (cart_base, {}),
    (cart_noise, {}),
    (cart_full, {"lambda_cost": 1.0}),
]


def _run_sample(
    i: int,
    n: int,
    sample: dict,
    model_policies: dict[str, UCBCostPolicy],
) -> list[dict]:
    q, gt = sample["question"], sample["answer"]
    paras = extract_paragraphs(sample)
    print(f"\n[{i + 1}/{n}] {q[:60]}...")

    rows = []
    for model_key, model_str in MODELS.items():
        for fn, extra_kw in METHODS:
            routed_to = None
            try:
                call_kw = dict(extra_kw)
                if fn is cart_full:
                    call_kw["policy"] = model_policies[model_key]
                r = fn(question=q, paragraphs=paras, model=model_str, **call_kw)
                routed_to = r.get("routed_to")
                f1 = f1_score(r["answer"], gt)
                
                # Update policy with actual performance reward if it was a think action
                # (which can't be assessed via similarity proxy at test-time)
                if fn is cart_full and routed_to in ("think", "think_fallback"):
                    model_policies[model_key].update("think", f1)
                
                rows.append(
                    {
                        "model": model_key,
                        "qid": sample.get("id", i),
                        "question": q,
                        "ground_truth": gt,
                        **r,
                        "f1": round(f1, 4),
                        "exact_match": exact_match(r["answer"], gt),
                        "cost_usd": round(cost_usd(r["input_tokens"], r["output_tokens"]), 6),
                        "efficiency": round(efficiency(f1, r["total_tokens"]), 5),
                    }
                )
                print(
                    f"  [{model_key}] {r['method']:<12} F1={f1:.3f} "
                    f"tok={r['total_tokens']} route={routed_to or 'n/a'}"
                )
            except Exception as e:
                print(f"  ERROR [{model_key}] {fn.__name__}: {e}")
            if routed_to != "think":
                time.sleep(0.4)
    return rows


def _build_summary(results: list[dict]) -> str:
    grouped: dict[tuple, list] = defaultdict(list)
    for r in results:
        grouped[(r["model"], r["method"])].append(r)

    # Determine baseline for token reduction per model (cart_base is our 'standard')
    baselines = {}
    for (model, method), rows in grouped.items():
        if method == "cart_base":
            baselines[model] = sum(r["total_tokens"] for r in rows) / len(rows)

    lines = ["# CART Component Ablation — Results (Global Metrics)\n"]
    lines.append(f"{'Model':<14}{'Method':<14}{'F1':>6}{'Tokens':>8}{'Reduct%':>10}{'Eff_Glb':>10}")
    lines.append("=" * 66)
    for (m, mth), rows in sorted(grouped.items()):
        f1_avg = sum(r["f1"] for r in rows) / len(rows)
        tok_avg = sum(r["total_tokens"] for r in rows) / len(rows)
        
        # New Global Metrics
        ref_tokens = baselines.get(m, tok_avg)
        reduction_pct = (1 - (tok_avg / ref_tokens)) * 100 if ref_tokens > 0 else 0.0
        eff_global = efficiency(f1_avg, int(tok_avg))
        
        lines.append(f"{m:<14}{mth:<14}{f1_avg:>6.3f}{tok_avg:>8.0f}{reduction_pct:>9.1f}%{eff_global:>10.4f}")

    lines.append("\n## CART-Full Routing (key proof of concept)\n")
    for model_key in MODELS:
        rows = [r for r in results if r["method"] == "cart_full" and r["model"] == model_key]
        if not rows:
            continue
        think_count = sum(1 for r in rows if r.get("routed_to", "") in ("think", "think_fallback"))
        retrieve_count = len(rows) - think_count
        lines.append(
            f"- {model_key}: think={think_count} ({100 * think_count / len(rows):.0f}%)  "
            f"retrieve={retrieve_count} ({100 * retrieve_count / len(rows):.0f}%)"
        )

    return "\n".join(lines)


def _csv_fieldnames(rows: list[dict]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def run(n: int = 50) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    samples = get_sample(n=n, seed=42)
    model_policies = {model_key: UCBCostPolicy(lambda_cost=1.0) for model_key in MODELS}
    results = [
        row
        for i, s in enumerate(samples)
        for row in _run_sample(i, n, s, model_policies)
    ]

    if not results:
        raise RuntimeError("No CART results were produced; CSV export aborted.")

    csv_path = RESULTS_DIR / f"results_cart_ablation_n{n}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_fieldnames(results))
        writer.writeheader()
        writer.writerows(results)

    summary = _build_summary(results)
    print("\n" + summary)

    md_path = RESULTS_DIR / f"results_cart_ablation_n{n}.md"
    md_path.write_text(summary)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CART component ablation experiment.")
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of HotpotQA samples to run (default: 50).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(n=args.n)
