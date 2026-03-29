"""λ cost penalty ablation — Figure 3 in paper (20 samples, gpt-4o-mini only).

Goal: show that the -λ·c(a) term in UCB-Cost is the operative cost signal.
Expected pattern:
  λ=0.0 → no penalty → aggressive retrieval → high F1, high tokens
  λ=0.5 → mild penalty → slight shift toward think
  λ=1.0 → default (Table 5 result)
  λ=2.0 → heavy penalty → routes mostly to think → lower F1, low tokens
"""

import csv
import time
from collections import defaultdict
from pathlib import Path

from baseline_analysis.dataset_prep import extract_paragraphs, get_sample
from baseline_analysis.eval_utils import cost_usd, efficiency, exact_match, f1_score

from .cart import cart_full
from .policy import UCBCostPolicy

RESULTS_DIR = Path(__file__).parent / "results"

MODEL_KEY = "gpt4o_mini"
MODEL_STR = "gpt-4o-mini"
LAMBDAS = [0.0, 0.5, 1.0, 2.0]
N = 20
SEED = 42


def _run_lambda(lam: float, samples: list[dict]) -> list[dict]:
    rows = []
    policy = UCBCostPolicy(lambda_cost=lam)
    for i, s in enumerate(samples):
        q, gt = s["question"], s["answer"]
        paras = extract_paragraphs(s)
        print(f"  [λ={lam}] [{i + 1}/{len(samples)}] {q[:55]}...")
        routed_to = None
        try:
            r = cart_full(
                question=q,
                paragraphs=paras,
                model=MODEL_STR,
                lambda_cost=lam,
                policy=policy,
            )
            routed_to = r.get("routed_to")
            f1 = f1_score(r["answer"], gt)
            rows.append(
                {
                    "lambda_cost": lam,
                    "model": MODEL_KEY,
                    "qid": s.get("id", i),
                    "question": q,
                    "ground_truth": gt,
                    **r,
                    "f1": round(f1, 4),
                    "exact_match": exact_match(r["answer"], gt),
                    "cost_usd": round(cost_usd(r["input_tokens"], r["output_tokens"]), 6),
                    "efficiency": round(efficiency(f1, r["total_tokens"]), 5),
                }
            )
            print(f"    F1={f1:.3f} tok={r['total_tokens']} route={routed_to or 'n/a'}")
        except Exception as e:
            print(f"    ERROR: {e}")
        if routed_to != "think":
            time.sleep(0.4)
    return rows


def _build_summary(results: list[dict]) -> str:
    grouped: dict[float, list] = defaultdict(list)
    for r in results:
        grouped[r["lambda_cost"]].append(r)

    lines = ["# CART λ Ablation — Results (Figure 3)\n"]
    lines.append(f"{'λ':>5}{'F1':>7}{'Tokens':>8}{'Eff':>8}{'Think%':>8}{'Retrieve%':>10}")
    lines.append("=" * 50)
    for lam, rows in sorted(grouped.items()):
        f1_avg = sum(r["f1"] for r in rows) / len(rows)
        tok_avg = sum(r["total_tokens"] for r in rows) / len(rows)
        eff_avg = sum(r["efficiency"] for r in rows) / len(rows)
        think_pct = 100 * sum(
            1 for r in rows if r.get("routed_to", "") in ("think", "think_fallback")
        ) / len(rows)
        lines.append(
            f"{lam:>5.1f}{f1_avg:>7.3f}{tok_avg:>8.0f}{eff_avg:>8.4f}"
            f"{think_pct:>7.0f}%{100 - think_pct:>8.0f}%"
        )

    return "\n".join(lines)


def run() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    samples = get_sample(n=N, seed=SEED)
    results = []
    for lam in LAMBDAS:
        print(f"\n=== λ = {lam} ===")
        results.extend(_run_lambda(lam, samples))

    csv_path = RESULTS_DIR / "results_lambda_ablation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    summary = _build_summary(results)
    print("\n" + summary)

    md_path = RESULTS_DIR / "results_lambda_ablation.md"
    md_path.write_text(summary)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    run()
