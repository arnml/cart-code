"""Component ablation experiment for Anthropic Claude models: cart_base / cart_noise / cart_full, 20 samples."""

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

from .cart_anthropic import cart_base, cart_full, cart_noise
from .policy import UCBCostPolicy

RESULTS_DIR = Path(__file__).parent / "results"

def _run_sample(
    i: int,
    n: int,
    sample: dict,
    model_key: str,
    model_str: str,
    model_policies: dict[str, UCBCostPolicy],
) -> list[dict]:
    q, gt = sample["question"], sample["answer"]
    paras = extract_paragraphs(sample)
    print(f"\n[{i + 1}/{n}] {q[:60]}...")

    rows = []
    for fn, extra_kw in [
        (cart_base, {}),
        (cart_noise, {}),
        (cart_full, {"lambda_cost": 1.0}),
    ]:
        routed_to = None
        try:
            call_kw = dict(extra_kw)
            if fn is cart_full:
                call_kw["policy"] = model_policies[model_key]
            r = fn(question=q, paragraphs=paras, model=model_str, **call_kw)
            routed_to = r.get("routed_to")
            f1 = f1_score(r["answer"], gt)
            
            # Update policy with actual performance reward if it was a think action
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
                    "cost_usd": round(cost_usd(r["input_tokens"], r["output_tokens"], model=model_str), 6),
                    "efficiency": round(efficiency(f1, r["total_tokens"]), 5),
                }
            )
            print(
                f"  [{model_key}] {r['method']:<12} F1={f1:.3f} "
                f"tok={r['total_tokens']} route={routed_to or 'n/a'}"
            )
        except Exception as e:
            print(f"  ERROR [{model_key}] {fn.__name__}: {e}")
        time.sleep(0.5)
    return rows


def _build_summary(results: list[dict], model_key: str) -> str:
    grouped: dict[tuple, list] = defaultdict(list)
    for r in results:
        grouped[(r["model"], r["method"])].append(r)

    lines = [f"# CART Component Ablation for {model_key} — Results (Table 5)\n"]
    lines.append(f"{'Model':<14}{'Method':<14}{'F1':>6}{'Tokens':>8}{'Eff':>8}")
    lines.append("=" * 52)
    for (m, mth), rows in sorted(grouped.items()):
        f1_avg = sum(r["f1"] for r in rows) / len(rows)
        tok_avg = sum(r["total_tokens"] for r in rows) / len(rows)
        eff_avg = sum(r["efficiency"] for r in rows) / len(rows)
        lines.append(f"{m:<14}{mth:<14}{f1_avg:>6.3f}{tok_avg:>8.0f}{eff_avg:>8.4f}")

    lines.append("\n## CART-Full Routing (key proof of concept)\n")
    rows = [r for r in results if r["method"] == "cart_full" and r["model"] == model_key]
    if rows:
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


def run(n: int = 20, model_str: str = "claude-haiku-4-5") -> None:
    model_key = model_str.replace("-", "_").replace(".", "_")
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"🤖 Running experiment for model: {model_str}")
    samples = get_sample(n=n, seed=42)
    model_policies = {model_key: UCBCostPolicy(lambda_cost=1.0)}
    results = []
    for i, s in enumerate(samples):
        results.extend(_run_sample(i, n, s, model_key, model_str, model_policies))

    if not results:
        raise RuntimeError("No CART results were produced; CSV export aborted.")

    csv_path = RESULTS_DIR / f"results_cart_ablation_anthropic_{model_key}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_fieldnames(results))
        writer.writeheader()
        writer.writerows(results)

    summary = _build_summary(results, model_key)
    print("\n" + summary)

    md_path = RESULTS_DIR / f"results_cart_ablation_anthropic_{model_key}.md"
    md_path.write_text(summary)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CART ablation for Anthropic models.")
    parser.add_argument("--n", type=int, default=20, help="Number of samples.")
    parser.add_argument("--model", type=str, default="claude-haiku-4-5", help="Anthropic model ID.")
    args = parser.parse_args()
    run(n=args.n, model_str=args.model)
