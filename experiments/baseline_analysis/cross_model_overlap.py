"""
Cross-model analysis: Find CART targets that appear in multiple models.

Identifies:
1. "Universally hard" questions (CART targets in ALL models)
2. Model-specific gaps (targets in some but not all models)
3. Model strengths and weaknesses

Usage:
    cd experiments/baseline_analysis
    uv run python cross_model_overlap.py

Output: analysis/cross_model_overlap.md
"""

from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .eval_utils import load_csv_safe


def load_targets(csv_path, think_method="always_think", k5_method="always_retrieve_k5"):
    """Load CART targets from a results CSV file."""
    if not Path(csv_path).exists():
        return {}

    rows = load_csv_safe(Path(csv_path))

    by_q = defaultdict(dict)
    for r in rows:
        by_q[r["question_id"]][r["method"]] = r

    targets = {}
    for qid, methods in by_q.items():
        t_row = methods.get(think_method, {})
        k5_row = methods.get(k5_method, {})

        try:
            t_f1 = float(t_row.get("f1_score", 0))
            k5_f1 = float(k5_row.get("f1_score", 0))

            if t_f1 < 0.3 and k5_f1 > 0.6:
                q = t_row.get("question", "")
                gt = t_row.get("ground_truth", "")
                targets[qid] = {
                    "q": q,
                    "gt": gt,
                    "t_f1": t_f1,
                    "k5_f1": k5_f1,
                    "t_tokens": int(t_row.get("total_tokens", 0)),
                    "k5_tokens": int(k5_row.get("total_tokens", 0)),
                }
        except (ValueError, TypeError):
            continue

    return targets


def main():
    """Analyze cross-model overlap of CART targets."""

    # Create analysis directory
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)

    # Load targets from all models
    models = {
        "GPT-4o-mini": Path("results_gpt_4o_mini") / "results.csv",
        "GPT-5.4-mini": Path("results_gpt_5_4_mini_2026_03_17") / "results.csv",
        "Claude Haiku 4.5": Path("results_claude_haiku_4_5") / "results.csv",
    }

    targets = {}
    for model_name, csv_path in models.items():
        targets[model_name] = load_targets(str(csv_path))
        print(f"Loaded {len(targets[model_name])} CART targets from {model_name}")

    print()

    # Find overlaps
    all_qids = set()
    for model_targets in targets.values():
        all_qids.update(model_targets.keys())

    # Build markdown report
    md = f"""# Cross-Model Analysis: CART Targets Overlap

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This analysis identifies which CART targets are consistent across models,
helping prioritize the most important questions for CART to solve.

---

## Summary

"""

    # Count overlaps
    model_names = list(targets.keys())
    all_targets = set(targets[model_names[0]].keys())
    for model_name in model_names[1:]:
        all_targets = all_targets & set(targets[model_name].keys())

    overlap_2plus = set()
    for qid in all_qids:
        count = sum(1 for m in model_names if qid in targets[m])
        if count >= 2:
            overlap_2plus.add(qid)

    md += f"""| Category | Count |
|----------|-------|
| Universally hard (all 3 models) | {len(all_targets)} |
| Hard in 2+ models | {len(overlap_2plus)} |
| Model-specific targets | {sum(len(targets[m]) for m in model_names) - len(overlap_2plus) * 3} |

---

## Universally Hard Questions (All 3 Models)

These questions should be CART's top priority — all models struggle with them.

"""

    if all_targets:
        for qid in sorted(all_targets):
            t4o = targets[model_names[0]].get(qid, {})
            md += f"**Q{qid}:** {t4o.get('q', '')[:70]}\n\n"
            md += f"- Ground truth: `{t4o.get('gt', '')}`\n"

            for model_name in model_names:
                if qid in targets[model_name]:
                    t = targets[model_name][qid]
                    gap = (t["k5_f1"] - t["t_f1"]) * 100
                    md += (
                        f"- **{model_name}**: think F1={t['t_f1']:.3f},"
                        f" k5 F1={t['k5_f1']:.3f} (gap={gap:.0f}%)\n"
                    )

            md += "\n"
    else:
        md += "> No questions are universally hard across all models.\n\n"

    # Hard in 2+ models
    md += """---

## Hard in 2+ Models (Consistent Difficulty)

Questions that multiple models struggle with but not all.

"""

    overlap_2 = overlap_2plus - all_targets
    if overlap_2:
        for qid in sorted(overlap_2):
            # Get first available info
            for model_name in model_names:
                if qid in targets[model_name]:
                    t = targets[model_name][qid]
                    md += f"**Q{qid}:** {t.get('q', '')[:70]}\n\n"
                    md += f"- Ground truth: `{t.get('gt', '')}`\n"
                    break

            for model_name in model_names:
                if qid in targets[model_name]:
                    t = targets[model_name][qid]
                    gap = (t["k5_f1"] - t["t_f1"]) * 100
                    md += f"- **{model_name}**: gap={gap:.0f}%\n"
                else:
                    md += f"- **{model_name}**: (not a CART target)\n"

            md += "\n"
    else:
        md += "> No questions in this category.\n\n"

    # Model-specific targets
    md += """---

## Model-Specific CART Targets

These are questions where only one model struggles.
They may represent model quirks rather than fundamental difficulty.

"""

    for model_name in model_names:
        specific = set(targets[model_name].keys()) - overlap_2plus
        md += f"### {model_name} Only ({len(specific)})\n\n"

        if specific:
            for qid in sorted(specific)[:5]:
                t = targets[model_name][qid]
                gap = (t["k5_f1"] - t["t_f1"]) * 100
                md += f"- Q{qid}: `{t['gt']}` — {t['q'][:50]} (gap={gap:.0f}%)\n"

            if len(specific) > 5:
                md += f"- ... and {len(specific) - 5} more\n"
        else:
            md += "> (None — all targets are shared with other models)\n"

        md += "\n"

    # Recommendations
    md += """---

## Recommendations for CART Implementation

### Priority 1: Universally Hard Questions
- These are your paper's strongest motivation
- Solving even 50% of these shows CART's value
- Use for main paper figures and tables

### Priority 2: Consistent Multi-Model Targets
- Shows CART generalizes across different LLMs
- Good for robustness claims
- Demonstrates the approach isn't model-specific

### Priority 3: Model-Specific Targets (Optional)
- Lower priority for the paper
- Consider skipping if time is limited
- Useful for appendix or future work

### Implementation Strategy
1. **Start with Priority 1** — solve universally hard questions
2. **Validate on Priority 2** — ensure generalization to other models
3. **Extend to Priority 3** — if time permits
"""

    # Save to file
    output_path = analysis_dir / "cross_model_overlap.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)

    print("=" * 80)
    print(f"✓ Overwrote: {output_path}")
    print("=" * 80)
    print(md)


if __name__ == "__main__":
    main()
