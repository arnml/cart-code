"""
Diagnostic analysis for baseline results across models.

Identifies:
1. Questions where think fails but k5 succeeds (where retrieval genuinely helps)
2. Sample think-only outputs to understand token usage differences

Saves results to: analysis/cart_targets.md
"""

import csv
from pathlib import Path
from datetime import datetime


def analyze_model(model_dir: str, model_name: str) -> str:
    """Analyze results for a specific model. Returns markdown text."""
    csv_path = Path(model_dir) / "results.csv"

    if not csv_path.exists():
        return f"\n### {model_name}\n\n⚠️ No results found at `{csv_path}`\n\n"

    output = f"\n## {model_name}\n\n"

    # Try different encodings
    try:
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except UnicodeDecodeError:
        with open(csv_path, encoding="latin-1") as f:
            rows = list(csv.DictReader(f))

    # Group by question
    by_q = {}
    for r in rows:
        qid = r['question_id']
        if qid not in by_q:
            by_q[qid] = {}
        by_q[qid][r['method']] = r

    # Find questions where think fails but k5 succeeds
    think_fails_k5_wins = []
    for qid, methods in by_q.items():
        t = methods.get('always_think', {})
        k5 = methods.get('always_retrieve_k5', {})

        try:
            t_f1 = float(t.get('f1_score', 0))
            k5_f1 = float(k5.get('f1_score', 0))

            if t_f1 < 0.3 and k5_f1 > 0.6:
                think_fails_k5_wins.append({
                    'qid': qid,
                    'question': t.get('question', '')[:70],
                    'ground_truth': t.get('ground_truth'),
                    'think_f1': t_f1,
                    'think_tokens': int(t.get('total_tokens', 0)),
                    'k5_f1': k5_f1,
                    'k5_tokens': int(k5.get('total_tokens', 0)),
                })
        except (ValueError, TypeError):
            continue

    # Section 1: Questions where retrieval genuinely helps
    output += f"### CART Targets: Questions Retrieval Solves ({len(think_fails_k5_wins)})\n\n"
    output += "Questions where think F1 < 0.3 but k5 F1 > 0.6 (retrieval is crucial)\n\n"

    if think_fails_k5_wins:
        for r in sorted(think_fails_k5_wins, key=lambda x: x['k5_f1'] - x['think_f1'], reverse=True)[:10]:
            gap = (r['k5_f1'] - r['think_f1']) * 100
            output += f"**Q{r['qid']}:** {r['question']}\n\n"
            output += f"- Ground truth: `{r['ground_truth']}`\n"
            output += f"- Think F1: **{r['think_f1']:.3f}** ({r['think_tokens']} tokens)\n"
            output += f"- K5 F1: **{r['k5_f1']:.3f}** ({r['k5_tokens']} tokens)\n"
            output += f"- Retrieval gap: **+{gap:.1f}%** (CART should close this)\n\n"
    else:
        output += "> No such questions found - think is competitive!\n\n"

    # Section 2: Sample think-only outputs
    output += f"### Sample Think-Only Outputs\n\n"

    think_rows = [r for r in rows if r['method'] == 'always_think']
    for i, r in enumerate(think_rows[:5], 1):
        answer_preview = r['answer'][:100] if r['answer'] else "(empty)"
        output += f"**Sample {i}:**\n\n"
        output += f"- **Q:** {r['question'][:60]}\n"
        output += f"- **Ground truth:** `{r['ground_truth']}`\n"
        output += f"- **Answer:** `{answer_preview}...`\n"
        output += f"- **F1:** {r['f1_score']} | **Tokens:** {r['total_tokens']}\n\n"

    # Section 3: Summary statistics table
    output += f"### Summary Statistics\n\n"
    output += f"| Method | Avg F1 | Avg Tokens | Avg Cost |\n"
    output += f"|--------|--------|-----------|----------|\n"

    for method in sorted(set(r['method'] for r in rows)):
        method_rows = [r for r in rows if r['method'] == method]
        avg_f1 = sum(float(r['f1_score']) for r in method_rows) / len(method_rows)
        avg_tokens = sum(int(r['total_tokens']) for r in method_rows) / len(method_rows)
        avg_cost = sum(float(r['cost_usd']) for r in method_rows) / len(method_rows)

        output += f"| {method} | {avg_f1:.4f} | {avg_tokens:.0f} | ${avg_cost:.5f} |\n"

    output += "\n"
    return output


def main():
    """Run diagnostics for all available models and save to markdown."""

    # Create analysis directory
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)

    # Build markdown output
    md = f"""# CART Targets: Baseline Diagnostics

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This analysis identifies questions where CART needs to excel:
- **CART Targets**: Questions where think fails but retrieval succeeds
- **Token Usage**: How different models use tokens on think-only tasks
- **Summary Stats**: Baseline performance across methods

---

"""

    print("\n" + "="*80)
    print("BASELINE DIAGNOSTICS: Which questions need CART to solve?")
    print("="*80)

    # Use correct directory names
    models = [
        ("results_gpt_4o_mini", "GPT-4o-mini"),
        ("results_gpt_5.4_mini_2026_03_17", "GPT-5.4-mini"),
        ("results_claude_haiku_4_5", "Claude Haiku 4.5"),
    ]

    for model_dir, model_name in models:
        result = analyze_model(model_dir, model_name)
        md += result
        print(result)

    # Interpretation section
    interpretation = """---

## CART Design Implications

### Questions Where Retrieval Genuinely Helps
These are your CART targets. CART should:
- Retrieve documents for these questions (like k5 does)
- Close the F1 gap between think and k5
- Use fewer tokens than k5 does it

### Token Usage Analysis
Compare think tokens across models:
- **GPT-4o-mini:** Baseline for think-only reasoning
- **GPT-5.4-mini:** Higher tokens might indicate longer reasoning
- **Haiku:** If tokens are higher than GPT-4o-mini, it may be:
  - More verbose reasoning
  - Different prompt interpretation
  - Padding or system message differences

### Performance Patterns
- If k5 doesn't help much: embedding retrieval is noisy
  - CART might need better re-ranking or filtering
- If k5 helps significantly: retrieval is valuable
  - CART should adaptively retrieve only when needed

### CART Success Criteria

For questions in CART Targets:
```
CART F1 >= always_retrieve_k5 F1
CART tokens < always_retrieve_k5 tokens (50%+ improvement?)
CART efficiency > always_retrieve_k5 efficiency
```

For other questions:
```
CART should be competitive with always_think
(low-cost fallback when retrieval not needed)
```
"""

    md += interpretation
    print(interpretation)

    # Save to file with UTF-8 encoding (overwrite if exists)
    output_path = analysis_dir / "cart_targets.md"
    print(f"\nWriting analysis to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)

    print("="*80)
    print(f"✓ Overwrote: {output_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
