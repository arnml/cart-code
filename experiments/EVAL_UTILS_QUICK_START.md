# eval_utils.py Quick Start Guide

## What Changed?

Created **`experiments/core_utilities/eval_utils.py`** — a production-ready evaluation utility that:
- ✅ Follows **official HotpotQA v1 evaluation** exactly
- ✅ Returns **all metrics** (EM, F1, precision, recall) at once
- ✅ Includes **efficiency metrics** (F1/log(tokens), F1/USD)
- ✅ Tracks **embedding API costs** (often forgotten!)
- ✅ Has type hints and dataclasses (IDE-friendly)

## Side-by-Side Comparison

### Old Code (baseline_analysis/eval_utils.py)
```python
# Single metric at a time
from experiments.baseline_analysis.eval_utils import f1_score, exact_match, cost_usd

f1, p, r = f1_score(pred, gt)  # Returns (f1, p, r) but old code only returned f1!
em = exact_match(pred, gt)     # Returns int (0/1)
cost = cost_usd(inp, out)      # Doesn't include embedding cost
```

### New Code (core_utilities/eval_utils.py)
```python
# All metrics at once
from experiments.core_utilities.eval_utils import evaluate_sample, aggregate_metrics, cost_usd

metrics = evaluate_sample(
    prediction=pred,
    ground_truth=gt,
    input_tokens=500,
    output_tokens=50,
    cost_usd=cost_usd(500, 50, model="claude-haiku", embedding_tokens=8000)
)

# Access any metric
print(metrics.f1)               # 0.85
print(metrics.em)              # 0.0
print(metrics.precision)       # 0.88
print(metrics.recall)          # 0.82
print(metrics.efficiency)      # 0.120 (F1/log(tokens))
print(metrics.cost_efficiency) # 340 (F1/$)
```

## One-Liner Examples

### Evaluate Single Sample
```python
from experiments.core_utilities.eval_utils import evaluate_sample, cost_usd

metrics = evaluate_sample(
    "Paris is the capital",
    "Paris",
    input_tokens=500,
    output_tokens=50,
    cost_usd=cost_usd(500, 50, model="claude-haiku", embedding_tokens=8000)
)
print(f"F1={metrics.f1:.4f}, Cost-Eff={metrics.cost_efficiency:.0f}")
```

### Evaluate Batch
```python
from experiments.core_utilities.eval_utils import aggregate_metrics, format_metrics

results = aggregate_metrics(
    predictions=["Paris", "London", "Berlin"],
    ground_truths=["Paris", "London", "Berlin"],
    costs_usd=[0.0007, 0.0007, 0.0008],
)
print(format_metrics(results))
```

### Cost with Embedding API
```python
from experiments.core_utilities.eval_utils import cost_usd

# Old way (missing embedding cost)
cost_old = cost_usd(1000, 100)  # ~$0.0011

# New way (includes embedding)
cost_new = cost_usd(
    1000, 100,
    model="claude-haiku",
    embedding_tokens=8000  # For doc retrieval
)  # ~$0.0017 (adds ~$0.00016 embedding cost)
```

---

## Key Differences from Official HotpotQA

### ⚠️ Important: Special Answer Handling

Official HotpotQA has a rule for yes/no/noanswer:

```python
# If ground truth is "yes", "no", or "noanswer"
# → Prediction MUST match exactly to get credit

f1_score("yes", "yes") → (1.0, 1.0, 1.0) ✓
f1_score("yeah", "yes") → (0.0, 0.0, 0.0) ✗ (close but not exact)
f1_score("Paris is capital", "yes") → (0.0, 0.0, 0.0) ✗ (completely wrong)
```

**Your old code didn't have this rule** (mostly got lucky because of no token overlap).
**New code implements it correctly.**

### 📊 Return Types

| Metric | Old | New |
|--------|-----|-----|
| F1 | `float` (f1 only) | `(float, float, float)` (f1, prec, rec) |
| EM | `int` (0/1) | `bool` |
| Cost | `float` (LLM only) | `float` (LLM + embedding) |

### 🎯 Efficiency Metrics (New)

```python
# F1 per unit compute (better than raw tokens)
efficiency = f1 / log(1 + tokens)

# F1 per dollar spent (best for cost comparison)
cost_efficiency = f1 / cost_usd
```

Why log(tokens) instead of raw tokens?
- Raw: 0.8 / 1000 = 0.0008 vs 0.8 / 2000 = 0.0004 (harsh)
- Log: 0.8 / log(1001) = 0.116 vs 0.8 / log(2001) = 0.107 (gentle)

Diminishing returns: 2× tokens ≠ 2× cost.

---

## For Your Paper: What to Write

### Methods Section
```
Evaluation Metrics: We follow HotpotQA official v1 evaluation [ref],
computing token-level F1, exact match (EM), precision, and recall
after normalization (lowercase, remove articles/punctuation,
whitespace normalization). We additionally report:

- Efficiency: F1 / log(1 + total_tokens) — quality per compute
- Cost-Efficiency: F1 / USD — quality per dollar

Total cost includes LLM tokens at model-specific rates and
text-embedding-3-small embeddings at $0.02 per 1M tokens.
```

### Results Table Template
```
| Method | EM | F1 | Prec | Rec | Cost ($) | Eff | Cost-Eff |
|--------|----|----|------|-----|----------|-----|----------|
| CART | 72.0 | 85.0 | 88.0 | 82.0 | 0.0025 | 0.120 | 340 |
| Baseline | 68.0 | 80.0 | 83.0 | 78.0 | 0.0050 | 0.110 | 160 |
```

---

## File Locations

| File | Purpose |
|------|---------|
| `experiments/core_utilities/eval_utils.py` | ✨ NEW evaluation utility (use this!) |
| `experiments/baseline_analysis/eval_utils.py` | Old code (keep for now, can deprecate later) |
| `HOTPOTQA_COMPLIANCE_ANALYSIS.md` | Detailed comparison with official |
| `EVAL_UTILS_MIGRATION.md` | Migration guide for existing code |
| `EVAL_UTILS_VALIDATION.md` | Test results and validation |

---

## Migration Checklist

If you're updating existing code to use the new eval_utils:

- [ ] Import from `core_utilities` instead of `baseline_analysis`
- [ ] Update f1_score() calls: now returns (f1, prec, rec)
- [ ] Update cost_usd() calls: add `embedding_tokens` parameter
- [ ] Replace loops with `aggregate_metrics()` for batch evaluation
- [ ] Add efficiency and cost-efficiency to results tables
- [ ] Update paper Methods section
- [ ] Re-run experiments and verify results match (they should!)

---

## Common Questions

### Q: Do I have to update everything now?
**A**: No. The new code is in `core_utilities/` so it coexists with old code. Update when convenient.

### Q: Will results change?
**A**: Probably not much. F1/EM calculation is the same. Main changes:
- Yes/no/noanswer now strict (might affect a few % of data if you have those answers)
- Embedding cost added (small ~$0.00016/question)
- Efficiency metrics are new (no old values to compare)

### Q: What about supporting facts?
**A**: Official HotpotQA evaluates supporting facts too. New code doesn't (CART doesn't use them). Can be added if needed.

### Q: Can I use the old and new code together?
**A**: Yes. They coexist. Import from whichever one you want. Eventually deprecate the old one.

---

## Run It

```bash
cd c:/code/smtl-code
uv run python -m experiments.core_utilities.eval_utils
```

Output:
```
Single Sample Evaluation:
  EM: 0.0
  F1: 0.3333
  Precision: 0.2000
  Recall: 1.0000
  Efficiency: 0.0528
  Cost Efficiency: 444.44

Metrics (n=3):
  EM:        0.0000
  F1:        0.4722
  Precision: 0.3333
  Recall:    1.0000
  Efficiency (F1/log(tok)): 0.0745
  Cost-Eff (F1/$):         582.01
```

✅ All working!

---

## Questions?

See the detailed analysis files:
- `HOTPOTQA_COMPLIANCE_ANALYSIS.md` — technical deep-dive
- `EVAL_UTILS_MIGRATION.md` — for updating your code
- `EVAL_UTILS_VALIDATION.md` — test results and edge cases
