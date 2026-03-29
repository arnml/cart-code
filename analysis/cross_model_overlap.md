# Cross-Model Analysis: CART Targets Overlap

**Generated:** 2026-03-29 09:22:37

This analysis identifies which CART targets are consistent across models,
helping prioritize the most important questions for CART to solve.

---

## Summary

| Category | Count |
|----------|-------|
| Universally hard (all 3 models) | 0 |
| Hard in 2+ models | 0 |
| Model-specific targets | 0 |

---

## Universally Hard Questions (All 3 Models)

These questions should be CART's top priority — all models struggle with them.

> No questions are universally hard across all models.

---

## Hard in 2+ Models (Consistent Difficulty)

Questions that multiple models struggle with but not all.

> No questions in this category.

---

## Model-Specific CART Targets

These are questions where only one model struggles.
They may represent model quirks rather than fundamental difficulty.

### GPT-4o-mini Only (0)

> (None — all targets are shared with other models)

### GPT-5.4-mini Only (0)

> (None — all targets are shared with other models)

### Claude Haiku 4.5 Only (0)

> (None — all targets are shared with other models)

---

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
