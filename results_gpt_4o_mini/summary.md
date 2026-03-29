# Day 2 Baseline Analysis Summary — gpt-4o-mini

**Evaluation Date:** 2026-03-29 08:37:20
**Total Samples:** 150 questions
**Model:** gpt-4o-mini

## Results by Method (Global Statistics)

| Method | Count | F1 Score | EM | Avg Tokens | Reduction % | Global Efficiency |
|---|---|---|---|---|---|---|
| always_retrieve_k3 | 50 | 0.6294 | 0.4600 | 492 | 34.6% | 0.10151 |
| always_retrieve_k5 | 50 | 0.7065 | 0.5200 | 752 | 0.0% | 0.10665 |
| always_think | 50 | 0.4383 | 0.3400 | 176 | 76.6% | 0.08468 |

## Metric Definitions

- **F1 Score**: Mean token-level accuracy (0-1).
- **Reduction %**: Token savings compared to `always_retrieve_k5` baseline.
- **Global Efficiency**: $\eta = \overline{F1} / \ln(1 + \overline{T})$. Measures quality per unit of 'log-effort'.

## Interpretation

- **Best F1:** always_retrieve_k5 (0.7065)
- **Most Efficient (Tokens):** always_think (176 tokens)
- **Best Global Efficiency:** always_retrieve_k5 (0.10665)
