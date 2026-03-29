# Day 2 Baseline Analysis Summary — claude-haiku-4-5

**Evaluation Date:** 2026-03-29 08:51:01
**Total Samples:** 150 questions
**Model:** claude-haiku-4-5

## Results by Method (Global Statistics)

| Method | Count | F1 Score | EM | Avg Tokens | Reduction % | Global Efficiency |
|---|---|---|---|---|---|---|
| always_retrieve_k3 | 50 | 0.3624 | 0.1800 | 648 | 30.5% | 0.05598 |
| always_retrieve_k5 | 50 | 0.4910 | 0.2800 | 932 | 0.0% | 0.07180 |
| always_think | 50 | 0.2516 | 0.1200 | 289 | 69.0% | 0.04437 |

## Metric Definitions

- **F1 Score**: Mean token-level accuracy (0-1).
- **Reduction %**: Token savings compared to `always_retrieve_k5` baseline.
- **Global Efficiency**: $\eta = \overline{F1} / \ln(1 + \overline{T})$. Measures quality per unit of 'log-effort'.

## Interpretation

- **Best F1:** always_retrieve_k5 (0.4910)
- **Most Efficient (Tokens):** always_think (289 tokens)
- **Best Global Efficiency:** always_retrieve_k5 (0.07180)
