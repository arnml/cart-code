# Day 2 Baseline Analysis Summary — gpt-5.4-mini-2026-03-17

**Evaluation Date:** 2026-03-29 08:42:09
**Total Samples:** 150 questions
**Model:** gpt-5.4-mini-2026-03-17

## Results by Method (Global Statistics)

| Method | Count | F1 Score | EM | Avg Tokens | Reduction % | Global Efficiency |
|---|---|---|---|---|---|---|
| always_retrieve_k3 | 50 | 0.7002 | 0.5200 | 481 | 34.9% | 0.11334 |
| always_retrieve_k5 | 50 | 0.7414 | 0.5800 | 739 | 0.0% | 0.11222 |
| always_think | 50 | 0.6031 | 0.4600 | 146 | 80.2% | 0.12085 |

## Metric Definitions

- **F1 Score**: Mean token-level accuracy (0-1).
- **Reduction %**: Token savings compared to `always_retrieve_k5` baseline.
- **Global Efficiency**: $\eta = \overline{F1} / \ln(1 + \overline{T})$. Measures quality per unit of 'log-effort'.

## Interpretation

- **Best F1:** always_retrieve_k5 (0.7414)
- **Most Efficient (Tokens):** always_think (146 tokens)
- **Best Global Efficiency:** always_think (0.12085)
