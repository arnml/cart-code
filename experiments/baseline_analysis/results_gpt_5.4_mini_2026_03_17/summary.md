# Day 2 Baseline Analysis Summary — gpt-5.4-mini-2026-03-17

**Evaluation Date:** 2026-03-28 10:58:36
**Total Samples:** 149 questions
**Model:** gpt-5.4-mini-2026-03-17

## Results by Method

| Method | Count | F1 Score | Exact Match | Avg Tokens | Avg Cost | Efficiency |
|---|---|---|---|---|---|---|
| always_retrieve_k3 | 50 | 0.6755 | 0.5200 | 483 | $0.00063 | 0.10953 |
| always_retrieve_k5 | 50 | 0.7526 | 0.5800 | 741 | $0.00081 | 0.11440 |
| always_think | 49 | 0.5798 | 0.4286 | 142 | $0.00035 | 0.11741 |

## Key Observations

- **F1 Score**: Measures answer correctness (0-1, higher is better)
- **Exact Match**: Binary perfect answer match (0 or 1)
- **Tokens**: Total input + output tokens (fewer = cheaper)
- **Cost**: Estimated USD cost for all queries
- **Efficiency**: F1 / log(1 + tokens) — quality per unit cost

## Interpretation

- **Best F1:** always_retrieve_k5 (0.7526)
- **Cheapest (tokens):** always_think (142 tokens)
- **Best Efficiency:** always_think (0.11741)
