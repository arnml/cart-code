# Day 2 Baseline Analysis Summary — gpt-4o-mini

**Evaluation Date:** 2026-03-28 12:15:30
**Total Samples:** 150 questions
**Model:** gpt-4o-mini

## Results by Method

| Method | Count | F1 Score | Exact Match | Avg Tokens | Avg Cost | Efficiency |
|---|---|---|---|---|---|---|
| always_retrieve_k3 | 50 | 0.6247 | 0.4600 | 492 | $0.00011 | 0.10113 |
| always_retrieve_k5 | 50 | 0.7109 | 0.5400 | 753 | $0.00015 | 0.10779 |
| always_think | 50 | 0.4699 | 0.3400 | 173 | $0.00007 | 0.09240 |

## Key Observations

- **F1 Score**: Measures answer correctness (0-1, higher is better)
- **Exact Match**: Binary perfect answer match (0 or 1)
- **Tokens**: Total input + output tokens (fewer = cheaper)
- **Cost**: Estimated USD cost for all queries
- **Efficiency**: F1 / log(1 + tokens) — quality per unit cost

## Interpretation

- **Best F1:** always_retrieve_k5 (0.7109)
- **Cheapest (tokens):** always_think (173 tokens)
- **Best Efficiency:** always_retrieve_k5 (0.10779)
