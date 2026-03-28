# Day 2 Baseline Analysis Summary

**Evaluation Date:** 2026-03-28 09:57:52
**Total Samples:** 150 questions

## Results by Method

| Method | Count | F1 Score | Exact Match | Avg Tokens | Avg Cost | Efficiency |
|---|---|---|---|---|---|---|
| always_retrieve_k3 | 50 | 0.6457 | 0.4800 | 493 | $0.00011 | 0.10451 |
| always_retrieve_k5 | 50 | 0.7275 | 0.5600 | 753 | $0.00015 | 0.11033 |
| always_think | 50 | 0.4552 | 0.3600 | 173 | $0.00007 | 0.08936 |

## Key Observations

- **F1 Score**: Measures answer correctness (0-1, higher is better)
- **Exact Match**: Binary perfect answer match (0 or 1)
- **Tokens**: Total input + output tokens (fewer = cheaper)
- **Cost**: Estimated USD cost for all queries
- **Efficiency**: F1 / log(1 + tokens) — quality per unit cost

## Interpretation

- **Best F1:** always_retrieve_k5 (0.7275)
- **Cheapest (tokens):** always_think (173 tokens)
- **Best Efficiency:** always_retrieve_k5 (0.11033)
