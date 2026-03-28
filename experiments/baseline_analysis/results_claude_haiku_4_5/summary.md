# Day 2 Baseline Analysis Summary — claude-haiku-4-5

**Evaluation Date:** 2026-03-28 11:49:26
**Total Samples:** 150 questions
**Model:** claude-haiku-4-5

## Results by Method

| Method | Count | F1 Score | Exact Match | Avg Tokens | Avg Cost | Efficiency |
|---|---|---|---|---|---|---|
| always_retrieve_k3 | 50 | 0.3832 | 0.1800 | 649 | $0.00141 | 0.06007 |
| always_retrieve_k5 | 50 | 0.4764 | 0.2400 | 935 | $0.00166 | 0.07017 |
| always_think | 50 | 0.2825 | 0.1200 | 279 | $0.00106 | 0.05142 |

## Key Observations

- **F1 Score**: Measures answer correctness (0-1, higher is better)
- **Exact Match**: Binary perfect answer match (0 or 1)
- **Tokens**: Total input + output tokens (fewer = cheaper)
- **Cost**: Estimated USD cost for all queries
- **Efficiency**: F1 / log(1 + tokens) — quality per unit cost

## Interpretation

- **Best F1:** always_retrieve_k5 (0.4764)
- **Cheapest (tokens):** always_think (279 tokens)
- **Best Efficiency:** always_retrieve_k5 (0.07017)
