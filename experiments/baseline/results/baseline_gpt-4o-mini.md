# Baseline Results: gpt-4o-mini

**Evaluation Setup**: n=2, dataset=HotpotQA (distractor, validation split)

## Aggregate Metrics

| Method | Count | EM | F1 | Precision | Recall | Token-Eff | Cost-Eff |
|--------|-------|----|----|-----------|--------|-----------|----------|
| always_think | 2 | 0.000 | 0.018 | 0.009 | 0.167 | 0.004 | 376.73 |
| retrieval_k3 | 2 | 0.000 | 0.043 | 0.025 | 0.167 | 0.007 | 626.04 |
| retrieval_k5 | 2 | 0.000 | 0.091 | 0.050 | 0.500 | 0.014 | 862.11 |
| retrieval_k10 | 2 | 0.000 | 0.068 | 0.037 | 0.500 | 0.009 | 301.42 |
