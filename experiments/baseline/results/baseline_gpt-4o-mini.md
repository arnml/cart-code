# Baseline Results: gpt-4o-mini

**Evaluation Setup**: n=2, dataset=HotpotQA (distractor, validation split)

## Aggregate Metrics

| Method | Count | EM | F1 | Precision | Recall | Token-Eff | Cost-Eff |
|--------|-------|----|----|-----------|--------|-----------|----------|
| always_think | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 |
| retrieval_k3 | 2 | 0.500 | 0.500 | 0.500 | 0.500 | 0.086 | 10010.01 |
| retrieval_k5 | 2 | 0.500 | 0.500 | 0.500 | 0.500 | 0.080 | 6337.14 |
| retrieval_k10 | 2 | 0.500 | 0.500 | 0.500 | 0.500 | 0.070 | 2673.08 |
