# Baseline Results: gpt-5.4-mini

**Evaluation Setup**: n=20, dataset=HotpotQA (distractor, validation split)

## Aggregate Metrics

| Method | Count | EM | F1 | Precision | Recall | Token-Eff | Cost-Eff |
|--------|-------|----|----|-----------|--------|-----------|----------|
| always_think | 20 | 0.300 | 0.383 | 0.400 | 0.375 | 0.084 | 15372.09 |
| retrieval_k3 | 20 | 0.600 | 0.702 | 0.756 | 0.683 | 0.115 | 7660.22 |
| retrieval_k5 | 20 | 0.500 | 0.627 | 0.700 | 0.596 | 0.096 | 4438.85 |
| retrieval_k10 | 20 | 0.550 | 0.688 | 0.760 | 0.658 | 0.094 | 2354.40 |
