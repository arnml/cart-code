# Baseline Results: gpt-4o-mini

**Evaluation Setup**: n=10, dataset=HotpotQA (distractor, validation split)

## Aggregate Metrics

| Method | Count | EM | F1 | Precision | Recall | Token-Eff | Cost-Eff |
|--------|-------|----|----|-----------|--------|-----------|----------|
| always_think | 10 | 0.300 | 0.417 | 0.450 | 0.400 | 0.093 | 28950.33 |
| retrieval_k3 | 10 | 0.500 | 0.636 | 0.625 | 0.650 | 0.104 | 9248.36 |
| retrieval_k5 | 10 | 0.600 | 0.747 | 0.800 | 0.717 | 0.115 | 7356.81 |
| retrieval_k10 | 10 | 0.700 | 0.932 | 0.975 | 0.917 | 0.130 | 4810.80 |
