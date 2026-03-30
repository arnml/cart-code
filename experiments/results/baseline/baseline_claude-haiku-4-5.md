# Baseline Results: claude-haiku-4-5

**Evaluation Setup**: n=20, dataset=HotpotQA (distractor, validation split)

## Aggregate Metrics

| Method | Count | EM | F1 | Precision | Recall | Token-Eff | Cost-Eff |
|--------|-------|----|----|-----------|--------|-----------|----------|
| always_think | 20 | 0.400 | 0.435 | 0.451 | 0.487 | 0.093 | 4133.06 |
| retrieval_k3 | 20 | 0.550 | 0.736 | 0.812 | 0.708 | 0.119 | 1809.20 |
| retrieval_k5 | 20 | 0.500 | 0.682 | 0.747 | 0.708 | 0.102 | 1031.24 |
| retrieval_k10 | 20 | 0.500 | 0.671 | 0.738 | 0.696 | 0.091 | 504.65 |
