# Baseline Results: claude-sonnet-4-6

**Evaluation Setup**: n=20, dataset=HotpotQA (distractor, validation split)

## Aggregate Metrics

| Method | Count | EM | F1 | Precision | Recall | Token-Eff | Cost-Eff |
|--------|-------|----|----|-----------|--------|-----------|----------|
| always_think | 20 | 0.350 | 0.471 | 0.482 | 0.608 | 0.100 | 1110.22 |
| retrieval_k3 | 20 | 0.550 | 0.736 | 0.812 | 0.708 | 0.119 | 482.45 |
| retrieval_k5 | 20 | 0.600 | 0.770 | 0.838 | 0.746 | 0.115 | 311.62 |
| retrieval_k10 | 20 | 0.600 | 0.770 | 0.838 | 0.746 | 0.104 | 155.79 |
