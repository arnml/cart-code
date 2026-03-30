# Baseline Results: gpt-4o-mini

**Evaluation Setup**: n=20, dataset=HotpotQA (distractor, validation split)

## Aggregate Metrics

| Method | Count | EM | F1 | Precision | Recall | Token-Eff | Cost-Eff |
|--------|-------|----|----|-----------|--------|-----------|----------|
| always_think | 20 | 0.200 | 0.258 | 0.275 | 0.250 | 0.057 | 17947.38 |
| retrieval_k3 | 20 | 0.500 | 0.631 | 0.671 | 0.625 | 0.103 | 9057.91 |
| retrieval_k5 | 20 | 0.550 | 0.688 | 0.760 | 0.658 | 0.105 | 6529.29 |
| retrieval_k10 | 20 | 0.650 | 0.797 | 0.847 | 0.783 | 0.110 | 3768.13 |
