# UCB1-TUNED Reranker Analysis: gpt-5.4-mini

**Total samples:** 1200

## Performance by K Value

| K | Samples | EM | F1 | Precision | Recall | Avg Tokens |
|---|---------|----|----|-----------|--------|------------|
| 2 | 400 | 0.440 | 0.564 | 0.594 | 0.566 | 362 |
| 3 | 400 | 0.475 | 0.603 | 0.620 | 0.614 | 476 |
| 5 | 400 | 0.570 | 0.704 | 0.740 | 0.704 | 723 |

## Cold-start Statistics (BM25 Fallback)

| K | Total Samples | Unseen Count | Unseen % |
|---|---------------|--------------|----------|
| 2 | 400 | 246 | 61.5% |
| 3 | 400 | 301 | 75.2% |
| 5 | 400 | 335 | 83.8% |

## Overall Metrics

**Exact Match (EM):** 0.495
**F1 Score:** 0.623
**Precision:** 0.651
**Recall:** 0.628

**Avg Input Tokens:** 513.6
**Avg Output Tokens:** 6.9
**Avg Total Tokens:** 520.5
**Avg Cost:** $0.0001
