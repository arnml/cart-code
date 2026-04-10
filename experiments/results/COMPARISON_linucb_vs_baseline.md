# LinUCB vs Baseline Comparison (gpt-4o-mini)

## Results Table

| Approach | K | F1 | EM | Tokens | Notes |
|----------|---|----|----|--------|-------|
| **Baseline** | 0 | 0.305 | 0.245 | 97 | No context (k=0 fallback) |
| **Baseline** | 3 | 0.588 | 0.479 | 469 | Fixed 3-retrieval |
| **Baseline** | 5 | 0.632 | 0.489 | 730 | Fixed 5-retrieval |
| **Baseline** | 10 | 0.707 | 0.559 | 1,451 | All 10 paragraphs |
| **LinUCB** | 2 | 0.395 | 0.310 | 344 | Learned selection |
| **LinUCB** | 3 | 0.424 | 0.340 | 457 | Learned selection |
| **LinUCB** | 5 | 0.534 | 0.400 | 684 | Learned selection |

## Key Findings

### 1. LinUCB Shows Promise But Lags Baseline

- **LinUCB k=5 (F1=0.534)** vs **Baseline k=3 (F1=0.588)**: -9% gap
- **LinUCB k=5 (F1=0.534)** vs **Baseline k=5 (F1=0.632)**: -15% gap
- **LinUCB k=5 (F1=0.534)** vs **Baseline k=10 (F1=0.707)**: -24% gap

### 2. Token Efficiency

- **LinUCB k=5**: 684 tokens (less than baseline k=5 with 730 tokens)
- **LinUCB k=3**: 457 tokens (essentially same as baseline k=3 with 469 tokens)
- Token savings: ~6% at k=5, negligible at k=3

### 3. Feature Signal Works

The diagnostic confirmed that **title_in_question** has strong predictive signal (2.61x ratio).
LinUCB correctly leverages this signal, but the feature alone is not sufficient to match baseline performance.

### Why Baseline Wins

The baseline uses **dense retrieval (BM25/embedding-based)**, which captures:
1. Semantic similarity between question and section
2. Named entity matching
3. Paraphrase recognition
4. Syntactic variations

LinUCB's feature set (`title_in_question`, `question_length`, `section_length`) captures:
1. Lexical overlap (title_in_question)
2. Document/question scale (length features)

### Why LinUCB Still Valuable

Despite lower F1, LinUCB demonstrates:
- **Interpretability**: Clear feature-based selection
- **Minimal cold-start**: Works immediately without pre-trained embeddings
- **Explainability**: Can show which question words matched which section titles
- **Online learning**: Can adapt features online with user feedback

## Recommendation

**Use baseline for best performance** (k=10: F1=0.707, k=5: F1=0.632)
**Use LinUCB if interpretability is critical** (F1~0.53 at k=5 with clear reasoning)

To improve LinUCB further, add:
- Named entity overlap (from diagnostic showed promise)
- Token overlap with better normalization
- Question type features (bridge vs comparison)
