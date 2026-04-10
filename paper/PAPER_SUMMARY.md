# Paper Structure Summary

**Title:** Cost-Aware Test-Time Retrieval Control for RAG

---

## 1. Introduction

We ask: "Can we answer questions as accurately as using all 10 retrieved documents, but using fewer tokens?" 

Current RAG systems use fixed k=10 retrieval (all paragraphs), which gives excellent F1 (0.792) but costs 14× more tokens than just thinking. We explore whether smarter filtering can match this quality with fewer tokens.

---

## 2. Related Work

We discuss four areas:

- **Fixed-Budget Retrieval:** Standard k=10 RAG (Lewis et al., etc.)
- **Adaptive Retrieval:** Learning when to retrieve (Self-RAG, DioR, etc.)
- **Context Selection Beyond Relevance:** SetR argues for set-level selection; Adaptive-k uses similarity gaps
- **Noise & Test-Time Compute:** How bad context hurts quality; adaptive compute allocation

---

## 3. Methods (Training-Free Controller)

We propose combining three ideas:

- **Adaptive-k:** Select documents via largest similarity-score gap
- **UCB-Cost Policy:** Route between "think only" vs "retrieve" based on expected quality vs token cost
- **Noise Gate:** Two-stage filter — (1) remove low cosine-similarity paragraphs, (2) remove redundant overlaps via Jaccard similarity

---

## 4. Experiments

**Dataset:** HotpotQA distractor split (10 docs per question; 2 supporting facts, 8 distractors)

**Models:** GPT-4o-mini, GPT-5.4-mini (primary); Claude Haiku, Sonnet (auxiliary)

**Sample size:** n=400 for main runs

**Baselines:** no-retrieval, k=3, k=5, k=10

---

## 5. Results

- **Baseline finding:** Fixed k=10 is strong (F1=0.792) but expensive (1452 tokens). No retrieval: F1=0.387, 103 tokens.

- **Adaptive-k alone:** Selects only ~3 docs, F1=0.627 (too low, worse than k=3 baseline)

- **Noise Gate sweep:** Tested three cosine thresholds (τ ∈ {0.2, 0.3, 0.5}):
  - τ=0.2 (lenient): F1=0.774, 1317 tokens (best quality)
  - τ=0.3 (medium): F1=0.751, 1064 tokens (26.8% savings, near-optimal tradeoff) ⭐
  - τ=0.5 (strict): F1=0.613, 404 tokens (too aggressive)

- **Jaccard ablation:** Tested redundancy thresholds (ρ ∈ {0.50, 0.55, 0.65}); best F1 = 0.782 at (ρ,τ)=(0.50,0.2)

- **UCB1-TUNED rank exploration:** Learning which document *positions* are valuable fails because all ranks have ~20% gold rate (no signal).

---

## 6. Discussion

- Retrieval helps a lot (F1 +0.4 from no-retrieval to k=10)
- Naive Adaptive-k doesn't work because distractor paragraphs are deceptively similar
- Simple noise filtering (cosine + Jaccard) helps but isn't a complete solution
- Medium cosine threshold (τ=0.3) is a practical sweet spot: saves 26.8% tokens while keeping F1 within ±0.05 of lenient setting
- Bandit learning fails here because HotpotQA's retrieval order doesn't correlate with relevance

---

## 7. Conclusion

Fixed k=10 costs ~14× more tokens for a quality bump. Adaptive-k is ineffective. A simple noise gate (especially τ=0.3) offers a better frontier: ~27% token savings while preserving quality. Future work: combine noise gate with stronger routing or set-level selection for larger savings.

---

## Key Takeaway

Embedding-based relevance filtering + redundancy removal can cut 27% of tokens without major quality loss on HotpotQA, but more sophisticated mechanisms are needed for larger savings.

---

## References

- Lewis et al. — RAG (Retrieval-Augmented Generation)
- Klesel & Wittmann — Plain RAG Survey
- Zhao et al. — AIGC Survey on RAG
- Yu et al. — RankRAG
- Lee et al. — SetR (Set-based Retrieval)
- Verma et al. — ReflectiveRAG
- Asai et al. — Self-RAG (Retrieval Control)
- Su et al. — DRAGIN
- Zhang et al. — RetrievalQA
- Guo et al. — DioR
- Hwang et al. — RA-RAG
- Taguchi et al. — Adaptive-k
- Goldstein et al. — Maximum Marginal Relevance (MMR)
- Trivedi et al. — IRCoT (Interleaved Retrieval & Reasoning)
- Yao et al. — ReAct
- Snell et al. — Test-Time Compute
- Li et al. — C2-Leva (Benchmark Contamination)
- Yi et al. — Membership Inference in Benchmarks
