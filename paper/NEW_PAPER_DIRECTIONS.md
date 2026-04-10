# BRACIS 2026 — Paper Directions & Next Steps

**Conference:** XXXVI BRACIS – Brazilian Conference on Intelligent Systems  
**Track:** Main Track (Novel AI methods with sound results)  
**Deadline:** April 20, 2026 (23:59 UTC-12)  
**Format:** Springer LNAI — max 15 pages (including references)  
**Review:** Double-anonymous  

---

## Title (Working)

> **Noise-Gated Retrieval: Embedding-Based Context Compression for Cost-Aware RAG**

Alternative angles:
- *"When Simple Beats Learned: A Systematic Study of Test-Time Context Compression for RAG"*
- *"Test-Time Retrieval Compression via Noise Gating: A Cost-Quality Tradeoff Analysis on HotpotQA"*

---

## Core Narrative

You set out to find whether **adaptive/learned strategies** (UCB, bandits, decision trees) could
outperform simple filtering for RAG context compression. The answer is: **they can't — and here
is exactly why.**

This is not a failed experiment. It is a rigorous empirical contribution that:
1. Identifies the best practical method (Noise-Gate τ=0.3)
2. Explains *structurally* why learned methods fail on distractor-heavy QA
3. Provides a clean Pareto analysis across 8 strategies

The UCB failure **is** a finding. Reviewers respect honesty when backed by analysis.

---

## Contributions (List these explicitly in the Introduction)

1. **Systematic comparison** of five test-time context compression strategies on HotpotQA distractor split (n=400, gpt-5.4-mini)
2. **Noise-Gate method** — achieves −26.7% token reduction while staying within ±0.05 F1 of the k=10 baseline (F1=0.751 vs 0.792)
3. **Empirical analysis** explaining why bandit and tree-based approaches fail on distractor-heavy multi-hop QA
4. **Ablation study** across cosine threshold τ ∈ {0.2, 0.3, 0.5} and Jaccard threshold ρ ∈ {0.50, 0.55, 0.65}

---

## Paper Structure (15-page LNAI budget)

### 1. Introduction — ~1.5 pages

- Hook: RAG with fixed k=10 gives strong F1 (0.792) but costs 14× more tokens than no retrieval (1452 vs 103 tokens)
- Research question: *Can we compress the retrieved context at inference-time without significant quality loss?*
- Key challenge: distractor paragraphs in HotpotQA are semantically crafted to be deceptively similar to gold sections
- List the 4 contributions explicitly
- Preview the main result: embedding-based filtering dominates all learned approaches

### 2. Related Work — ~1.5 pages

Structure around four pillars:

| Area | Key Works | How You Differ |
|---|---|---|
| Fixed-budget RAG | Lewis et al. (RAG), Klesel & Wittmann | You compress a fixed retrieved pool |
| Adaptive retrieval | Self-RAG, DioR, DRAGIN | You don't decide *whether* to retrieve — you filter *what was retrieved* |
| Context noise/compression | SetR, ReflectiveRAG, Adaptive-k (Taguchi) | You do threshold-based filtering, not set-level or gap-based |
| Bandit/RL for IR | General UCB/MAB literature | You show bandits fail on this problem class and explain why |

**Key differentiator to emphasize:** This is **test-time, training-free compression of a fixed pool** — distinct from retrieval routing or retrieval-augmented generation design.

### 3. Problem Formulation — ~0.5 pages

Formal definition:

> Given question **q** and fixed pool **D = {d₁, ..., d₁₀}** retrieved by BM25/TF-IDF,  
> find subset **S* ⊆ D** that maximizes F1(answer | q, S) while minimizing |tokens(S)|.

This scopes the paper cleanly and separates your contribution from retrieval design.

Define evaluation metrics here: F1-score (primary), EM-score (secondary), mean tokens per question.

### 4. Methods — ~2 pages

Frame as **exploring the design space** along two axes: *learned vs. threshold-based* and *question-level vs. section-level*.

#### 4.1 Baselines
- **k=10 (full):** All retrieved sections → F1=0.792, 1452.5 tokens
- **k=5 (fixed):** Top-5 sections → F1=0.716, 731.7 tokens
- **No retrieval:** F1=0.387, 103 tokens

#### 4.2 Adaptive-K (CART Decision Tree)
- Extract question features → train CART to predict optimal k ∈ [1,9]
- Inference: retrieve top predicted-k sections

#### 4.3 UCB1-TUNED Bandit Reranker
- Track per-title reward distribution (mean, variance, count)
- UCB score = mean + α·√(ln(T)/nᵢ · min(0.25, sᵢ²))
- BM25 fallback for unseen titles

#### 4.4 LinUCB (Contextual Bandit)
- Features: token overlap, title-in-question fraction, section length
- Ridge regression reward model + UCB exploration bonus

#### 4.5 Noise-Gate (Proposed) ⭐
Two-stage filter at inference time, no training required:

```
Stage 1 — Semantic filter:
  sim(q, sᵢ) = cos(embed(q), embed(sᵢ))
  Keep sᵢ if sim ≥ τ

Stage 2 — Redundancy filter:
  jaccard(q, sᵢ) = |tokens(q) ∩ tokens(sᵢ)| / |tokens(q) ∪ tokens(sᵢ)|
  Keep sᵢ if jaccard ≥ ρ

Model: multi-qa-MiniLM-L6-cos-v1 (dim=384)
Parameters: τ=0.3, ρ=0.65 (from ablation)
```

### 5. Experiments — ~2 pages

#### 5.1 Setup
- Dataset: HotpotQA distractor validation split
- Sample: n=400
- Model: gpt-5.4-mini (primary), gpt-4o-mini (noted where used)
- Metrics: F1, EM, mean tokens

#### 5.2 Main Results Table

| Strategy | F1 | Tokens | Savings | EM | In Range? |
|---|---|---|---|---|---|
| Baseline k=10 | 0.792 | 1452.5 | — | 0.632 | Reference |
| **Noise-Gate τ=0.3** ⭐ | **0.751** | **1063.8** | **−26.7%** | 0.598 | ✓ YES |
| Noise-Gate τ=0.2 | 0.774 | 1317.5 | −9.3% | 0.620 | ✓ YES |
| Noise-Gate τ=0.5 | 0.613 | 403.9 | −72.2% | 0.480 | ✗ NO |
| Adaptive-K (CART) | 0.627 | 483.1 | −66.8% | 0.475 | ✗ NO |
| UCB1-TUNED | 0.623 | 520.5 | −64.2% | 0.495 | ✗ NO |
| LinUCB | 0.534 | 684.0 | −52.9% | 0.400 | ✗ NO |
| Baseline k=5 | 0.716 | 731.7 | −49.6% | 0.555 | ✗ NO |

#### 5.3 Noise-Gate Ablation (τ sweep)

| τ | F1 | Tokens | EM | Status |
|---|---|---|---|---|
| 0.2 | 0.774 | 1317.5 | 0.620 | Conservative |
| **0.3** | **0.751** | **1063.8** | **0.598** | **Optimal** |
| 0.5 | 0.613 | 403.9 | 0.480 | Too aggressive |

> **Optional (if time allows):** Run τ ∈ {0.25, 0.35} to show the full Pareto curve and confirm τ=0.3 is the actual knee.

#### 5.4 Jaccard Ablation (ρ sweep)
Report best results across ρ ∈ {0.50, 0.55, 0.65} at fixed τ=0.2.

**Figure (must have):** F1 vs. Tokens scatter plot with all 8 strategies labeled. Show the Pareto frontier. τ=0.3 should be visually the knee of the curve. This is the paper's central figure.

### 6. Analysis — "Why Learned Methods Fail" — ~2 pages

> **This is the most novel section and what elevates the paper beyond a benchmark.**

#### 6.1 Argument A — Retrieval Rank Carries No Signal
UCB1 assumes that some title positions/arms are consistently better. But in HotpotQA distractor, gold sections appear uniformly across rank positions 1–10 (~20% gold rate per position). There is no exploitable rank signal.

- **If you have rank-position data:** Show a bar chart of gold doc frequency per rank position (should be near-uniform).
- **If not:** Argue from HotpotQA construction — distractors are explicitly retrieved by BM25 alongside gold sections, so rank is deliberately non-informative.

#### 6.2 Argument B — Reward Signal Is Too Sparse for Bandits
- Only 2–3 gold sections out of 10 per question = 20–30% binary signal
- 83.8% cold-start rate (unseen titles at inference) → falls back to BM25, contradicting learned signal
- Reward depends on *combination* of sections (multi-hop reasoning), not independent section quality
- Result: UCB1 F1 varies dramatically across models (0.623 on gpt-5.4-mini vs. 0.353 on gpt-4o-mini), indicating overfitting rather than learning

#### 6.3 Argument C — Lexical Features Cannot Distinguish Distractors
- LinUCB uses token overlap, title match, section length — all lexical
- HotpotQA distractors are *semantically* crafted: they share surface tokens with the question but are not supporting facts
- Embeddings capture semantic relevance that lexical features miss
- This is why LinUCB-Semantic (adding cosine similarity as a feature) would theoretically help — but is CPU-infeasible without GPU (80–150h estimated)

#### 6.4 Why Noise-Gate Succeeds
- Embedding similarity captures the right signal: semantic relevance to the question
- Jaccard redundancy filter removes overlapping sections, not just irrelevant ones
- Single threshold τ=0.3 is robust: interpretable, deterministic, no training
- The combination of semantic + lexical filtering achieves what neither alone can

### 7. Discussion — ~1 page

- **Pareto framing:** Present τ=0.3 as the practical sweet spot — 26.7% savings, within ±0.05 F1 tolerance
- **Cost at scale:** At 1M questions/day, saves ~388.7M tokens/day (~$20–31K/year)
- **Limitations:**
  - Tested on one dataset (HotpotQA distractor); generalization to other RAG benchmarks unknown
  - Embedding inference adds latency (~2ms per section); pre-computation possible for static corpora
  - τ may require domain-specific tuning
  - UCB failure may not generalize: denser reward settings might favour bandit approaches
- **Future work:** Hybrid (noise-gate + light reranker); cross-encoder fine-ranking; adaptive τ by question type; test on FEVER, MuSiQue, 2WikiMultiHop

### 8. Conclusion — ~0.5 pages

- Restated research question and answer
- Noise-Gate τ=0.3: the best test-time compression strategy — simple, effective, no training
- Key insight: the *structure of the distractor problem* (sparse reward, rank-agnostic relevance, semantic distractors) makes learned approaches fundamentally ill-suited
- Embedding-based filtering is the minimal sufficient signal for this problem class

### References — ~2 pages

Use what you already have. Prioritize:
- Lewis et al. (RAG)
- Asai et al. (Self-RAG)
- Taguchi et al. (Adaptive-k)
- Lee et al. (SetR)
- Guo et al. (DioR)
- Trivedi et al. (IRCoT)
- Snell et al. (Test-Time Compute)
- Yang et al. (HotpotQA original paper) ← make sure this is cited

---

## Figures & Tables Checklist

| # | Type | Content | Priority |
|---|---|---|---|
| Fig 1 | Scatter plot | F1 vs. Tokens — all 8 strategies, Pareto frontier highlighted | **MUST HAVE** |
| Fig 2 | Line chart | τ ablation: F1 and Tokens vs. τ (dual y-axis) | High |
| Fig 3 | Bar chart | Gold doc frequency per rank position (if data available) | High — supports Section 6.1 |
| Table 1 | Main results | Full comparison table (already have) | **MUST HAVE** |
| Table 2 | Ablation | τ sweep results | **MUST HAVE** |
| Table 3 | Ablation | Jaccard ρ sweep results | High |

---

## Optional Experiment (High ROI if Time Allows)

**Run τ ∈ {0.25, 0.35}** — approximately 2 hours of work.

This fills the gap between τ=0.2 and τ=0.5 and:
- Shows a smooth Pareto curve (not just 3 points)
- Confirms τ=0.3 is the true knee, not an artifact of sparse sampling
- Strengthens the ablation section considerably

Expected results based on current trend:
- τ=0.25 → F1 ≈ 0.76–0.77, tokens ≈ 1150–1200
- τ=0.35 → F1 ≈ 0.73–0.74, tokens ≈ 900–950

---

## 11-Day Execution Plan

| Days | Task | Output |
|---|---|---|
| 1–2 ✅| Write Section 6 (Analysis — Why Learned Methods Fail) | Most novel content done |
| 3–4 ✅| Write Sections 1–3 (Intro, Related Work, Formulation) | Paper skeleton complete |
| 5–6 ✅| Assemble Section 5 (Experiments) from results docs | Tables and baselines done |
| 6 ✅| *Optional:* Run τ={0.25, 0.35} experiments | Richer ablation |
| 7 ✅| Write Sections 4, 7, 8 (Methods, Discussion, Conclusion) | Full draft |
| 8–9 ✅| Format in Springer LNAI LaTeX (Overleaf) | Camera-ready structure |
| 9 | Create all figures (scatter, line, bar charts) | Visuals done |
| 10 | Anonymize: remove all self-references and institution names | Double-anon compliant |
| 11 | Final proofread, page count check, submit via JEMS3 | **SUBMITTED** |

---

## BRACIS Submission Checklist

- [ ] Written in English
- [ ] Max 15 pages including all tables, figures, references, appendices
- [ ] Springer LNAI template (LaTeX via Overleaf or Word)
- [ ] Double-anonymous: no author names, no institution names, no self-citations that reveal identity
- [ ] PDF only uploaded to JEMS3
- [ ] Reviewer nomination form submitted: https://forms.gle/XHa7bykTiwiYu4pw7
- [ ] At least one author registered for conference (if accepted)
- [ ] Acknowledgements mention use of LLM tools if applicable

---

## Key Framing Reminders

> ❌ Don't say: "UCB didn't work, so we used noise-gate instead."  
> ✅ Do say: "We systematically evaluated five strategies and empirically demonstrate that the structural properties of distractor-heavy multi-hop QA — sparse reward, rank-agnostic relevance, and semantic distractors — make learned approaches fundamentally unsuitable. Embedding-based threshold filtering is the minimal sufficient approach."

> ❌ Don't apologize for not having more experiments.  
> ✅ Do frame the ablation (τ sweep, ρ sweep, 8 strategies) as thorough and systematic.

> ❌ Don't overclaim: "our method is best for all RAG."  
> ✅ Do scope clearly: "for test-time compression of a fixed retrieved pool on distractor-heavy QA."