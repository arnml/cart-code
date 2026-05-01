# Paper Specification — Noise-Gated Retrieval for Cost-Aware RAG

This document is a declarative specification for the BRACIS submission. Each section states what the paper must contain, in what order, with what floats. Per-paragraph bullets are added in a follow-up pass.

---

## 0. Style and Voice

The paper must follow BRACIS / Springer LNCS conventions for academic writing.

**Required register.**
- Formal academic tone throughout. Third-person and "we" only when introducing contributions or describing the experimental setup.
- Hypotheses, methods, and findings stated as observations, not claims of novelty.
- Past tense for experiments performed; present tense for definitions and standing claims; future tense only in *Future Work*.

**Forbidden patterns.**
- Marketing or commercial language: *amazing, powerful, magical, simple but powerful, novel, groundbreaking, surprisingly, beautifully, dramatically, remarkable*.
- First-person singular ("I", "my").
- Loaded adverbs: *clearly, obviously, of course, simply*.
- Em-dashes for narrative emphasis; rhetorical questions in body text; exclamation marks anywhere.
- Self-referential narration ("we now turn to", "as we discussed earlier") beyond minimal forward/backward references.

**Required practices.**
- Each section opens with one declarative sentence stating its purpose.
- Numbers are reported with consistent precision: F1 and EM to 3 decimals, mean tokens to 1 decimal, percentages to 1 decimal, costs in scientific notation.
- All quantitative claims must reference Tab 1, Tab 2, Tab 3, Fig 2, or be cited.
- Symbols are introduced before use. Equations are numbered and referenced explicitly.
- Citations follow LNCS numbered style (`splncs04`, rendered as `[n]`); every `\bibitem` retained at submission must be cited at least once. A reservoir of uncited entries is allowed during drafting; see §0.1.
- Hyperparameters are reported in §5.1 (Setup) and not re-defined in §5.2–§5.4.

**Diction list (use / avoid).**
- Use: *evaluate, propose, characterize, observe, find, indicate, consistent with, in line with, sufficient, reduce, preserve, transfer*.
- Avoid: *crush, beat, smash, blow away, breakthrough, killer, sweet spot, magic, juice*.

**Length.** LNCS 12-page limit including references. Target body length is approximately 4800 words, distributed across sections in round numbers that sum exactly to 4800:

| Section | Words |
|---|---:|
| Abstract | 200 |
| 1. Introduction | 700 |
| 2. Related Work | 500 |
| 3. Problem Formulation | 250 |
| 4. Methods | 1100 |
| 5. Experiments | 1100 |
| 6. Failure Mode Analysis | 600 |
| 7. Discussion | 250 |
| 8. Conclusion | 100 |
| **Total** | **4800** |

These targets are guidance, not hard limits: ±10 % per section is acceptable as long as the section budgets stay close to the totals above. Captions, table cells, and bibliography do not count.

---

## 0.1 Citation Guidelines

The paper must follow Springer LNCS / BRACIS bibliography conventions.

**Bibliography style.**
- LNCS uses numbered citations rendered as `[n]` (the `splncs04` bibliography style). Citations are placed immediately after the author name, method name, or claim they support.
- During drafting, an uncited `\bibitem` reservoir is permitted so candidate references can be added or removed without churn. Before submission, every entry remaining in the bibliography must be cited at least once in the body; any reservoir entry not adopted is removed at that point.
- Avoid duplicate \bibitem entries for the same paper (one canonical key per work).

**Where to cite.**
- *First mention of any prior method or dataset:* cite immediately. Example: "Self-RAG~\cite{asai2024selfrag}", "HotpotQA distractor~\cite{yang2018hotpotqa}".
- *Definitional or established claims:* cite the canonical reference, not a recent paper that re-states the claim.
- *Empirical claims about the field* (e.g., "fixed top-$k$ remains the deployed default"): cite a recent survey or a representative deployment study.
- *Baselines and comparators:* cite at first mention in §4 and at first quantitative comparison in §5.
- *Bandit and learning-to-rank techniques:* cite the original algorithmic paper, not a recent application.

**Where not to cite.**
- Own contributions, own results, own experimental details.
- Mathematical identities the reader can verify.
- Self-evident background that does not depend on a specific source.
- Inside captions of figures or tables, unless the float reuses material from a prior work.

**Citation form.**
- Multiple sources for one claim: `\cite{a,b,c}` (renders as `[1,2,3]`), not three separate `\cite{}` calls in one sentence.
- Method-name-first phrasing is preferred over author-name-first: "Adaptive-$k$~\cite{taguchi2025adaptivek}" rather than "Taguchi et al.~\cite{taguchi2025adaptivek} propose Adaptive-$k$".
- Author-name phrasing is acceptable when the method has no canonical name: "Lewis et al.~\cite{lewis2020rag}".
- Do not cite the same source more than once per paragraph; if the paragraph centers on a single work, cite at first mention only.
- Inline citations sit *before* punctuation: "as in prior work~\cite{x}." not "as in prior work.~\cite{x}".

**Bibliography entry hygiene.**
- Each `\bibitem` includes: author list (Lastname, F.), full title, venue (italicized via `\emph{}`), volume / pages where applicable, year. ArXiv preprints labeled "arXiv preprint arXiv:NNNN.NNNNN" without an alternative venue.
- Use the venue name a reader would recognize (e.g., "Proceedings of EMNLP", "International Conference on Learning Representations (ICLR)") rather than the publisher's series name alone.
- Sort within the file by author surname for ease of audit; the rendered order is determined by `splncs04` regardless.

**Coverage requirements (must cite at least once each).**
- *RAG foundations:* `lewis2020rag`.
- *Distractor / multi-hop benchmark properties:* `yang2018hotpotqa`, `chiang2024plausible`, `trivedi2023interleave`.
- *Datasets:* `yang2018hotpotqa` (HotpotQA), `ho-etal-2020-constructing` (2WikiMultiHopQA), `trivedi-etal-2022-musique` (MuSiQue-Ans).
- *Adaptive retrieval:* `asai2024selfrag`, `guo2025dior`, `su2024dragin`.
- *Multi-agent / iterative:* `chang2024mainrag`, `nahid2025prism`.
- *Compression:* `jeong2025ecorag`, `singh2025chunkrag`, `xiong2025drrag`, `taguchi2025adaptivek`, `verma2026reflectiverag`.
- *Set selection / reranking:* `lee2025setr`, `yu2024rankrag`.
- *Bandits:* `auer2002ucb1`, `chu2011linucb`, `thompson1933`.
- *Diversity / MMR:* `goldstein1998mmr`.
- *Deployment / best practices:* `klesel2025rag`, `li2025bestpractices`.

Entries currently in the `paper.tex` bibliography that have no citation in the present narrative (e.g., `li2024c2leva`, `snell2025ttc`, `wang2024bestpractices`, `yi2026membership`, `zhang2024retrievalqa`, `zhao2026aigcsurvey`, `jiang2016adaptive`, `mao2021gar`, `wang2023query2doc`, `gao2023hyde`, `karpukhin2020dpr`, `nogueira2019bert`, `sun2023rankgpt`, `qin2024prp`, `yao2023react`, `hwang2025rarag`) must either be cited in the appropriate paragraph or removed before submission.

---

## 0.2 Cross-Section Conventions

These conventions apply across the paper and are stated once in §5.1, then assumed.

- **Generation model.** All experiments use a single generation model: `gpt-5.4-mini`. The model is selected as the cost-effective tier from a major commercial provider (OpenAI), so that token-cost reductions translate directly to deployment savings rather than to frontier-model overhead. Fixing the model controls for tokenization, pricing, and provider-specific prompt behavior.
- **Embedding model.** `multi-qa-MiniLM-L6-cos-v1` (384 dimensions). Pre-trained on QA corpora; precomputed and cached.
- **Sample sizes.** $n{=}400$ for HotpotQA distractor (validation split). $n{=}100$ for transfer pilots on 2WikiMultiHopQA and MuSiQue-Ans.
- **Metrics.** Token-level F1 (primary), exact match (secondary, reported in text only), mean total tokens per example, mean per-query USD cost.
- **Tolerance band.** A strategy is *quality-preserving* if its F1 is within $\pm 0.05$ of fixed $k{=}10$ on the same dataset.
- **Hyperparameters of the proposed method.** Cosine threshold $\tau\in\{0.2, 0.25, 0.3, 0.35, 0.5\}$. Jaccard threshold $\rho\in\{0.50, 0.55, 0.65\}$. Recommended operating point for HotpotQA: $(\tau{=}0.3, \rho{=}0.65)$.

**Float budget (final): 2 figures, 3 tables, 1 algorithm, 7 equations.**

---

## 1. Introduction

The introduction is structured as **six paragraphs**, in this order. Paragraph plan follows advisor guidance (see "Advisor preferences" note at the end of this section).

### P1 — RAG as architecture for grounding language model outputs
- Open: "Retrieval-augmented generation is an architecture for reducing hallucination in large language models by grounding generation in retrieved evidence."
- Briefly survey retrieval mechanisms (sparse BM25-style retrieval, dense embedding-based retrieval, hybrid retrieval) so the reader understands the layer Noise-Gate operates on.
- State that the LLM is the state-of-the-art final layer of a RAG pipeline, and that the question of *what context to pass* to that layer is the focus of this work.

### P2 — Fixed-$k$ retrieval is the deployed default, and explain why
- Open: "Yet, most retrieval systems retrieve a fixed number of documents, regardless of the query."
- Argue *why* fixed-$k$ persists despite known weaknesses: training learned selectors per domain is expensive in data and compute; deployment infrastructure favors a single deterministic rule; the fixed-budget policy is straightforward to implement and reason about.
- Establish the cost consequence with concrete numbers: at $k{=}10$ on HotpotQA distractor, mean cost per query is 1452.5 tokens versus 103.5 tokens with no retrieval (a ratio of approximately $14\times$), at F1 0.792 versus 0.387.

### P3 — How the field has tried to address fixed-$k$ context-selection
- Detail the alternatives to fixed-$k$: training-based adaptive retrieval, multi-agent / iterative filtering, compression-based methods, and gap-based truncation.
- For each family, state in one phrase its weakness in deployment: training-based methods need supervision and retraining for new domains; multi-agent methods incur multiple LLM calls per query; compression-based methods need additional infrastructure; gap-based methods rest on a strong assumption about embedding-rank monotonicity.
- This is also where the **comparator-ladder** sentence lands (one orienting line): the three learned comparators evaluated in this paper — gap-based truncation, title-arm bandit, contextual bandit — form a progression of increasing structure exploited.

### P4 — Distractor-heavy / multi-hop data makes context-selection harder
- Open: "In particular, for multi-hop or distractor-heavy data the problem can be even worse."
- Identify three structural reasons distractor-heavy QA resists naive filtering: BM25 distractors are topically adjacent to gold evidence, multi-hop reasoning requires jointly complementary evidence sets, and the fixed-pool 2-gold-plus-8-distractors format is a structured filtering task rather than an open-domain retrieval task.
- State, citing prior work, that **data structure is a primary determinant of learned-selector performance** on these benchmarks — preview the failure-mode finding without yet quantifying it.

### P5 — Wrap-up, research questions, and contributions
- One sentence summarizing the gap: a training-free filter that exploits dense semantic similarity has not been systematically evaluated against learned selectors on distractor-heavy multi-hop QA.
- Introduce **Noise-Gate** as a training-free two-stage filter combining cosine-similarity thresholding with Jaccard-overlap redundancy removal.
- State the **research questions** explicitly:
  - RQ1: Can a training-free test-time filter reduce token cost on distractor-heavy multi-hop QA without degrading answer quality?
  - RQ2: Why do learned context-selection strategies underperform on this class of benchmarks?
  - RQ3: Do the same threshold settings transfer across distractor-heavy multi-hop datasets?
- State the **contributions**, as a numbered list:
  1. Noise-Gate as a training-free filter with $-26.7\%$ token reduction at F1 0.764 versus 0.792 at $k{=}10$, validated by ablations and transferred to two further benchmarks.
  2. A structural failure analysis of the three learned strategies on distractor-heavy multi-hop QA.
  3. A matched-sample comparison of five context-compression strategies under a single generation model and prompt.
- Close with the practical implication: at one million queries per day on `gpt-5.4-mini`, the annualized saving is approximately USD 28,000.

### P6 — Paper organization
- Open: "The remainder of this paper is organized as follows."
- One sentence per section, naming what each section delivers (Related Work, Problem Formulation, Methods, Experiments, Failure Mode Analysis, Discussion, Conclusion).

**No floats in this section.**

**Advisor preferences (must follow when feasible).** The six-paragraph plan above mirrors the advisor's recommendation. Maintain the openings (P2 begins with "Yet, most retrieval systems retrieve a fixed number of documents…"; P4 begins with "In particular, for multi-hop or distractor-heavy data…"; P6 begins with "The remainder of this paper is organized as follows…"). The plan is binding for paragraph order and openings; the wording within each paragraph is free as long as it satisfies the bullets.

## 2. Related Work

Two organizational layouts are acceptable. **Layout A** (default, advisor-preferred) opens with one paragraph naming the three relevant research lines, then follows with one paragraph per family. **Layout B** uses three named subsections (one per research line) when the paragraphs are long enough to justify the structural overhead. Pick whichever reads cleaner at the LNCS page count; do not introduce subsections shorter than half a column.

### Opening paragraph (research-line framing, advisor-preferred)
- State that this work draws on three lines of research:
  1. **RAG for fact-checking and truth-grounding** — establishing why grounded generation matters and why selecting *useful* context is a quality-critical step.
  2. **Multi-hop QA and distractors** — the difficulty of distinguishing gold evidence from BM25-retrieved distractors when their embeddings are close.
  3. **Cost-aware evaluation of RAG pipelines** — supporting the claim that token cost is a meaningful metric to track in this setting.
- This paragraph names the lines and cites the canonical references; the following paragraphs go into method-by-method detail.

### Body paragraphs (one per family, in this order)
- **Fixed-budget retrieval.** Anchor the baseline. Cite Lewis et al. (2020) and a recent survey establishing fixed top-$k$ as the deployed default. Connect to research line 1 (RAG / truth-grounding).
- **Training-based adaptive retrieval.** Self-RAG, DioR, DRAGIN. State that they require additional supervision or model modification, which is a deployment cost.
- **Multi-agent / iterative filtering.** MAIN-RAG, PRISM. State that they are training-free but incur multiple LLM calls per query, an inference cost. Connect to research line 3 (cost-aware evaluation).
- **Compression-based methods.** ECoRAG, ChunkRAG, DR-RAG, Adaptive-$k$. Distinguish chunk-level filtering, evidentiality scoring, classifier-based selection, and similarity-gap truncation.
- **Bandits and learning-to-rank.** UCB, Thompson Sampling, LinUCB; SetR and RankRAG for set selection. Cite as classical decision-making techniques applied here as comparators, not as RAG-specific systems. Connect to research line 2 (multi-hop / distractor difficulty).

### Closing sentence
- Position Noise-Gate as a paragraph-level, training-free filter that combines dense semantic similarity with redundancy removal, sitting at the intersection of the three research lines.

**No floats in this section.**

**Advisor preferences (try to satisfy when feasible).** The opening research-line paragraph and the optional three-subsection structure are advisor recommendations; treat them as preferences rather than hard requirements. If the body fits cleanly into the four-family order without subsections, that is acceptable. The non-negotiable elements are: (i) the opening paragraph names the three research lines, (ii) the four families are covered, and (iii) the closing sentence positions Noise-Gate.

## 3. Problem Formulation

The section must:

- Define the question $q$, the candidate context $D=(d_1,\dots,d_n)$, and the selected subsequence $S\preceq D$. State that retrieval itself is fixed; only the selection over $D$ is under control.
- Define the answer $\hat{a}=g(q,S)$ and the reference $a^*$.
- Introduce the cost-aware objective:

  **Eq 1.** $\quad J(S) = F_1(\hat{a}, a^*) - \lambda\, C(q, S),\quad \lambda > 0,$ where $C(q,S)$ is the total token cost of the model call.

- State the evaluation choice: F1 as primary quality metric, tokens as cost metric; $\lambda$ left implicit and replaced with the Pareto presentation of §5.2.
- Close with **Fig 1** as the bridge into Methods: a side-by-side gold-evidence vs BM25-distractor example illustrating why selection over $D$ is non-trivial.

## 4. Methods

The section opens with one sentence: five strategies are evaluated, ordered by the structure they exploit (none, similarity gap, arm reward, lexical features, dense semantic similarity). Each subsection opens with one declarative sentence stating the design hypothesis the strategy embodies, then defines it formally.

### 4.1 Baselines
- Define the four fixed-$k$ anchors $k\in\{0,3,5,10\}$. Reference numbers (HotpotQA, $n{=}400$): F1 / total tokens = 0.387 / 103.5 ($k{=}0$); 0.651 / 471.3 ($k{=}3$); 0.716 / 731.7 ($k{=}5$); 0.792 / 1452.5 ($k{=}10$).
- Establish $k{=}10$ as the quality reference and $k{=}0$ as the parametric-knowledge floor.

### 4.2 Adaptive-$k$
- Design hypothesis: the largest gap in sorted similarity scores marks the boundary between relevant and irrelevant context.
- **Eq 2.** $\quad k^* = \arg\max_{1\le i<n}(s_i - s_{i+1})$.
- Implementation note: $\delta=0.08$ minimum-gap floor; if $\max(s_i - s_{i+1}) < \delta$, fall back to $k=\min(5,n)$.

### 4.3 UCB1-TUNED
- Design hypothesis: per-arm reward statistics learned from a training split, with variance-aware exploration, can rank arms more accurately than a static rule, at light training cost.
- Welford's online estimator for per-title mean and variance over the HotpotQA training split ($n{=}90{,}447$). Binary reward: 1 if title appears in supporting facts, 0 otherwise.
- **Eq 3.** $\quad \mathrm{score}(a) = \mu_a + \sqrt{\frac{\ln T}{n_a}\,\min(0.25, \sigma_a^2)}$.
- Top-$k$ titles by score; BM25 fallback for arms unseen in training.

### 4.4 LinUCB
- Design hypothesis: parameterizing reward as a linear function of context features generalizes beyond the training arm vocabulary, addressing UCB1-TUNED's cold-start regime.
- **Eq 4.** $\quad x(q,d) = [f_1, f_2, f_3]^\top$, with $f_1$ the title-in-question fraction, $f_2$ the normalized question length, $f_3$ the normalized section length. Ridge update $\hat{\theta} = A^{-1}b$, with $A=I+\sum_t x_t x_t^\top$ and $b=\sum_t r_t x_t$.
- **Eq 5.** $\quad p(q,d) = \hat{\theta}^\top x(q,d) + \alpha\sqrt{x(q,d)^\top A^{-1} x(q,d)}$.
- Top-$k$ documents by score.

### 4.5 Noise-Gate (proposed)
- Design hypothesis: dense semantic similarity provides a discriminative signal that distinguishes gold evidence from BM25-retrieved distractors when surface and rank features cannot. A subsequent redundancy filter removes near-duplicates introduced by paragraph-level granularity.
- **Stage 1 (relevance).**
  **Eq 6.** $\quad \cos(q, d_i) = \dfrac{e_q \cdot e_i}{\|e_q\|\,\|e_i\|} \ge \tau$.
- **Stage 2 (redundancy).**
  **Eq 7.** $\quad \max_{d_j\in S}\,\dfrac{|d_i\cap d_j|}{|d_i\cup d_j|} \le \rho$.
- The gate operates on precomputed embeddings; no per-dataset training is performed.
- **Algorithm 1.** Pseudocode of the two-stage filter.

## 5. Experiments

### 5.1 Setup
- Restate the cross-section conventions of §0.1: model, embeddings, samples, metrics, tolerance band, hyperparameters.
- State the dataset roles in three sentences: HotpotQA distractor (reference setting, $n{=}400$); 2WikiMultiHopQA (closest structural transfer test, $n{=}100$); MuSiQue-Ans (semantic-generalization test with nested context flattened to paragraphs, $n{=}100$).
- State that learned strategies (UCB1-TUNED, LinUCB) are trained per-dataset on the corresponding training split; fixed baselines, Adaptive-$k$, and Noise-Gate are applied unchanged across datasets.
- **No standalone dataset table.**

### 5.2 Main Results
- One paragraph reading the headline numbers off Tab 1 for HotpotQA: Noise-Gate at $\tau{=}0.3$ achieves F1 0.764 at 1065.3 tokens, $-26.7\%$ versus $k{=}10$ and within the $\pm 0.05$ tolerance band; no learned strategy enters the band.
- One paragraph contrasting the learned strategies on HotpotQA: Adaptive-$k$ at F1 0.627, UCB1-TUNED at F1 0.704 ($k{=}5$), LinUCB at F1 0.620 ($k{=}5$). State the gap to $k{=}5$ baseline (F1 0.716).
- **Tab 1.** Unified Main + Transfer.
- **Fig 2.** Pareto F1-vs-tokens with $\tau$-sweep overlay.

### 5.3 Transfer Evaluation
- One paragraph reading the 2WikiMultiHopQA rows of Tab 1: Noise-Gate at $\tau{=}0.25$ gives F1 0.662 at 810.9 tokens, a Pareto improvement on both axes versus $k{=}10$ (F1 0.617 at 1040.1 tokens). Threshold shifts one step from the HotpotQA-optimal $\tau{=}0.3$.
- One paragraph reading the MuSiQue-Ans rows: $\tau{=}0.3$ transfers directly as the operating point with F1 0.551 at 1193.2 tokens versus $k{=}10$ at F1 0.499 and 1342.9 tokens. No threshold change.
- One sentence summarizing: across three benchmarks, Noise-Gate matches or exceeds fixed $k{=}10$ at lower token cost, with at most one threshold step of recalibration.
- **No additional tables.**

### 5.4 Threshold Ablations
- One paragraph reading Tab 2: as $\tau$ increases, F1 and tokens both decrease monotonically; $\tau{=}0.3$ is the Pareto knee, with nearly identical F1 to $\tau{=}0.25$ and 142 fewer tokens.
- One paragraph on $\rho$: at fixed $\tau$, varying $\rho\in\{0.50, 0.55, 0.65\}$ changes F1 by at most 0.015 and mean tokens by at most 8. The Jaccard stage is therefore a secondary refinement, retained for redundancy control rather than for primary tuning.
- **Tab 2.** Threshold ablation (combined $\tau\times\rho$).

## 6. Failure Mode Analysis

The section opens by stating that the three learned strategies are evaluated specifically as comparators on distractor-heavy multi-hop QA, and that their underperformance is structural rather than incidental. Each subsection recalls the design hypothesis from §4 and identifies the property of the benchmark that violates it.

### 6.1 Adaptive-$k$ — similarity gaps land in the wrong cluster
- The maximum-gap rule assumes that the largest similarity gap separates relevant from irrelevant context. On HotpotQA distractor, BM25 selects distractors precisely because they are topically adjacent to gold evidence, so embedding-similarity scores cluster and the largest gap typically falls inside a distractor cluster.
- Quantitative anchors: mean $k^*=3.02$, mean total tokens 483.1, F1 0.627 — below fixed $k{=}3$ (0.651), $k{=}5$ (0.716), and $k{=}10$ (0.792).

### 6.2 UCB1-TUNED — title coverage is sparse across train and validation
- The title-arm formulation assumes that the training-time arm vocabulary covers the inference-time arm vocabulary. HotpotQA draws titles from a Wikipedia-scale distribution, so 83.8 % of validation queries at $k{=}5$ touch arms unseen during training (cold-start rates 61.5 / 75.2 / 83.8 % at $k\in\{2,3,5\}$), forcing BM25 fallback on most queries.
- Best operating point ($k{=}5$): F1 0.704, mean tokens 723.0 — below fixed $k{=}5$ at F1 0.716.
- Numbers stated in prose; the synthesis lands in Tab 3.

### 6.3 LinUCB — lexical features cannot separate crafted distractors
- The contextual-bandit formulation addresses cold-start by scoring documents through context features. The features tractable at scale (token overlap, title-in-question fraction, normalized section length) are matched-by-construction with BM25-retrieved distractors, so a linear model in those features cannot discriminate gold evidence from distractors.
- Per-$k$ F1: 0.510 / 0.522 / 0.620 at $k\in\{2,3,5\}$, mean tokens 344 / 455 / 686.
- Numbers stated in prose; the synthesis lands in Tab 3.

### 6.4 Why the noise gate succeeds
- The three failure mechanisms share a common cause: the signals on which the learned policies depend are not discriminative on this benchmark. Dense paragraph-level cosine similarity provides a signal that the other policies do not exploit, and a deterministic threshold avoids both arm-vocabulary cold-start and the misplaced-gap problem.
- One paragraph closing the loop on the cost side: at $(\tau{=}0.3, \rho{=}0.65)$, F1 is 0.764, tokens 1065.3, $-26.7\%$ versus $k{=}10$, while the closest learned operating point (UCB1-TUNED at $k{=}5$) reaches F1 0.704 at 723.0 tokens but with 83.8 % cold-start.
- **Tab 3.** Failure summary.

## 7. Discussion

The section must contain four named paragraphs:

- **Pareto interpretation.** State that the recommended $\tau{=}0.3$ corresponds to the Pareto knee in Tab 2 and Fig 2, with $\tau{=}0.25$ a quality-leaning alternative within the band.
- **Cost implications at scale.** Per-query cost on `gpt-5.4-mini` drops from $2.95\times 10^{-4}$ USD at $k{=}10$ to $2.17\times 10^{-4}$ USD at $\tau{=}0.3$. At one million queries per day, the annualized saving is approximately USD 28,000.
- **Limitations.** (i) Transfer evidence is from $n{=}100$ pilots on two further datasets, not full validation splits. (ii) A single generation model is fixed; stronger models may tolerate noisier context and shift the optimal threshold. (iii) The recommended $\tau$ is dataset-specific; calibration is required per domain. (iv) The bandit comparators are evaluated only on HotpotQA-style benchmarks; closed-vocabulary settings may admit different conclusions.
- **Future work.** A two-stage hybrid combining the gate with a lightweight cross-encoder; an adaptive $\tau$ chosen from the per-query similarity distribution; full-scale evaluation on 2WikiMultiHopQA, MuSiQue-Ans, and open-domain benchmarks (FEVER, Natural Questions); evaluation on stronger generation models.

## 8. Conclusion

The conclusion must contain three statements, in this order:

- The research question is answered: a training-free two-stage filter (cosine threshold $\tau$ followed by Jaccard redundancy threshold $\rho$) achieves F1 0.764 at 1065.3 tokens on HotpotQA distractor, within the $\pm 0.05$ tolerance band of $k{=}10$ and at $-26.7\%$ token cost.
- The structural finding: the three learned comparators (Adaptive-$k$, UCB1-TUNED, LinUCB) underperform because their signals — gap monotonicity, arm identity, surface-lexical features — are not discriminative on distractor-heavy multi-hop QA.
- The practical takeaway: for this class of benchmarks, deterministic threshold filtering with dense embeddings is the strongest test-time strategy observed, with thresholds calibratable on a small held-out sample and no training pipeline.

---

## Tables — Final Content

### Tab 1. Main Results and Transfer Evaluation
- **Caption.** Main results on HotpotQA distractor and transfer evaluation on 2WikiMultiHopQA and MuSiQue-Ans, all on `gpt-5.4-mini`. $\Delta$F1 and Savings are computed against fixed $k{=}10$ within the same dataset. Bold rows are within the $\pm 0.05$ F1 tolerance band. EM is reported in text.
- **Layout.** Three vertical blocks separated by `\midrule`; one row per strategy.

| Dataset | Strategy | F1 | Tokens | $\Delta$F1 | Savings |
|---|---|---:|---:|---:|---:|
| HotpotQA | No retrieval ($k{=}0$) | 0.387 | 103.5 | $-0.405$ | $-92.9\%$ |
| HotpotQA | Fixed $k{=}3$ | 0.651 | 471.3 | $-0.141$ | $-67.6\%$ |
| HotpotQA | Fixed $k{=}5$ | 0.716 | 731.7 | $-0.076$ | $-49.6\%$ |
| HotpotQA | Fixed $k{=}10$ (ref) | 0.792 | 1452.5 | --- | --- |
| HotpotQA | Adaptive-$k$ | 0.627 | 483.1 | $-0.165$ | $-66.7\%$ |
| HotpotQA | UCB1-TUNED ($k{=}5$) | 0.704 | 723.0 | $-0.088$ | $-50.2\%$ |
| HotpotQA | LinUCB ($k{=}5$) | 0.620 | 686.0 | $-0.172$ | $-52.8\%$ |
| HotpotQA | **Noise-Gate ($\tau{=}0.3$)** | **0.764** | **1065.3** | **$-0.028$** | **$-26.7\%$** |
| 2WikiMultiHopQA | Fixed $k{=}10$ (ref) | 0.617 | 1040.1 | --- | --- |
| 2WikiMultiHopQA | **Noise-Gate ($\tau{=}0.25$)** | **0.662** | **810.9** | **$+0.045$** | **$-22.1\%$** |
| MuSiQue-Ans | Fixed $k{=}10$ (ref) | 0.499 | 1342.9 | --- | --- |
| MuSiQue-Ans | **Noise-Gate ($\tau{=}0.3$)** | **0.551** | **1193.2** | **$+0.052$** | **$-11.2\%$** |

### Tab 2. Threshold Ablation (Combined $\tau \times \rho$)
- **Caption.** Noise-Gate threshold sweep on HotpotQA distractor ($n{=}400$, `gpt-5.4-mini`). F1 and total tokens reported at the recommended Jaccard threshold $\rho{=}0.65$. The "F1 range across $\rho$" column shows minimum and maximum F1 across $\rho\in\{0.50, 0.55, 0.65\}$, demonstrating that $\rho$ is a secondary control (variation of at most 0.015 F1 and 8 tokens at fixed $\tau$). The bold row is the recommended operating point.
- **Layout.** One row per $\tau$ value; five columns.

| $\tau$ | F1 | Tokens | Savings | F1 range across $\rho$ |
|---:|---:|---:|---:|---|
| 0.20 | 0.779 | 1317.5 | $-9.3\%$ | $0.771$–$0.779$ |
| 0.25 | 0.765 | 1207.3 | $-16.9\%$ | $0.765$–$0.778$ |
| **0.30** | **0.764** | **1065.3** | **$-26.7\%$** | $0.749$–$0.764$ |
| 0.35 | 0.726 | 889.8 | $-38.7\%$ | $0.726$–$0.731$ |
| 0.50 | 0.619 | 403.9 | $-72.2\%$ | $0.613$–$0.619$ |

### Tab 3. Failure Summary on HotpotQA Distractor
- **Caption.** Design hypothesis, structural property of HotpotQA distractor that violates it, and observed F1 for each learned comparator at its best operating point. The final row reports the proposed Noise-Gate at the same retrieval budget for direct comparison.
- **Layout.** Four rows; the final row separated by `\midrule`.

| Strategy | Design hypothesis | Structural mismatch on HotpotQA | F1 |
|---|---|---|---:|
| Adaptive-$k$ | Largest similarity gap separates evidence from noise | Topically adjacent BM25 distractors cluster in similarity; the largest gap falls inside a distractor cluster | $0.627$ ($k^*\!\approx\!3$) |
| UCB1-TUNED | Per-arm reward statistics rank arms accurately at light training cost | Wikipedia-scale title vocabulary; $83.8\%$ cold-start at $k{=}5$ forces BM25 fallback | $0.704$ ($k{=}5$) |
| LinUCB | Linear reward in lexical features generalizes beyond the training arm vocabulary | Lexical features matched-by-construction with BM25 distractors; linear model cannot discriminate | $0.620$ ($k{=}5$) |
| **Noise-Gate ($\tau{=}0.3$)** | Dense paragraph-level cosine similarity discriminates evidence from BM25-adjacent distractors | --- (signal preserved) | **$0.764$** |

---

## Figures — Final Content

### Fig 1. Gold Evidence vs BM25 Distractor
- **Placement.** End of §3 (Problem Formulation), bridging into §4.
- **Type.** Two side-by-side text panels with annotations, rendered as a single figure.
- **Source.** Slide deck `paper/Noise-Gated_Retrieval.pdf` (the gold-vs-distractor panel illustrated with the Corliss Archer / Shirley Temple example).
- **Content.**
  - *Left panel* — labeled "Gold evidence". One HotpotQA-distractor paragraph that contains the supporting fact for the example question. Question shown at the top.
  - *Right panel* — labeled "BM25 distractor". A paragraph from the same example pool that is topically adjacent (shares entities, vocabulary, and discourse style) but does not answer the question.
  - *Below each panel* — shared-entity highlights and, where computed, the cosine similarity to the question (e.g., *gold* $\cos = 0.42$, *distractor* $\cos = 0.39$). If exact cosines are not available, omit the numbers and keep the entity-overlap annotation only.
- **Caption (draft).** "An example HotpotQA distractor instance. Gold evidence and BM25-retrieved distractor share entities and vocabulary with the question, producing similar embedding scores. The structural similarity makes paragraph-level filtering non-trivial."

### Fig 2. Pareto Frontier with $\tau$-Sweep Overlay
- **Placement.** §5.2 (Main Results), immediately after Tab 1.
- **Type.** Single scatter+line plot.
- **Axes.** $x$: mean total tokens per query, linear, range $[0, 1500]$. $y$: token-level F1, range $[0.30, 0.85]$.
- **Reference band.** Horizontal shaded band at $F_1 \in [0.742, 0.842]$ (the $\pm 0.05$ tolerance around the $k{=}10$ reference).
- **Points to plot (HotpotQA, $n{=}400$).**
  - *Fixed baselines (square markers):* $(103.5, 0.387)$, $(471.3, 0.651)$, $(731.7, 0.716)$, $(1452.5, 0.792)$.
  - *Non-fixed comparator selectors (triangle markers):* Adaptive-$k$ $(483.1, 0.627)$; UCB1-TUNED $k{=}5$ $(723.0, 0.704)$; LinUCB $k{=}5$ $(686.0, 0.620)$.
  - *Noise-Gate $\tau$-sweep (circle markers, connected by line):* $(1317.5, 0.779)$, $(1207.3, 0.765)$, $(1065.3, 0.764)$, $(889.8, 0.726)$, $(403.9, 0.619)$, with $\tau$ value labeled at each marker.
- **Annotations.** Arrow labelled "knee, $\tau{=}0.3$" pointing to $(1065.3, 0.764)$. Reference label "$k{=}10$" near $(1452.5, 0.792)$.
- **Caption (draft).** "F1 versus mean total tokens on HotpotQA distractor ($n{=}400$, `gpt-5.4-mini`). The shaded band is the $\pm 0.05$ tolerance around the $k{=}10$ reference. The connected curve traces the Noise-Gate $\tau$-sweep at $\rho{=}0.65$. Noise-Gate at $\tau{=}0.3$ is the only point that combines a token reduction with quality inside the tolerance band."

---

## Algorithm 1 — Noise-Gate

- **Placement.** §4.5 (Noise-Gate).
- **Inputs.** Question $q$; paragraph pool $D=(d_1,\dots,d_n)$; thresholds $\tau$, $\rho$.
- **Output.** Selected subset $S$.
- **Body.** Compute $e_q$ and $\{e_i\}$; rank paragraphs by $\cos(e_q, e_i)$ in descending order; iterate; skip if $\cos<\tau$; skip if $\max_{d_j\in S}\mathrm{Jaccard}(d_i,d_j)>\rho$; otherwise append $d_i$ to $S$; return $S$.

---

## Equation Inventory (final, in order of appearance)

1. **Eq 1** — cost-aware objective $J(S)$ (§3).
2. **Eq 2** — Adaptive-$k$ truncation (§4.2).
3. **Eq 3** — UCB1-TUNED score (§4.3).
4. **Eq 4** — LinUCB feature vector and ridge update (§4.4).
5. **Eq 5** — LinUCB scoring rule (§4.4).
6. **Eq 6** — Noise-Gate cosine relevance test (§4.5).
7. **Eq 7** — Noise-Gate Jaccard redundancy test (§4.5).
