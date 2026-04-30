# Paper Agent Brief

Use this brief when asking another agent to revise `paper.tex`. The goal is to keep edits consistent with the paper's purpose, tone, evidence standard, and current argumentative shape.

## Paper Purpose

This paper evaluates **Noise-Gate**, a training-free context-selection method for cost-aware RAG. Noise-Gate filters a fixed retrieved paragraph pool using:

1. cosine-similarity thresholding against the question, and
2. Jaccard-overlap redundancy removal.

The central question is whether a simple training-free filter can reduce token cost while preserving answer quality on distractor-heavy multi-hop QA benchmarks.

The paper is not a general RAG survey and should not become one. It is a controlled experimental comparison of context-compression strategies under one generation model, one prompt setup, and matched samples.

## How To Use This Brief

This brief is meant for agents working on **any section of the paper**, not only the Introduction. If the assigned task is about Related Work, Problem Formulation, Methods, Experiments, Failure Mode Analysis, Discussion, Conclusion, references, or tables, treat the Introduction as background context and do not rewrite it unless the task explicitly asks for that.

When an agent is assigned one section, it should keep edits local to that section and only touch other sections to fix broken references, duplicated claims, inconsistent terminology, or bibliography issues caused by the assigned edit.

## Core Claim

Noise-Gate preserves answer quality within the paper's tolerance band while reducing token cost on HotpotQA distractor, and transfers to two related multi-hop benchmarks with at most one threshold step of recalibration.

The paper also explains why the learned comparators underperform on this benchmark class:

- Adaptive-$k$ depends on embedding-rank monotonicity, which plausible BM25 distractors violate.
- UCB1-TUNED depends on reusable title-arm statistics, but the title vocabulary is sparse across train and validation.
- LinUCB depends on shallow lexical features, which are matched by BM25-retrieved distractors.

## Audience

The expected reader is a BRACIS / Springer LNCS reviewer who understands RAG and machine learning but may not know the details of multi-hop QA benchmarks. Terms such as **multi-hop**, **distractor**, **fixed-$k$**, **quality-preserving**, and **candidate pool** should be defined or made clear on first use.

Current definition of multi-hop: answering requires combining or linking evidence across multiple paragraphs, rather than extracting the answer from a single passage.

## Tone And Style

Use formal academic prose in Springer LNCS style.

Prefer:

- "evaluate", "propose", "characterise", "observe", "find", "indicate", "consistent with", "reduce", "preserve", "transfer"
- precise claims tied to tables, figures, or citations
- present tense for definitions and standing claims
- past tense for experiments already run

Avoid:

- marketing language such as "powerful", "groundbreaking", "remarkable", "dramatic", "magic"
- overclaiming novelty
- first-person singular
- rhetorical questions
- exclamation marks
- em dashes for emphasis
- unsupported field-wide claims

The writing should feel controlled, careful, and reviewer-friendly. Do not make the method sound like a product.

## Argument Structure

The current Introduction establishes the paper's argumentative frame. Agents working on later sections should preserve this frame:

1. RAG grounds generation in retrieved evidence, making context selection important.
2. Fixed-$k$ retrieval remains common because it is operationally simple, but it inflates token cost.
3. Existing alternatives include training-based adaptive retrieval, multi-agent or iterative filtering, compression methods, and similarity-gap truncation.
4. Multi-hop and distractor-heavy benchmarks make selection harder because evidence is complementary, distractors are plausible, and the candidate pool is fixed.
5. Noise-Gate is introduced as a training-free filter; research questions and contributions follow.
6. Paper organization.

Keep the comparator-ladder framing:

- Adaptive-$k$: similarity-gap structure
- UCB1-TUNED: per-title reward statistics
- LinUCB: contextual lexical features
- Noise-Gate: dense paragraph-level semantic similarity plus redundancy removal

## Citation Standard

Only cite prior work for claims it actually supports.

Important examples:

- Cite `goldstein1998mmr` only for the redundancy/diversity lineage of the Jaccard stage, not as if it introduced Noise-Gate.
- Cite `chiang2024plausible` for plausible distractors in multi-hop reasoning.
- Cite `trivedi2023interleave` for multi-step or multi-hop evidence complementarity and retrieval/reasoning interaction.
- Cite benchmark papers at first dataset mention:
  - HotpotQA: `yang2018hotpotqa`
  - 2WikiMultiHopQA: `ho-etal-2020-constructing`
  - MuSiQue: `trivedi-etal-2022-musique`
- Cite algorithmic comparators at first method mention:
  - UCB1: `auer2002ucb1`
  - LinUCB: `chu2011linucb`
  - Adaptive-$k$: `taguchi2025adaptivek`

Do not cite the paper's own results. Refer to the relevant section, table, or figure instead.

## Current Bibliography State

The duplicate `\bibitem` keys previously present near the end of the bibliography (`goldstein1998mmr`, `chang2024mainrag`, `jeong2025ecorag`, `nahid2025prism`, `chiang2024plausible`) have been removed; every key is now defined exactly once.

A reservoir of currently-uncited entries is intentionally retained in `paper.tex` because they may be cited in later revisions:

- `jiang2016adaptive`, `mao2021gar`, `wang2023query2doc`, `gao2023hyde`, `nogueira2019bert`, `sun2023rankgpt`, `qin2024prp`, `yao2023react`, `yi2026membership`, `zhang2024retrievalqa`.

An agent revising references should preserve these reservoir entries unless the author explicitly asks for cleanup. Before submission, every remaining `\bibitem` must be either cited at first relevant mention or removed.

## Fixed Experimental Facts

Do not invent or recompute numbers. Use the existing reported values unless the author explicitly supplies updated results.

HotpotQA distractor:

- No retrieval: F1 0.387, 103.5 tokens
- Fixed $k{=}3$: F1 0.651, 471.3 tokens
- Fixed $k{=}5$: F1 0.716, 731.7 tokens
- Fixed $k{=}10$: F1 0.792, 1452.5 tokens
- Adaptive-$k$: F1 0.627, 483.1 tokens
- UCB1-TUNED at $k{=}5$: F1 0.704, 723.0 tokens
- LinUCB at $k{=}5$: F1 0.620, 686.0 tokens
- Noise-Gate at $\tau{=}0.30$: F1 0.764, 1065.3 tokens, 26.7% token reduction

Transfer pilots:

- 2WikiMultiHopQA fixed $k{=}10$: F1 0.617, 1040.1 tokens
- 2WikiMultiHopQA Noise-Gate at $\tau{=}0.25$: F1 0.662, 810.9 tokens
- MuSiQue-Ans fixed $k{=}10$: F1 0.499, 1342.9 tokens
- MuSiQue-Ans Noise-Gate at $\tau{=}0.30$: F1 0.551, 1193.2 tokens

Quality-preserving means within $\pm 0.05$ F1 of fixed $k{=}10$ on the same dataset.

## Current Edits Already Made

Recent wording fixes in `paper.tex`:

- The prior-work sentence in the introduction was narrowed so prior work supports benchmark-structure sensitivity, while Section 6 quantifies selector impact.
- "Multi-hop" is now defined in the abstract and introduction.
- The MMR citation was moved so it supports only the redundancy-stage lineage, not the invention of Noise-Gate.

Preserve these improvements unless replacing them with stronger wording that keeps the same claim boundaries.

## Section-Specific Guidance

Use the Introduction only as the promise that later sections must fulfill.

- **Related Work:** Do not repeat the full Introduction. Position prior methods by family and explain how Noise-Gate differs in deployment cost, supervision, and selector signal.
- **Problem Formulation:** Keep definitions compact. Introduce symbols before use and connect the objective to the Pareto evaluation.
- **Methods:** Explain each strategy through its design hypothesis, then formalize it. Do not add new comparators or new hyperparameters.
- **Experiments:** Report matched-sample results and transfer pilots. Do not reinterpret numbers beyond what the tables support.
- **Failure Mode Analysis:** Tie each comparator's failure to one structural property of the benchmark. Keep "Recall the appeal" and "Why it breaks" if that structure is already present.
- **Discussion:** Keep claims practical and bounded: Pareto interpretation, cost implications, limitations, future work.
- **Conclusion:** Restate the result, the reason learned comparators underperform, and the deployment implication without adding new claims.
- **Bibliography:** Remove duplicate keys and uncited entries. Do not add references unless they support a specific sentence in the body.

## Editing Priorities

When revising, prefer high-signal fixes:

1. Clarify unsupported or overbroad claims.
2. Define terms before relying on them.
3. Keep citations attached to the exact claim they support.
4. Preserve all experimental numbers and labels.
5. Remove duplicate or uncited bibliography entries.
6. Keep the page budget tight.

Do not perform broad rewrites unless explicitly requested. This paper needs careful tightening more than stylistic reinvention.
