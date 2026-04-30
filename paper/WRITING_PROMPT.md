# Writing Prompt — Rewriting `paper.tex` to Match `PAPER_STRUCTURE.md`

This file is the prompt to give an LLM (or writing agent) when transforming the BRACIS submission from its current state in `paper.tex` to the structure declared in `PAPER_STRUCTURE.md`.

---

## Prompt to copy into the writing session

You are revising a research paper for BRACIS 2026 (Springer LNCS format). Two files define the task:

- `paper/paper.tex` — the **starting point**. It contains the current LaTeX source: title, abstract, eight sections (Introduction, Related Work, Problem Formulation, Methods, Experiments, Failure Mode Analysis, Discussion, Conclusion), figures, tables, an algorithm, and a bibliography. Treat its experimental numbers, citations, and `.bibitem` keys as authoritative. Do **not** invent new data.
- `paper/PAPER_STRUCTURE.md` — the **target specification**. It declares the style (§0), citation rules (§0.1), cross-section conventions (§0.2), the per-section content the paper must contain (§§1–8), the final tables (Tab 1, Tab 2, Tab 3), the final figures (Fig 1, Fig 2), Algorithm 1, and the equation inventory. Treat it as binding.

### Your task

Rewrite `paper.tex` so that it matches `PAPER_STRUCTURE.md` exactly. The output is a single updated `paper.tex` that compiles with the existing LNCS template (`llncs` documentclass, `splncs04` bibliography style).

### Rules

1. **Style.** Follow §0 of `PAPER_STRUCTURE.md` strictly: formal academic register, no marketing language, no first-person singular, no em-dashes for narrative emphasis, no exclamation marks, no rhetorical questions in the body. Use the diction lists.
2. **Length.** Hit the per-section word budgets in §0 (Abstract 200, Intro 700, Related Work 500, Problem Formulation 250, Methods 1100, Experiments 1100, Failure Mode Analysis 600, Discussion 250, Conclusion 100; total 4800). ±10 % per section is acceptable. Captions, table cells, and the bibliography do not count.
3. **Citations.** Follow §0.1. Every `\bibitem` in the final file must be cited at least once; cite each work at first mention with method-name-first phrasing where possible. Remove any uncited bibitems listed in §0.1's "must either be cited or removed" line. Do not invent new references.
4. **Reference validation.** For every citation kept in the final file, verify the reference against an open source (arXiv, ACL Anthology, OpenReview, Springer, the publisher's official page, or a permanent DOI) and confirm that (a) the work cited actually exists, (b) the cited claim, method, or result is in fact in that paper, and (c) the bibliography entry's title, authors, venue, and year match the canonical record. Correct any drift before submission. If a reference cannot be located in an open source, flag it for the author rather than leave it in.
5. **Reference selection.** Prefer recent peer-reviewed work (last 3–5 years) for empirical and methodological claims, and use canonical older references for established results — original algorithm papers (e.g., UCB1, LinUCB), foundational benchmarks (e.g., HotpotQA), classical techniques (e.g., MMR). Recency is a tiebreaker, not the primary criterion: a 2002 algorithm paper is preferred over a 2024 application paper when the claim is about the algorithm itself. Avoid arXiv-only preprints when a peer-reviewed version exists; use the published venue. Do not pad with marginal recent papers when a stronger older citation already supports the claim.
6. **Conventions.** Apply §0.2 once, in §5.1 (Setup), and do not re-state hyperparameters elsewhere. The generation model is `gpt-5.4-mini`, chosen as the cost-effective tier from a major commercial provider.
7. **Section content.** For each section §1–§8, the bullets in `PAPER_STRUCTURE.md` enumerate what the section must contain and in what order. Convert each bullet into one paragraph (or one named paragraph block where the spec says "named paragraphs"). Do not add bullets that are not in the spec; do not skip bullets that are.
   - **§1 Introduction has a fixed six-paragraph plan (P1–P6) declared by the advisor.** Maintain the paragraph order and the prescribed openings ("Yet, most retrieval systems retrieve a fixed number of documents…", "In particular, for multi-hop or distractor-heavy data…", "The remainder of this paper is organized as follows…"). Wording within each paragraph is free as long as it satisfies the bullets.
   - **§2 Related Work follows the advisor-preferred layout** (opening research-line paragraph + four-family body + closing sentence). The three-subsection variant is optional; do not introduce subsections shorter than half a column.
8. **Floats.** The paper has exactly **2 figures, 3 tables, 1 algorithm, 7 equations**. Remove any float in the current `paper.tex` that is not in the spec. Re-create Tab 1, Tab 2, Tab 3 from the tables given in the "Tables — Final Content" section of the spec. For Fig 1 and Fig 2, write the LaTeX figure environments with `\includegraphics`, `\caption`, and `\label` filled in from the "Figures — Final Content" section, but **do not generate the image files** — point each `\includegraphics` at a placeholder path under `paper/images/` (e.g., `images/fig1_gold_vs_distractor.pdf`, `images/fig2_pareto_tau_overlay.pdf`). The actual image assets will be produced and dropped in later. Keep Algorithm 1 in §4.5.
9. **Numbers.** Use the exact numbers from `PAPER_STRUCTURE.md`. Do not regenerate or recompute them. Apply the precision rule: F1 and EM to 3 decimals, mean tokens to 1 decimal, percentages to 1 decimal, costs in scientific notation.
10. **Comparator framing (3 doses).** Apply the "comparator-ladder" framing as specified: one orienting sentence in §1, "Why try it" lines in §§4.2–4.4, and the "Recall the appeal / Why it breaks" structure in §§6.1–6.3.
11. **Preserve the LaTeX scaffolding.** Keep the existing `\documentclass`, package list, title, author block, `\maketitle`, and bibliography mechanism. Only the body and float content change.

### Workflow

Work in this order so each step is verifiable:

1. Read `paper/paper.tex` end-to-end. Read `paper/PAPER_STRUCTURE.md` end-to-end.
2. Rewrite the abstract to ≈200 words in the formal register; keep the existing keywords line.
3. Rewrite §1 Introduction as **six paragraphs (P1–P6)** following the advisor's plan in the spec. P3 carries the comparator-ladder sentence; P5 lists the research questions and contributions; P6 is the paper-organization paragraph.
4. Rewrite §2 Related Work using the advisor-preferred layout: opening research-line paragraph (RAG / multi-hop distractors / cost-aware evaluation), then one paragraph per family in the order given by the spec, then closing sentence positioning Noise-Gate.
5. Rewrite §3 (Problem Formulation), placing Eq 1 and the Fig 1 reference at the end.
6. Rewrite §4 (Methods) with one "Why try it" / design-hypothesis line per subsection. Place Eqs 2–7 and Algorithm 1 as specified.
7. Rewrite §5 (Experiments). Replace the existing tables with Tab 1 and Tab 2 from the spec. Replace the existing figure with the Fig 2 environment from the spec (figure command + caption + label only, placeholder `\includegraphics` path; image asset added later). Move dataset-comparison content into three sentences in §5.1.
8. Rewrite §6 (Failure Mode Analysis) with the "Recall the appeal / Why it breaks" structure. Replace the prior three failure-related tables with Tab 3.
9. Rewrite §7 (Discussion) as four named paragraphs (Pareto, Cost, Limitations, Future work).
10. Rewrite §8 (Conclusion) as the three statements in the specified order.
11. Audit citations: for each `\bibitem` in the file, confirm it is cited; for each method, dataset, or claim listed in §0.1's coverage list, confirm the citation is present at first mention. Remove uncited bibitems. Then validate each remaining reference against an open source (arXiv, ACL Anthology, OpenReview, Springer, DOI) per Rules 4–5 — confirm the work exists, supports the cited claim, and that title/authors/venue/year are correct. Flag anything that cannot be located.
12. Compile mentally: confirm 2 figures, 3 tables, 1 algorithm, 7 numbered equations, and a body length close to 4800 words.

### Output format

Return the complete rewritten `paper.tex`. No diff, no commentary, no preamble. The file must be compilable as-is.

### What not to do

- Do not change the title, author block, or LNCS template setup.
- Do not introduce new datasets, models, methods, or numerical results.
- Do not add headers, subsections, floats, or equations beyond what `PAPER_STRUCTURE.md` declares.
- Do not change the float numbers or labels (`tab:main_results`, `fig:pareto`, etc.) without updating every `\ref` accordingly.
- Do not exceed the LNCS 12-page limit.
- Do not insert your own opinions, novelty claims, or marketing language.

---

## Files referenced

- `paper/paper.tex` — current source (starting point).
- `paper/PAPER_STRUCTURE.md` — target specification (destination).
- `paper/images/` — figure assets directory. Image files will be produced separately and dropped in later. The rewrite only writes the LaTeX `figure` environments with `\includegraphics{...}` placeholder paths, captions, and labels.

## When to re-run this prompt

- After updates to `PAPER_STRUCTURE.md` (re-derive the body so it matches).
- After new experimental numbers land (update the spec first, then re-run).
- After advisor feedback that affects scope (update the spec first, then re-run).
