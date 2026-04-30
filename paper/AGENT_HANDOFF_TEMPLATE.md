# Agent Handoff Template

Use this file as a reusable template when giving another agent context for a task. Replace the bracketed text before sending it.

## Task

[State the concrete task in one or two sentences.]

Example:

Revise the introduction of `paper.tex` to improve clarity and citation support without changing experimental claims.

## Goal

[Explain what success looks like.]

Example:

The result should read as a careful BRACIS / LNCS research-paper introduction, define key terms before relying on them, and avoid overclaiming what prior work supports.

## Project Context

[Summarize the project in plain terms.]

Example:

This repository contains a research paper about Noise-Gate, a training-free context-selection method for cost-aware RAG. The paper compares Noise-Gate against fixed-$k$ retrieval, Adaptive-$k$, UCB1-TUNED, and LinUCB on distractor-heavy multi-hop QA benchmarks.

## Files To Read First

- `[primary file]`
- `[spec or context file]`
- `[optional supporting file]`

Example:

- `paper.tex`
- `PAPER_STRUCTURE.md`
- `PAPER_AGENT_BRIEF.md`

## Scope

The agent should change:

- [List files, sections, or modules that are in scope.]

The agent should not change:

- [List files, sections, numbers, APIs, formatting, or claims that must stay fixed.]

Example:

Change only the Introduction and Related Work sections. Do not change tables, experimental numbers, figure labels, bibliography keys, or the author block.

## Tone And Style

[Describe the expected voice.]

Example:

Use formal academic prose. Be precise and restrained. Avoid marketing language, novelty inflation, rhetorical questions, em dashes for emphasis, and unsupported field-wide claims. Prefer concrete claims tied to citations, tables, or sections.

## Important Facts

[List facts that must not be invented, changed, or recomputed.]

Example:

- Noise-Gate at $\tau{=}0.30$ reaches F1 0.764 at 1065.3 mean tokens on HotpotQA distractor.
- Fixed $k{=}10$ reaches F1 0.792 at 1452.5 mean tokens.
- Quality-preserving means within $\pm 0.05$ F1 of fixed $k{=}10$ on the same dataset.

## Current Decisions To Preserve

[List recent edits or decisions the next agent should not undo.]

Example:

- Multi-hop is defined as requiring evidence across multiple paragraphs.
- The MMR citation supports only the redundancy-stage lineage, not the introduction of Noise-Gate as a whole.
- Prior work supports benchmark-structure sensitivity; Section 6 quantifies selector-specific impact.

## Evidence Standard

[Explain when citations or tests are required.]

Example:

Every claim about prior work must be directly supported by a citation. Do not cite the paper's own results; refer to the relevant table, figure, or section. Do not add references unless they can be verified from a reliable source.

## Output Format

[Say exactly what the agent should return.]

Examples:

- Return a patch only.
- Edit the files directly and summarize the changes.
- Return revised prose only, not the full file.
- Return findings first, then suggested edits.

## Known Risks

[List likely problems the agent should watch for.]

Example:

- Duplicate `\bibitem` keys may cause LaTeX warnings.
- Some citations may support related ideas but not the exact sentence they are attached to.
- The paper has a tight LNCS page budget, so avoid expanding prose unless necessary.

## Done Criteria

The task is complete when:

- [Criterion 1]
- [Criterion 2]
- [Criterion 3]

Example:

- The edited section preserves all experimental numbers.
- All added or retained citations support the exact claims they are attached to.
- The prose matches the requested tone and does not introduce unsupported novelty claims.

