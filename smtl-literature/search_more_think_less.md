# Search More, Think Less

Paper: *Search More, Think Less: Rethinking Long-Horizon Agentic Search for Efficiency and Generalization*  
Authors: Qianben Chen et al.  
Link: https://arxiv.org/abs/2602.22675

## Purpose

Working notes for short, question-driven summaries of the paper and closely related context.

## Current summary

SMTL argues that long-horizon research agents can improve efficiency and generalization by shifting effort away from long sequential reasoning and toward parallel evidence acquisition, tighter context management, and stronger search orchestration.

## Key ideas

- Replace long chains of reasoning with parallel evidence gathering.
- Manage evidence under bounded context budgets instead of carrying large raw histories.
- Train across both deterministic QA tasks and open-ended research tasks.
- Use supervised fine-tuning and reinforcement learning for end-to-end agent training.
- Target lower inference cost and latency while preserving or improving benchmark performance.

## Notes from questions

- `OPPO`: In this paper, `OPPO AI Agent Team` is the author affiliation, not a method component. It refers to the OPPO research team/company behind the work; the paper also lists the corresponding contact as `zhouwangchunshu@oppo.com`.

## Supporting papers for the "more depth / more tool calls help" claim

- [[2503.09516|Search-R1]]: Trains models to issue multiple search queries during step-by-step reasoning and reports strong gains over RAG baselines.
- [[2501.05366|Search-o1]]: Inserts agentic retrieval into long reasoning chains so the model can fetch external knowledge when its internal knowledge is insufficient.
- [[2504.21776|WebThinker]]: Interleaves thinking, web search, page navigation, and drafting to improve deep research performance.
- [[2504.03160|DeepResearcher]]: Shows that end-to-end RL in real web environments improves deep research agents over prompt-engineered and RAG-based baselines.
- [[2506.12928|Scaling Test-time Compute for LLM Agents]]: Gives direct empirical evidence that more test-time compute and more diversified rollouts improve agent performance.

## Contrast papers

- [[2603.09906|Thinking to Recall]]: Argues that extra reasoning can unlock latent parametric knowledge even without external search, making it a useful counterpoint to SMTL's "search more, think less" framing.

