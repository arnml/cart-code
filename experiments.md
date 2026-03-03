# Experiments

## Option A — SMTL-like behavior with zero training

1) Parallel evidence acquisition — parallel repo “sensors”

Instead of parallel web searches, run parallel codebase probes:

- Symbol search using `ripgrep` for the error string, function name, or feature flag
- Call-graph probes: “who calls this?” / “what calls this?”
- Test probes: run targeted unit tests or a reproduce script
- Blame/history probes: `git blame` and recent commits touching files
- Config probes: environment variables, feature toggles, build scripts
- Observability probes: logs, stack traces, metrics (if available)

These can run concurrently and return short evidence snippets.

2) Structured context management — evidence ledger + bounded windows

Maintain three artifacts (JSON or plain text):

- **Hypotheses (ranked):** possible root causes with confidence
- **Evidence (tagged):** each snippet includes source, file, line range, and why it's relevant
- **Patch plan:** minimal change list and tests to validate the fix

Enforce rules:

- Keep only the top-K evidence per hypothesis
- Summarize older evidence into 3–5 concise facts
- Always cite file:line ranges in the working notes
- Avoid storing entire files in context — use slices and summaries

This provides the structured-context benefits of SMTL without additional training.

3) Plan refinement loop (SMTL-style)

Run short cycles:

1. Generate hypotheses
2. Spawn probes (parallel)
3. Update hypothesis scores
4. Pick 1–2 fixes to try
5. Validate with tests
6. Iterate

Cost: inference + local compute.

## Option B — Strong model + tool orchestration (no training)

Invest in tooling around a strong coding model:

- A fast repo index (`ripgrep` + `ctags` / `tree-sitter`)
- A lightweight retrieval layer that returns only relevant spans
- An execution sandbox (run tests, lint, typecheck)
- A patch applicator (`git diff`) and verifier

This approach spends engineering effort rather than GPU training and scales well.

## Option C — Cheap learning: “distill your own traces” (SFT-lite)

If you want the agent to internalize your workflow, do small SFT:

- Collect 200–2,000 real bugfix episodes from your repos: issue text, stack traces, commands run, files opened, diffs, and the final fix
- Convert episodes into trajectories: `probe plan → evidence → hypothesis update → patch → tests → result`
- Fine-tune a 7B–14B model (LoRA/QLoRA) with context windows of 8k–32k

This is much cheaper than training huge models and is worthwhile when you repeatedly fix similar bug categories in similar stacks.

## Option D — RL, but only where it matters (micro-RL)

Full RL is costly due to long rollouts. For codebases, use RL on short horizons:

- Reward = tests passing, bug reproducer fixed, or no new lint/type errors
- Keep trajectories short: cap file opens, tool calls, and tokens
- Train only the policy for choosing probes/files (not for full patch writing)

Use RL for:

- Triage (what to inspect next)
- Localization (where the bug is)
- Verification strategy (which tests to run)

Let the base model handle patch generation.

## Option E — Don’t train a model; train a ranker

A cheap, effective alternative is to train a lightweight ranker:

- Use embeddings or a classifier to rank which files to open, which symbols are related, and which commits are suspicious
- Feed the top-ranked results to an LLM to synthesize a fix

This mimics “search more” efficiently and is low-cost to operate.
