# CART: Cost-Penalized Adaptive Routing for Test-Time Retrieval

## Problem
Current RAG agents apply a fixed retrieval budget (top-k documents) to every query. The problem is that this is often wasteful and sometimes unnecessary.

Why?

- Some questions really need external evidence.
- Some can already be answered by the model’s own parametric knowledge.
- Retrieving too much adds tokens, latency, and noise.
- The best retrieval behavior can change across queries and across models.

So the real problem is:

> How should an LLM decide, at inference time, whether to retrieve at all, and if so, how much to retrieve, while accounting for token cost?

## What does the paper want to do?
The paper wants to turn retrieval from a fixed pipeline choice into an adaptive test-time decision.

More specifically, it wants to propose a method that:

- is training-free
- works at inference time
- decides whether to retrieve
- decides how much context to retrieve
- balances answer quality against token cost

That method is CART.

So the paper’s central move is:

> Treat retrieval as a cost-aware routing problem, not as a fixed top-𝑘 k preprocessing step.

## The Contribution
CART is a **training-free test-time controller** with three components:
1. **Adaptive-K selection**: find k* via largest similarity score gap (Taguchi et al. EMNLP 2025)
2. **UCB-Cost policy**: bandit action selector extended with explicit cost penalty (novel term)
3. **Noise gate**: filter low-similarity and redundant documents before generation

CART **automatically discovers the model-appropriate strategy** without
configuration, adapting routing as model capability changes.

## Run project
### Create and activate venv
py -m venv .venv
.\.venv\Scripts\Activate.ps1

### Upgrade pip, install uv
```bash
py -m pip install --upgrade pip
py -m pip install uv
```

### Install packages (runtime + tools)
```bash
uv sync
```

### Download dataset
```bash
uv run python -m experiments.download_dataset
```

### Baselines
```bash
uv run python -m experiments.run_baseline <MODEL_NAME> <SAMPLE_SIZE> [max_workers]
```

### Analise baselines
```bash
uv run python -m experiments.analyse_baseline <MODEL_NAME>
```

### Plot baseline F1 vs k
```bash
uv run python -m experiments.plot_baseline_f1_vs_k
```

### Plot baseline EM split
```bash
uv run python -m experiments.plot_baseline_em_percentage
```

### Run adaptive_k
```bash
uv run python -m experiments.run_adaptive_k <MODEL_NAME> <SAMPLE_SIZE> [max_workers]
```

### Analyse adaptive_k
```bash
uv run python -m experiments.analyse_adaptive_k <MODEL_NAME> <SAMPLE_SIZE>
```
