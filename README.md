# Cost-Aware Test-Time Retrieval Control for RAG

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
A **training-free test-time controller** that **automatically discovers the model-appropriate strategy** without configuration, adapting routing as model capability changes.

## Project context
We evaluate on HotpotQA using the distractor subset and the validation split. 
- The baseline analysis shows that the bigger the k the better the F1
- We want to save token, so we are seeking a strategy with similar F1 as top-k (k=10) but with less tokens.
- The adaptive-k would sound a good idea to save tokens but we got poor F1 results
- The noise gate got similar F1 with 26% less tokens
- The UCB1-TUNED reranker learns title utility from training data and selects top-k titles at inference time

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

### 2Wiki baseline
```bash
uv run python -m experiments.2wiki.run_baseline <SAMPLE_SIZE> [max_workers]
```

### 2Wiki analysis
```bash
uv run python -m experiments.2wiki.analyse_baseline <MODEL_NAME>
```

### MuSiQue-Ans baseline
```bash
uv run python -m experiments.musique_ans.run_baseline <SAMPLE_SIZE> [max_workers]
```

### MuSiQue-Ans analysis
```bash
uv run python -m experiments.musique_ans.analyse_baseline <MODEL_NAME>
```

Both scripts use the cached validation split by default and write their CSVs
under their own dataset folders:
- `experiments/2wiki/results/baseline/`
- `experiments/musique_ans/results/baseline/`

### Analyse baselines
```bash
uv run python -m experiments.analyse_baseline <MODEL_NAME>
```

### Plot baseline F1 by method
```bash
uv run python -m experiments.plot_baseline_f1_vs_k
```

### Plot baseline methods vs adaptive-k F1 by method
```bash
uv run python -m experiments.plot_baseline_vs_adaptive_k_f1_vs_k gpt-5.4-mini
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

### Run noise_gate
```bash
uv run python -m experiments.run_noise_gate <MODEL_NAME> <SAMPLE_SIZE> [max_workers]
```

This runs the noise-gate ablation with similarity thresholds
`0.2`, `0.25`, `0.3`, `0.35`, and `0.5`, and writes
`experiments/results/cart/results_noise_gate_<MODEL_NAME>.csv` with the
`threshold` column alongside the per-sample metrics.

### Analyse noise_gate
```bash
uv run python -m experiments.analyse_noise_gate <MODEL_NAME>
```

This reads `experiments/results/cart/results_noise_gate_<MODEL_NAME>.csv`
and writes `experiments/results/cart/analysis_noise_gate_<MODEL_NAME>.md`
with one metric table whose columns are the thresholds.

### Run noise_gate ablation
```bash
uv run python -m experiments.run_noise_gate_ablation <MODEL_NAME> <SAMPLE_SIZE> [max_workers]
```

### Analyse noise_gate ablation
```bash
uv run python -m experiments.analyse_ablation_noise_gate <MODEL_NAME>
```

This reads both `experiments/results/cart/results_noise_gate_<MODEL_NAME>.csv`
and `experiments/results/cart/results_ablation_noise_gate_<MODEL_NAME>.csv`
and writes `experiments/results/cart/analysis_ablation_noise_gate_<MODEL_NAME>.md`
with two tables:

- F1 by Jaccard threshold vs similarity threshold
- Total tokens (mean) by Jaccard threshold vs similarity threshold

### Train UCB1-TUNED reranker
```bash
uv run python -m experiments.train_ucb1_tuned
```

### Run UCB1-TUNED reranker
```bash
uv run python -m experiments.run_ucb1_tuned <MODEL_NAME> <SAMPLE_SIZE> [max_workers]
```

Tests k ∈ {2, 3, 5}, writes `experiments/results/ucb1/ucb1_<MODEL_NAME>.csv`.

### Analyse UCB1-TUNED reranker
```bash
uv run python -m experiments.analyse_ucb1 <MODEL_NAME>
```

Analyzes results by k value, cold-start statistics, and overall metrics.

