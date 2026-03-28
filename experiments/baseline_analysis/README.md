# Day 2: Baseline Analysis

This directory evaluates three baseline methods on HotpotQA for the CART paper.

## Quick Start

### 1. Set API Keys

For **OpenAI models** (GPT-4o-mini, GPT-5.4-mini):
```powershell
$env:OPENAI_API_KEY = "sk-proj-..."
```

For **Claude models** (Haiku 4.5):
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

### 2. Run Baselines

**OpenAI models:**
```powershell
cd experiments/baseline_analysis
uv run python run_baselines.py gpt-4o-mini
uv run python run_baselines.py gpt-5.4-mini-2026-03-17
```

**Anthropic Claude:**
```powershell
uv run python run_baselines_anthropic.py
```

**Expected runtime:** ~5-10 minutes per model (50 questions × 3 methods)

---

## What It Does

Evaluates three baseline methods on 50 random HotpotQA questions (validation set, distractor setting):

1. **Always-Think**: Chain-of-thought reasoning, no retrieval
2. **Always-Retrieve k=3**: Retrieve top-3 documents, then answer
3. **Always-Retrieve k=5**: Retrieve top-5 documents, then answer

**Output:**
- `results_[MODEL]/results.csv` — Detailed results for each question/method
- `results_[MODEL]/summary.md` — Summary statistics and interpretation

---

## Files

### Core Scripts
- `run_baselines.py` — Run OpenAI models (GPT-4o-mini, GPT-5.4-mini)
- `run_baselines_anthropic.py` — Run Claude models (Haiku 4.5)
- `validate_setup.py` — Pre-flight checks

### Baseline Methods
- `baseline_always_think.py` — Pure reasoning (OpenAI)
- `baseline_always_retrieve.py` — RAG baseline (OpenAI)
- `baseline_always_think_anthropic.py` — Pure reasoning (Anthropic)
- `baseline_always_retrieve_anthropic.py` — RAG baseline (Anthropic)

### Utilities
- `dataset_prep.py` — Load and prepare HotpotQA data
- `eval_utils.py` — F1 score, cost, efficiency metrics

---

## Results Structure

Each model run creates `results_[MODEL]/` with:

### `results.csv`

One row per (question, method) combination:
- `question_id`, `question`, `ground_truth`
- `method`, `answer`
- `f1_score` — Answer correctness (0-1)
- `exact_match` — Perfect match (0 or 1)
- `input_tokens`, `output_tokens`, `total_tokens`
- `cost_usd` — Estimated cost
- `efficiency` — F1 / log(1 + tokens) = quality per unit cost
- `docs_retrieved`, `avg_similarity` (retrieve methods only)

### `summary.md`

Human-readable summary table with averages by method:

| Method | Count | F1 Score | Exact Match | Avg Tokens | Avg Cost | Efficiency |
|--------|-------|----------|-------------|-----------|----------|------------|
| always_think | 50 | 0.5798 | 0.4286 | 142 | $0.00035 | 0.1174 |
| always_retrieve_k3 | 50 | 0.6755 | 0.5200 | 483 | $0.00063 | 0.1095 |
| always_retrieve_k5 | 50 | 0.7526 | 0.5800 | 741 | $0.00081 | 0.1144 |

---

## Metrics Explained

- **F1 Score** (0-1): How much of the correct answer your method got right
- **Exact Match** (0 or 1): Perfect word-for-word match with ground truth
- **Tokens**: Total input + output tokens (proxy for cost)
- **Cost**: Estimated USD cost for all 50 questions
- **Efficiency**: `F1 / log(1 + tokens)` — quality per unit cost
  - Higher is better
  - This is what **CART should beat** in Day 3

---

## Model Comparison

### Pricing (per 1M tokens)

| Model | Input | Output |
|-------|-------|--------|
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-5.4-mini | $0.75 | $4.50 |
| claude-haiku-4-5 | $1.00 | $5.00 |

### Example Results

**GPT-5.4-mini on 50 HotpotQA samples:**
- Best F1: always_retrieve_k5 (0.7526)
- Best efficiency: always_think (0.1174)
- Insights: Retrieval helps accuracy but costs more tokens

---

## Setup Validation

Run the pre-flight check:
```powershell
uv run python validate_setup.py
```

Checks:
- ✓ All Python packages installed (openai, anthropic, datasets, etc.)
- ✓ API keys in environment variables
- ✓ HotpotQA dataset accessible

---

## Day 3: CART Implementation

Once you have these baseline results, you'll implement CART v1 (adaptive context selection) and compare:

```
CART efficiency > always_retrieve_k5 efficiency (current best)
CART F1 ≥ always_retrieve_k5 F1
CART tokens < always_retrieve_k5 tokens
```

The baselines define your improvement targets! 🎯
