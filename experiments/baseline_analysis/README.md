# Day 2: Baseline Analysis

This directory contains code to evaluate two baseline methods on HotpotQA for your CART paper.

## What It Does

1. **Always-Think (Baseline 2)**: Chain-of-thought reasoning, no retrieval
2. **Always-Retrieve k=3**: Retrieve top-3 documents, then answer
3. **Always-Retrieve k=5**: Retrieve top-5 documents, then answer

Each method is evaluated on 50 random HotpotQA questions (validation set, distractor setting).

**Outputs:**
- `results/results.csv` — Detailed results for each question and method
- `results/summary.md` — Summary statistics and interpretation

## Files

- `run_baselines.py` — Main entry point (execute this)
- `baseline_always_think.py` — Pure reasoning baseline
- `baseline_always_retrieve.py` — Retrieval-augmented baseline
- `dataset_prep.py` — Load and prepare HotpotQA data
- `eval_utils.py` — F1 score, cost, efficiency metrics

## Setup

### 1. Activate venv

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows cmd
.venv\Scripts\activate.bat

# macOS/Linux
source .venv/bin/activate
```

### 2. Verify OpenAI API key is set

Your `OPENAI_API_KEY` should already be in your Windows environment variables.

To verify it's set:
```powershell
$env:OPENAI_API_KEY
```

If not set, set it once (Windows PowerShell):
```powershell
$env:OPENAI_API_KEY = "sk-your-actual-key-here"
```

Get your key from: https://platform.openai.com/account/api-keys

### 3. Check dependencies

All required packages should already be in `pyproject.toml`:
- `openai` — API calls
- `datasets` — HotpotQA loading
- `tiktoken` — Token counting
- `numpy`, `scikit-learn` — Embeddings and similarity

If missing, install with:
```powershell
uv pip install scikit-learn
```

Or if not using `uv`:
```powershell
pip install scikit-learn
```

## Run

```powershell
cd experiments/baseline_analysis
py run_baselines.py
```

**Expected runtime:** ~5-10 minutes (50 questions × 3 methods, with rate limiting)

## Outputs

### `results/results.csv`

One row per (question, method) combination. Columns:
- `question_id` — Sample index
- `question` — The HotpotQA question
- `ground_truth` — Correct answer
- `method` — Baseline method name
- `answer` — Model's response
- `f1_score` — Answer correctness (0-1)
- `exact_match` — Perfect match (0 or 1)
- `input_tokens` — Tokens in prompt
- `output_tokens` — Tokens in response
- `total_tokens` — Total (input + output)
- `cost_usd` — Estimated cost in dollars
- `efficiency` — F1 / log(1 + tokens) — quality per unit cost
- `llm_calls` — Number of API calls
- `docs_retrieved` — (retrieve methods only) number of documents
- `avg_similarity` — (retrieve methods only) average similarity score

### `results/summary.md`

Human-readable summary table with averages by method:
- Average F1 and exact match
- Average token usage
- Average cost
- Efficiency scores

Example output:
```
| Method                | F1 Score | Exact Match | Avg Tokens | Cost | Efficiency |
|---|---|---|---|---|---|
| always_think          | 0.5234   | 0.1200     | 1800       | $0.32 | 0.2107 |
| always_retrieve_k3    | 0.6145   | 0.2400     | 2400       | $0.41 | 0.2315 |
| always_retrieve_k5    | 0.6823   | 0.3200     | 3100       | $0.52 | 0.2401 |
```

## Interpreting Results

- **F1 Score**: How much of the correct answer your method got right (0-1, higher is better)
- **Tokens**: How many tokens you used (fewer = cheaper)
- **Cost**: Total $ spent on API calls
- **Efficiency**: Quality per unit cost — your CART method should beat this

## Next Steps (Day 3)

Once you have baseline results, you'll implement CART v1 (adaptive context selection) and compare its efficiency against these baselines.
