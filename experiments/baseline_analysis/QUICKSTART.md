# ⚡ Quick Start: Run Baselines in 2 Steps

## Step 1: Verify Your Setup

This project uses `uv` as the package manager. Make sure your `OPENAI_API_KEY` is set:

```powershell
$env:OPENAI_API_KEY
```

You should see: `sk-proj-vN...` (your API key)

## Step 2: Run Validation (Optional but Recommended)

```powershell
cd experiments/baseline_analysis
py validate_setup.py
```

This checks that:
- ✓ All Python packages are installed
- ✓ OPENAI_API_KEY is in your environment variables
- ✓ HotpotQA dataset can be accessed

If all checks pass ✅, continue to step 3.

## Step 3: Run the Baselines

```powershell
py run_baselines.py
```

> **Note:** The script automatically reads your `OPENAI_API_KEY` from Windows environment variables. No manual setting needed.

**Expected runtime:** 5-10 minutes (with rate limiting)

---

## What Happens

The script will:

1. Load 50 random HotpotQA questions
2. Run 3 methods on each:
   - **Always-Think**: Pure CoT, no retrieval
   - **Always-Retrieve k=3**: Top 3 documents
   - **Always-Retrieve k=5**: Top 5 documents
3. Measure F1 score, tokens used, and estimated cost
4. Save results to `results/results.csv`
5. Create summary report in `results/summary.md`

---

## Outputs

### `results/results.csv`
150 rows (50 questions × 3 methods) with:
- F1 score for answer correctness
- Token counts (input + output)
- Estimated cost in USD
- Efficiency metric (F1 per token)

### `results/summary.md`
Human-readable summary table showing average performance by method.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'openai'`
→ Run: `uv sync`

### `openai.error.AuthenticationError`
→ Check your API key: `echo $OPENAI_API_KEY` (macOS/Linux)
→ Make sure it starts with `sk-`

### `KeyError: model 'gpt-4o-mini' not found`
→ Make sure OpenAI account has access to GPT-4o-mini (check https://platform.openai.com/account/billing/overview)

### Script runs very slowly
→ This is normal! Each API call takes ~2-3 seconds. 50 questions × 3 methods ≈ 150 calls ≈ 5-10 minutes total.

---

## Next Steps

Once you have baseline results:

1. **Review the summary:** `results/summary.md`
2. **Compare F1 scores:** Which method got the highest F1?
3. **Analyze efficiency:** Which method uses fewer tokens?
4. **Day 3:** Implement CART v1 and test it against these baselines

---

## File Structure

```
baseline_analysis/
├── QUICKSTART.md              ← You are here
├── README.md                  ← Full documentation
├── run_baselines.py           ← Main script
├── main.py                    ← Alternative launcher
├── validate_setup.py          ← Pre-flight checks
│
├── baseline_always_think.py   ← Pure CoT baseline
├── baseline_always_retrieve.py ← RAG baseline
├── dataset_prep.py            ← Load HotpotQA
├── eval_utils.py              ← F1, cost, metrics
│
└── results/                   ← Output folder
    ├── results.csv            ← Detailed results
    └── summary.md             ← Summary report
```

---

**Ready?** Run: `py run_baselines.py` 🚀
