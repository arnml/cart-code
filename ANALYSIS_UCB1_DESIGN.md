# UCB1-TUNED RERANKER — DESIGN ANALYSIS & IMPLEMENTATION PLAN

## 📋 USER DECISIONS (LOCKED IN)

### Training Data
- **Source**: HotpotQA distractor **train split** (full, ~90k examples)
- **Supporting Facts**: Extract from `supporting_facts["title"]` — direct title matching (no normalization)
- **Reward Signal**: Binary — title appears in supporting_facts ⇒ reward=1.0, else reward=0.0

### Inference Setup
- **Test Splits**: Evaluate on HotpotQA distractor **validation split** (~10k examples)
- **K Values**: Test multiple in one run — **k ∈ {2, 3, 5}**
- **Pattern**: Like noise_gate — loop over k values in `_process_sample()` ⇒ one CSV with multiple rows per question_id

### Cold Start Strategy (Unseen Titles)
- **Primary**: Optimistic init (return 1.0 for unseen titles)
- **Fallback**: BM25 text similarity to question (if available)
  - Titles never seen at training (~5–15% of validation sections) → compute BM25(title text, question)
  - Rationale: Simple, interpretable, no LLM cost

### CSV Output Columns (Option B — Track Reranker Behavior)
```
question_id, method, k, answer_pred, answer_gt, em, f1, precision, recall,
input_tokens, output_tokens, cost_usd, selected_titles, unseen_fallback
```
- **selected_titles**: JSON list of top-k selected titles (for analysis)
- **unseen_fallback**: Boolean — was this question affected by cold start?

---

## 🏗️ IMPLEMENTATION ARCHITECTURE

### Module 1: `experiments/ucb1_tuned.py`
**The reranker class** (no I/O, no LLM calls)

```python
@dataclass
class TitleStats:
    mean: float = 0.0       # Running mean of rewards (F1 or EM)
    M: float = 0.0          # Welford sum of squared deviations
    count: int = 0          # Number of times seen

class UCB1TunedReranker:
    def __init__(self, optimistic_init: float = 1.0):
        self.scoreboard: dict[str, TitleStats] = {}
        self.t: int = 0  # Total observations
        self.optimistic_init = optimistic_init

    # Training
    def update(self, title: str, reward: float) -> None
    def train(self, dataset) -> None
    
    # Scoring
    def score(self, title: str) -> float
    def select(self, titles: list[str], k: int) -> list[tuple[str, float]]
    
    # Persistence
    def save(self, path: str) -> None
    def load(self, path: str) -> None

    # BM25 cold start (optional)
    def set_bm25_fallback(self, questions_and_titles) -> None
    def get_bm25_score(self, title: str, question: str) -> float
```

### Module 2: `train_ucb1_tuned.py`
**One-time training script** — runs once, saves scoreboard

```
Usage:
  uv run python -m experiments.train_ucb1_tuned [--output-path <path>]

Flow:
  1. Load HotpotQA train split (full)
  2. Extract all (title, reward) pairs from supporting_facts
  3. Call reranker.train(dataset)
  4. Save scoreboard to JSON (default: experiments/cache/ucb1_scoreboard.json)
  5. Log stats: unique titles, reward distribution, etc.
```

### Module 3: `run_ucb1_tuned.py`
**Many inference runs** — loads saved scoreboard, evaluates at k=2,3,5

```
Usage:
  uv run python -m experiments.run_ucb1_tuned gpt-4o-mini 100
  uv run python -m experiments.run_ucb1_tuned claude-sonnet-4-6 100 20

Flow:
  1. Load scoreboard from JSON
  2. Load validation split (n_rows from CLI)
  3. For each sample & k in {2, 3, 5}:
     a. Get top-k selected titles via reranker.select()
     b. Call LLM with selected context
     c. Compute metrics
     d. Track unseen_fallback status
  4. Save to experiments/results/ucb1/ucb1_{model}.csv
```

---

## 📊 DATA FLOW — DETAILED

### Training Phase

**Input**: HotpotQA train split
```json
{
  "id": "5ae143ed55429920d5234360",
  "question": "Are Assi Heredia and Gianni Morandi both entertainers?",
  "answer": "yes",
  "supporting_facts": {
    "title": ["Assi Heredia", "Gianni Morandi"],  // ← Extract these
    "sent_id": [[0, 1], [2]]
  },
  "context": {
    "title": ["Assi Heredia", "Gianni Morandi", "Other Title 1", ...],
    "sentences": [["sent 0", "sent 1", ...], [...], ...]
  }
}
```

**Processing**:
1. Extract `gold_titles = set(supporting_facts["title"])`
2. For each title in `context["title"]`:
   - reward = 1.0 if title in gold_titles, else 0.0
   - Call `reranker.update(title, reward)`

**Output**: Trained reranker with scoreboard
```json
{
  "t": 900000,
  "scoreboard": {
    "Assi Heredia": {"mean": 0.71, "M": 123.45, "count": 847},
    "Gianni Morandi": {"mean": 0.65, "M": 89.12, "count": 623},
    ...
  }
}
```

---

### Inference Phase

**Input**: Validation example + loaded scoreboard
```json
{
  "id": "5ae143ed55429920d5234360",
  "question": "...",
  "answer": "yes",
  "context": {
    "title": ["Title A", "Title B", "Title C", ...],  // 10 titles
    "sentences": [...]
  }
}
```

**For each k in {2, 3, 5}**:
1. Score all 10 titles:
   ```
   For each title:
     if title in scoreboard:
       score = mean + bonus (UCB formula)
     else:
       score = optimistic_init OR bm25(title, question)
   ```
2. Sort by score, select top-k
3. Build retrieval prompt with selected context
4. Call LLM → get answer
5. Compute metrics (EM, F1, etc.)
6. Record: `selected_titles`, `unseen_fallback`

**Output**: CSV row (one per k value per question)
```csv
question_id, method, k, answer_pred, answer_gt, em, f1, ..., cost_usd, selected_titles, unseen_fallback
5ae143..., ucb1, 2, "yes", "yes", 1.0, 1.0, ..., 0.0015, ["Title A", "Title B"], false
5ae143..., ucb1, 3, "yes", "yes", 1.0, 1.0, ..., 0.0018, ["Title A", "Title B", "Title C"], false
5ae143..., ucb1, 5, "yes", "yes", 1.0, 1.0, ..., 0.0024, [...], true
```

---

## 🔧 TECHNICAL DETAILS

### Reward Signal — Why Binary (0/1)?
- Supporting facts are sparse (avg 2–3 titles per question)
- A title is either "needed for the chain of reasoning" (1) or not (0)
- Not using partial credit (e.g., F1 of sections) — too expensive at training time (90k examples × ~10 titles = 900k updates)

### BM25 Fallback Implementation
```python
# At reranker init or after loading scoreboard:
def set_bm25_fallback(self, question_text: str, all_titles: list[str]) -> None:
    from rank_bm25 import BM25Okapi
    # Tokenize titles
    tokenized = [title.lower().split() for title in all_titles]
    self.bm25 = BM25Okapi(tokenized)
    self.bm25_question_tokens = question_text.lower().split()

def get_bm25_score(self, title: str) -> float:
    if not hasattr(self, 'bm25'):
        return self.optimistic_init
    tokens = title.lower().split()
    return self.bm25.get_scores(self.bm25_question_tokens)[index_of_title]
```

### Unseen Title Detection
```python
def select(self, titles: list[str], k: int) -> tuple[list[str], bool]:
    scored = []
    unseen_count = 0
    for title in titles:
        if title not in self.scoreboard:
            unseen_count += 1
        score = self.score(title)
        scored.append((title, score))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    selected = [t for t, _ in scored[:k]]
    had_unseen = unseen_count > 0
    return selected, had_unseen
```

---

## 📁 FILE STRUCTURE

```
experiments/
├── ucb1_tuned.py              [NEW] Reranker class
├── train_ucb1_tuned.py        [NEW] Training script
├── run_ucb1_tuned.py          [NEW] Inference script
├── cache/
│   └── ucb1_scoreboard.json   [NEW] Trained scoreboard (after first train run)
├── results/
│   └── ucb1/                  [NEW] Results directory
│       ├── ucb1_gpt-4o-mini.csv
│       ├── ucb1_claude-sonnet-4-6.csv
│       └── ...
```

---

## ✅ WHAT'S CLEAR

1. **Dataset Loading**: `load_dataset_cached()` handles train & validation splits seamlessly
2. **Evaluation**: `evaluate_sample()` computes EM, F1, precision, recall (same as baseline/noise_gate)
3. **LLM Integration**: Use same approach as `run_baseline.py` — import from `llm_openai`, `llm_anthropic`
4. **Parallel Processing**: ThreadPoolExecutor pattern already proven in `run_baseline.py`
5. **CSV Saving**: Use `save_results_csv()` from utils.py with configurable fieldnames
6. **Prompt Building**: Use `build_retrieval_prompt()` and `build_always_think_prompt()` from preprocessing.py
7. **Title Extraction**: Just grab `supporting_facts["title"]` — direct, no normalization needed

---

## 🤔 MINOR DECISIONS STILL NEEDED

1. **BM25 Library**: Use `rank-bm25` (lightweight) or another?
2. **Scoreboard Path**: Default `experiments/cache/ucb1_scoreboard.json` — ok?
3. **Results Directory**: `experiments/results/ucb1/` — ok?
4. **Results Baseline**: Compare against noise_gate + always_think, or add fixed-k baseline?
5. **Analysis Script**: After runs, should we add `analyse_ucb1.py` to aggregate metrics (like `analyse_baseline.py`)?

---

## 🚀 IMPLEMENTATION SEQUENCE

1. **Create `ucb1_tuned.py`** — core reranker with Welford, UCB formula, BM25
2. **Create `train_ucb1_tuned.py`** — loads train split, trains, saves JSON
3. **Create `run_ucb1_tuned.py`** — loads scoreboard, runs inference on validation
4. **Create results directory** — `experiments/results/ucb1/`
5. **Test with small n** — e.g., `uv run python -m experiments.train_ucb1_tuned` then `uv run python -m experiments.run_ucb1_tuned gpt-4o-mini 10`
6. **Create analysis script** (optional) — aggregate results like other methods

---

## 📈 EXPECTED OUTCOMES

- **Unseen titles at validation**: 5–15% (tracked in CSV)
- **Selected titles diversity**: Should favor high-signal Wikipedia articles (names, places, events)
- **F1 vs token reduction**: With k=3 out of 10, ~65% token savings; F1 likely 85–95% of k=10 baseline
- **Cold start impact**: If BM25 fallback works, should mitigate unseen title penalty

