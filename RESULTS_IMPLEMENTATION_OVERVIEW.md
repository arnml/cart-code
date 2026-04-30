# Results Implementation Overview

This file consolidates the code paths that generate the markdown result reports in this repo.
It is meant as a single place to inspect how each experiment writes CSV results and how the
corresponding analysis script turns those CSVs into `analysis_*.md`.

The code blocks below are excerpts of the core logic, not full file dumps.

Covered pipelines:

- fixed baselines
- noise gate
- adaptive-k
- UCB1-TUNED
- LinUCB

Note on UCB1:

- `experiments/run_ucb1_tuned.py` + `experiments/ucb1_tuned.py` is the current rank-position
  pipeline that writes `experiments/results/ucb1/`.
- `experiments/run_ucb1_title_tuned.py` + `experiments/ucb1_title_tuned.py` is a legacy
  title-based variant kept for reproducibility and diagnostics.

## Where The Markdown Reports Come From

| Pipeline | CSV writer | Markdown writer | Output folder |
|---|---|---|---|
| Fixed baselines | `experiments/run_baseline.py` | `experiments/analyse_baseline.py` | `experiments/results/baseline/` |
| Noise gate | `experiments/run_noise_gate.py` | `experiments/analyse_noise_gate.py` | `experiments/results/cart/` |
| Adaptive-k | `experiments/run_adaptive_k.py` | `experiments/analyse_adaptive_k.py` | `experiments/results/cart/` |
| UCB1-TUNED | `experiments/run_ucb1_tuned.py` | `experiments/analyse_ucb1.py` | `experiments/results/ucb1/` |
| LinUCB | `experiments/run_linucb.py` | `experiments/analyse_linucb.py` | `experiments/results/linucb/` |

The generated markdown files already in the repo follow the same pattern:

- `experiments/results/baseline/analysis_*.md`
- `experiments/results/cart/analysis_*.md`
- `experiments/results/ucb1/analysis_*.md`
- `experiments/results/linucb/analysis_*.md`

## Shared Result Pattern

The analysis scripts all do the same thing:

1. load a CSV
2. aggregate rows by method, threshold, or `k`
3. render a markdown table
4. write `analysis_*.md`

The shared aggregation helpers live in `experiments/utils.py`:

```python
def load_csv(csv_path: Path) -> list[dict[str, str]]: ...
def aggregate_metrics(
    results: list[dict[str, str]],
    include_k_star: bool = False,
) -> dict[str, Any]: ...
def aggregate_by_method(
    results: list[dict[str, str]],
) -> dict[str, dict[str, float]]: ...
def aggregate_by_k(
    results: list[dict[str, str]],
) -> dict[int, dict[str, float]]: ...
```

## Fixed Baselines

Files:

- [experiments/run_baseline.py](experiments/run_baseline.py)
- [experiments/analyse_baseline.py](experiments/analyse_baseline.py)

The baseline runner computes one retrieval ranking per sample and reuses it for the
retrieval baselines:

```python
RETRIEVAL_METHOD_TO_K = {
    "retrieval_k3": 3,
    "retrieval_k5": 5,
    "retrieval_k10": 10,
}

def _process_sample(sample_idx: int, sample: dict[str, Any], model: str):
    qid = sample["id"]
    question = sample["question"]
    sample_results = []
    statuses = []

    top_paragraphs = None
    if any(method_name in RETRIEVAL_METHOD_TO_K for method_name in METHODS):
        paragraphs = flatten_context(sample["context"])
        emb_config = LLM_TO_EMBEDDING[model]
        top_paragraphs, _ = retrieve_top_k(
            question=question,
            paragraphs=paragraphs,
            k=10,
            provider=emb_config["provider"],
            embedding_model=emb_config["embedding_model"],
            token_budget=emb_config["max_tokens"],
        )

    for method_name in METHODS:
        if method_name == "always_think":
            prompt = build_always_think_prompt(question)
            pred, input_tokens, output_tokens, cost_usd = call_llm(prompt, model)
        elif method_name in RETRIEVAL_METHOD_TO_K:
            k = RETRIEVAL_METHOD_TO_K[method_name]
            prompt = build_retrieval_prompt(question, top_paragraphs[:k])
            pred, input_tokens, output_tokens, cost_usd = call_llm(prompt, model)
```

The markdown summary groups rows by method and writes a simple table:

```python
def aggregate_by_method(results: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    ...

def generate_summary(model: str, aggregated: dict[str, dict[str, float]]) -> str:
    lines = [
        f"# Baseline Analysis: {model}\\n",
        "## Summary by Method\\n",
        "| Method | Count | input | output | total token | cost_usd | EM | F1 | Precision | Recall | Token-Eff | Cost-Eff |",
    ]
    ...
    return "\\n".join(lines) + "\\n"

def save_summary(model: str, summary: str) -> None:
    md_path = results_dir / f"analysis_{model}.md"
    md_path.write_text(summary, encoding="utf-8")
```

## Noise Gate

Files:

- [experiments/run_noise_gate.py](experiments/run_noise_gate.py)
- [experiments/analyse_noise_gate.py](experiments/analyse_noise_gate.py)

The noise gate scores all paragraphs against the question, then filters by cosine similarity and
Jaccard redundancy:

```python
NOISE_GATE_SIM_THRESHOLDS = (0.2, 0.25, 0.3, 0.35, 0.5)
NOISE_GATE_REDUNDANCY_THRESHOLD = 0.65

def _apply_noise_gate(
    docs: list[str],
    scores: list[float],
    sim_threshold: float,
    jac_threshold: float,
) -> list[str]:
    filtered_docs = []
    seen_docs = []

    for doc, score in zip(docs, scores, strict=True):
        if score < sim_threshold:
            continue
        if any(_jaccard(doc, prev_doc) > jac_threshold for prev_doc in seen_docs):
            continue
        filtered_docs.append(doc)
        seen_docs.append(doc)

    return filtered_docs

def noise_gate_select(
    question: str,
    paragraphs: list[str],
    provider: str,
    embedding_model: str,
    token_budget: int,
    sim_threshold: float,
    jac_threshold: float = NOISE_GATE_REDUNDANCY_THRESHOLD,
) -> list[str]:
    question_embedding = embed_text(question, provider, embedding_model, token_budget)
    paragraph_embeddings = [
        embed_text(paragraph, provider, embedding_model, token_budget)
        for paragraph in paragraphs
    ]
    similarities = cosine_similarity(question_arr, paragraph_arr)[0]
    ranked_indices = np.argsort(similarities)[::-1]
    ranked_docs = [paragraphs[i] for i in ranked_indices]
    ranked_scores = [float(similarities[i]) for i in ranked_indices]
    return _apply_noise_gate(ranked_docs, ranked_scores, sim_threshold, jac_threshold)
```

The markdown analyzer groups by threshold:

```python
def aggregate_by_threshold(results: list[dict[str, str]]) -> dict[float, dict[str, float]]:
    ...

def generate_summary(model: str, aggregated: dict[float, dict[str, float]]) -> str:
    thresholds = sorted(aggregated.keys())
    lines = [
        f"# Noise-Gate Analysis: {model}\\n",
        "| Metric | 0.2 | 0.25 | 0.3 | 0.35 | 0.5 |",
        ...
    ]
    return "\\n".join(lines) + "\\n"
```

## Adaptive-k

Files:

- [experiments/run_adaptive_k.py](experiments/run_adaptive_k.py)
- [experiments/analyse_adaptive_k.py](experiments/analyse_adaptive_k.py)

The adaptive-k selector finds the largest adjacent similarity gap:

```python
def adaptive_k_select(scores: list[float], delta: float = 0.08) -> int:
    if len(scores) <= 1:
        return 1

    max_gap = 0
    max_gap_idx = 1

    for i in range(len(scores) - 1):
        gap = scores[i] - scores[i + 1]
        if gap > max_gap:
            max_gap = gap
            max_gap_idx = i + 1

    if max_gap < delta:
        return min(5, len(scores))

    return max_gap_idx

def run_adaptive_k_method(sample: dict, model: str):
    paragraphs = flatten_context(sample["context"])
    emb_config = LLM_TO_EMBEDDING[model]
    top_paragraphs, top_scores = retrieve_top_k(
        question=sample["question"],
        paragraphs=paragraphs,
        k=10,
        provider=emb_config["provider"],
        embedding_model=emb_config["embedding_model"],
        token_budget=emb_config["max_tokens"],
    )
    k_star = adaptive_k_select(top_scores, delta=0.08)
    selected_paragraphs = top_paragraphs[:k_star]
    prompt = build_retrieval_prompt(sample["question"], selected_paragraphs)
    answer, input_tokens, output_tokens, cost_usd = call_llm(prompt, model)
    return answer, input_tokens, output_tokens, cost_usd, k_star
```

The markdown analyzer stores the main metrics plus `k_star` statistics:

```python
def aggregate_metrics(results: list[dict[str, str]]) -> dict[str, float]:
    ...
    if k_stars:
        aggregated["k_star_count"] = len(k_stars)
        aggregated["k_star_min"] = min(k_stars)
        aggregated["k_star_max"] = max(k_stars)
        aggregated["k_star_mean"] = statistics.mean(k_stars)
        aggregated["k_star_median"] = statistics.median(k_stars)
```

## UCB1-TUNED

Current pipeline files:

- [experiments/ucb1_tuned.py](experiments/ucb1_tuned.py)
- [experiments/train_ucb1_tuned.py](experiments/train_ucb1_tuned.py)
- [experiments/run_ucb1_tuned.py](experiments/run_ucb1_tuned.py)
- [experiments/analyse_ucb1.py](experiments/analyse_ucb1.py)

Current implementation note: this version scores rank positions, not title strings.

```python
@dataclass
class TitleStats:
    mean: float = 0.0
    M: float = 0.0
    count: int = 0

class UCB1TunedReranker:
    def update(self, rank: int, reward: float) -> None:
        ...

    def score(self, rank: int) -> float:
        stats = self.scoreboard[rank]
        variance = stats.M / stats.count
        bonus = math.sqrt(math.log(self.t) / stats.count * min(0.25, variance))
        return stats.mean + bonus

    def select(self, num_ranks: int, k: int) -> list[int]:
        scored = [(rank, self.score(rank)) for rank in range(num_ranks)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [rank for rank, _ in scored[:k]]
```

The inference runner uses those selected rank positions to build the prompt:

```python
selected_ranks = reranker.select(num_ranks, k)
selected_paragraphs = [paragraphs[rank] for rank in selected_ranks]
selected_titles = [titles[rank] for rank in selected_ranks]
prompt = build_retrieval_prompt(question, selected_paragraphs)
```

The markdown analysis reuses the shared `k` aggregation helpers:

```python
results = load_csv(csv_path)
by_k = aggregate_by_k(results)
summary = generate_summary(model, results, by_k)
output_path = results_dir / f"analysis_ucb1_{model}.md"
```

Legacy title-based files kept in the repo:

- [experiments/ucb1_title_tuned.py](experiments/ucb1_title_tuned.py)
- [experiments/train_ucb1_title_tuned.py](experiments/train_ucb1_title_tuned.py)
- [experiments/run_ucb1_title_tuned.py](experiments/run_ucb1_title_tuned.py)

## LinUCB

Files:

- [experiments/linucb.py](experiments/linucb.py)
- [experiments/train_linucb.py](experiments/train_linucb.py)
- [experiments/run_linucb.py](experiments/run_linucb.py)
- [experiments/analyse_linucb.py](experiments/analyse_linucb.py)

LinUCB uses a small lexical feature vector and a linear UCB score:

```python
class LinUCBReranker:
    def _extract_features(self, question: str, title: str, section_text: str) -> np.ndarray:
        question_tokens = set(re.findall(r"\\w+", question.lower()))
        title_tokens = set(re.findall(r"\\w+", title.lower()))

        f1 = len(question_tokens & title_tokens) / (len(title_tokens) + 1e-9)
        f2 = min(len(question_tokens) / 50.0, 1.0)

        section_tokens = set(re.findall(r"\\w+", section_text.lower()))
        f3 = min(len(section_tokens) / 200.0, 1.0)

        return np.array([f1, f2, f3], dtype=np.float32)

    def score(self, question: str, title: str, section_text: str) -> float:
        x = self._extract_features(question, title, section_text)
        w = np.linalg.solve(self.A_global, self.b_global)
        exploit = np.dot(w, x)
        A_inv = np.linalg.inv(self.A_global)
        uncertainty = np.sqrt(np.dot(x, A_inv @ x))
        return float(exploit + self.alpha * uncertainty)

    def select(self, question: str, titles: list[str], sentences_list: list[list[str]], k: int) -> list[str]:
        scores = []
        for title, sentences in zip(titles, sentences_list):
            section_text = " ".join(sentences)
            scores.append((title, self.score(question, title, section_text)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [title for title, _ in scores[:k]]
```

The inference runner turns selected titles into prompts:

```python
selected_titles = reranker.select(question, titles, sentences_list, k)
selected_paragraphs = [paragraphs[titles.index(t)] for t in selected_titles]
prompt = build_retrieval_prompt(question, selected_paragraphs)
```

The markdown analyzer again uses `aggregate_by_k` and `aggregate_metrics`:

```python
results = load_csv(csv_path)
by_k = aggregate_by_k(results)
summary = generate_summary(model, results, by_k)
output_path = results_dir / f"analysis_linucb_{model}.md"
```

## Useful Commands

Generate CSV results:

```bash
uv run python -m experiments.run_baseline gpt-5.4-mini 100
uv run python -m experiments.run_noise_gate gpt-5.4-mini 100
uv run python -m experiments.run_adaptive_k gpt-5.4-mini 100
uv run python -m experiments.train_ucb1_tuned
uv run python -m experiments.run_ucb1_tuned gpt-5.4-mini 100
uv run python -m experiments.train_linucb
uv run python -m experiments.run_linucb gpt-5.4-mini 100
```

Generate markdown analyses:

```bash
uv run python -m experiments.analyse_baseline gpt-5.4-mini
uv run python -m experiments.analyse_noise_gate gpt-5.4-mini
uv run python -m experiments.analyse_adaptive_k gpt-5.4-mini
uv run python -m experiments.analyse_ucb1 gpt-5.4-mini
uv run python -m experiments.analyse_linucb gpt-5.4-mini
```
