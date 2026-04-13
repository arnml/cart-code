"""Legacy UCB1-TUNED reranker for HotpotQA title selection.

This is the older title-based variant of the UCB1 experiment. It is kept
alongside the current rank-based implementation so the cold-start diagnostic
and BM25-fallback behavior remain reproducible.

The reranker uses:
- Welford online mean/variance tracking (no history storage)
- UCB1-TUNED exploration bonus: mean + sqrt(ln(t)/n_i * min(0.25, V_i))
- Per-question BM25 fallback for unseen titles
- Binary reward signal: title in supporting_facts -> 1.0, else 0.0

Usage:
    # Training (one-time):
    from experiments.ucb1_title_tuned import UCB1TitleTunedReranker
    reranker = UCB1TitleTunedReranker()
    reranker.train(dataset)
    reranker.save("path/to/scoreboard.json")

    # Inference:
    reranker = UCB1TitleTunedReranker()
    reranker.load("path/to/scoreboard.json")
    reranker.set_bm25_fallback(question, context_titles)
    selected_titles, had_unseen = reranker.select(context_titles, k=3, question=question)
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TitleStats:
    """Running statistics for a single title using Welford's method."""

    mean: float = 0.0
    M: float = 0.0
    count: int = 0


class UCB1TitleTunedReranker:
    """UCB1-TUNED contextual bandit reranker for title selection."""

    def __init__(self, optimistic_init: float = 1.0) -> None:
        """Initialize the reranker."""
        self.scoreboard: dict[str, TitleStats] = {}
        self.t: int = 0
        self.optimistic_init = optimistic_init
        self._bm25 = None
        self._bm25_titles: list[str] = []
        self._bm25_query: list[str] = []

    def update(self, title: str, reward: float) -> None:
        """Update statistics for a title using Welford's online algorithm."""
        if title not in self.scoreboard:
            self.scoreboard[title] = TitleStats()

        stats = self.scoreboard[title]
        stats.count += 1
        self.t += 1

        delta = reward - stats.mean
        stats.mean += delta / stats.count
        delta2 = reward - stats.mean
        stats.M += delta * delta2

    def train(self, dataset: list[dict[str, Any]]) -> None:
        """Train the reranker on a dataset."""
        for sample in dataset:
            supporting_facts = sample.get("supporting_facts", {})
            gold_titles: set[str] = set(supporting_facts.get("title", []))

            context = sample.get("context", {})
            all_titles = context.get("title", [])

            for title in all_titles:
                reward = 1.0 if title in gold_titles else 0.0
                self.update(title, reward)

    def score(self, title: str, question: str = "") -> float:
        """Compute UCB1-TUNED score for a title."""
        if title not in self.scoreboard:
            return self._get_bm25_score(title) if self._bm25 else self.optimistic_init

        stats = self.scoreboard[title]
        if stats.count == 0:
            return self.optimistic_init

        variance = stats.M / stats.count

        if self.t <= 1:
            bonus = 0.0
        else:
            exploration_term = math.log(self.t) / stats.count
            bonus = math.sqrt(exploration_term * min(0.25, variance))

        return stats.mean + bonus

    def select(self, titles: list[str], k: int, question: str = "") -> tuple[list[str], bool]:
        """Select top-k titles by UCB1-TUNED score."""
        if not titles:
            return [], False

        scored: list[tuple[str, float]] = []
        for title in titles:
            score = self.score(title, question)
            scored.append((title, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [title for title, _ in scored[:k]]
        had_unseen = any(title not in self.scoreboard for title in selected)
        return selected, had_unseen

    def save(self, path: str) -> None:
        """Save the reranker to a JSON file."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "t": self.t,
            "optimistic_init": self.optimistic_init,
            "scoreboard": {
                title: asdict(stats) for title, stats in self.scoreboard.items()
            },
        }

        with open(path_obj, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load the reranker from a JSON file."""
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Scoreboard not found at {path_obj}")

        with open(path_obj, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.t = data.get("t", 0)
        self.optimistic_init = data.get("optimistic_init", 1.0)

        self.scoreboard = {}
        for title, stats_dict in data.get("scoreboard", {}).items():
            self.scoreboard[title] = TitleStats(
                mean=stats_dict.get("mean", 0.0),
                M=stats_dict.get("M", 0.0),
                count=stats_dict.get("count", 0),
            )

    def set_bm25_fallback(self, question: str, titles: list[str]) -> None:
        """Set up BM25 fallback for unseen titles in this question."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            self._bm25 = None
            return

        tokenized_titles = [t.lower().split() for t in titles]
        self._bm25 = BM25Okapi(tokenized_titles)
        self._bm25_titles = list(titles)
        self._bm25_query = question.lower().split()

    def _get_bm25_score(self, title: str) -> float:
        """Get BM25 score for an unseen title."""
        if not self._bm25 or title not in self._bm25_titles:
            return self.optimistic_init

        all_scores = self._bm25.get_scores(self._bm25_query)
        idx = self._bm25_titles.index(title)
        raw_score = float(all_scores[idx])
        max_score = max(all_scores) if max(all_scores) > 0 else 1.0
        return raw_score / max_score
