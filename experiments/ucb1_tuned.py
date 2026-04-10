"""UCB1-TUNED rank-position reranker for HotpotQA.

Implements a contextual bandit algorithm that learns which rank positions (0-9)
are high-value across the training split. Each context section is pre-ranked by
BM25, so "rank 0" always means the highest-scoring BM25 section.

The reranker uses:
- Rank positions as arms (0 through 9)
- Welford online mean/variance tracking (no history storage)
- UCB1-TUNED exploration bonus: mean + sqrt(ln(t)/n_i * min(0.25, V_i))
- Binary reward signal: rank position in supporting_facts → 1.0, else 0.0

Usage:
    # Training (one-time):
    from experiments.ucb1_tuned import UCB1TunedReranker
    reranker = UCB1TunedReranker()
    reranker.train(dataset)
    reranker.save("path/to/scoreboard.json")

    # Inference:
    reranker = UCB1TunedReranker()
    reranker.load("path/to/scoreboard.json")
    selected_ranks = reranker.select(num_ranks=10, k=3)
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TitleStats:
    """Running statistics for a single title using Welford's method."""

    mean: float = 0.0  # Running mean of rewards
    M: float = 0.0  # Welford sum of squared deviations (for variance)
    count: int = 0  # Number of observations


class UCB1TunedReranker:
    """UCB1-TUNED contextual bandit reranker for rank position selection."""

    def __init__(self, num_ranks: int = 10) -> None:
        """Initialize the reranker.

        Args:
            num_ranks: Number of rank positions (default 10 for HotpotQA context)
        """
        self.num_ranks = num_ranks
        self.scoreboard: dict[int, TitleStats] = {}
        self.t: int = 0  # Total observations across all ranks

        # Initialize scoreboard for all rank positions
        for rank in range(num_ranks):
            self.scoreboard[rank] = TitleStats()

    def update(self, rank: int, reward: float) -> None:
        """Update statistics for a rank position using Welford's online algorithm.

        Args:
            rank: Rank position (0-9)
            reward: Reward for this observation (0.0 or 1.0)
        """
        if rank < 0 or rank >= self.num_ranks:
            return

        stats = self.scoreboard[rank]
        stats.count += 1
        self.t += 1

        # Welford online mean/variance update
        delta = reward - stats.mean
        stats.mean += delta / stats.count
        delta2 = reward - stats.mean  # Note: uses NEW mean, not old
        stats.M += delta * delta2

    def train(self, dataset: list[dict[str, Any]]) -> None:
        """Train the reranker on a dataset.

        For each sample, extracts gold titles from supporting_facts,
        then marks reward for each rank position based on whether that
        position contains a gold title.

        Args:
            dataset: List of HotpotQA samples with 'supporting_facts' and 'context'
        """
        for sample in dataset:
            # Extract gold titles from supporting_facts
            supporting_facts = sample.get("supporting_facts", {})
            gold_titles: set[str] = set(supporting_facts.get("title", []))

            # Get titles in rank order from context
            context = sample.get("context", {})
            context_titles = context.get("title", [])

            # Update stats for each rank position
            for rank, title in enumerate(context_titles):
                reward = 1.0 if title in gold_titles else 0.0
                self.update(rank, reward)

    def score(self, rank: int) -> float:
        """Compute UCB1-TUNED score for a rank position.

        UCB1-TUNED formula:
            score = mean + sqrt(ln(t) / n_i * min(0.25, V_i))

        where:
            - mean: empirical mean reward for rank i
            - t: total observations across all ranks
            - n_i: number of observations for rank i
            - V_i: sample variance for rank i

        Args:
            rank: Rank position (0-9)

        Returns:
            UCB1-TUNED score
        """
        if rank < 0 or rank >= self.num_ranks:
            return 0.0

        stats = self.scoreboard[rank]
        if stats.count == 0:
            return 0.0

        # Compute sample variance: V_i = M / count
        variance = stats.M / stats.count

        # UCB1-TUNED exploration bonus
        if self.t <= 1:
            bonus = 0.0  # Avoid log(0) or log(1)
        else:
            exploration_term = math.log(self.t) / stats.count
            bonus = math.sqrt(exploration_term * min(0.25, variance))

        return stats.mean + bonus

    def select(self, num_ranks: int, k: int) -> list[int]:
        """Select top-k rank positions by UCB1-TUNED score.

        Args:
            num_ranks: Number of available rank positions
            k: Number of positions to select

        Returns:
            List of selected rank positions (integers), sorted by score descending
        """
        if num_ranks <= 0 or k <= 0:
            return []

        # Score all available ranks
        scored: list[tuple[int, float]] = []
        for rank in range(num_ranks):
            score = self.score(rank)
            scored.append((rank, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Select top-k
        selected = [rank for rank, _ in scored[:k]]

        return selected

    def save(self, path: str) -> None:
        """Save the reranker to a JSON file.

        Args:
            path: File path for the JSON snapshot
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "t": self.t,
            "num_ranks": self.num_ranks,
            "scoreboard": {
                str(rank): asdict(stats) for rank, stats in self.scoreboard.items()
            },
        }

        with open(path_obj, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load the reranker from a JSON file.

        Args:
            path: File path for the JSON snapshot
        """
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Scoreboard not found at {path_obj}")

        with open(path_obj, "r") as f:
            data = json.load(f)

        self.t = data.get("t", 0)
        self.num_ranks = data.get("num_ranks", 10)

        self.scoreboard = {}
        for rank_str, stats_dict in data.get("scoreboard", {}).items():
            rank = int(rank_str)
            self.scoreboard[rank] = TitleStats(
                mean=stats_dict.get("mean", 0.0),
                M=stats_dict.get("M", 0.0),
                count=stats_dict.get("count", 0),
            )
