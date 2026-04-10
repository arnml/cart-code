"""LinUCB (Contextual Linear Bandit) reranker for HotpotQA.

LinUCB learns a linear model of section relevance using features that compare
the question to each context section. It handles the random shuffle of HotpotQA
distractor by learning from question-content relationships rather than fixed
titles or rank positions.

Key insight: title_in_question (from diagnostic) is a 2.61x predictor of gold.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ArmState:
    """Per-arm (section) statistics for LinUCB."""
    A: np.ndarray  # d x d design matrix (XtX)
    b: np.ndarray  # d-length vector (Xty)
    count: int = 0  # number of times this arm was pulled


class LinUCBReranker:
    """Contextual linear bandit reranker.

    Learns a linear model: reward ≈ w^T * features(question, section)
    Selects arms with highest UCB1 score: w^T*x + alpha*sqrt(x^T*(A^-1)*x)
    """

    def __init__(self, d: int = 3, alpha: float = 1.0):
        """Initialize LinUCB reranker.

        Args:
            d: feature dimension (default: 3 features)
            alpha: exploration bonus coefficient
        """
        self.d = d  # feature dimension
        self.alpha = alpha
        self.A_global = np.eye(d)  # global design matrix (XtX)
        self.b_global = np.zeros(d)  # global response vector (Xty)
        self.arms: dict[str, ArmState] = {}  # per-arm statistics
        self.t = 0  # total training observations

    def _extract_features(
        self, question: str, title: str, section_text: str
    ) -> np.ndarray:
        """Extract feature vector for question-section pair.

        Features:
        1. title_in_question: fraction of title words in question
        2. question_length: normalized question length
        3. section_length: normalized section length

        Args:
            question: question text
            title: section title
            section_text: section body

        Returns:
            feature vector [f1, f2, f3]
        """
        question_tokens = set(re.findall(r'\w+', question.lower()))
        title_tokens = set(re.findall(r'\w+', title.lower()))

        # Feature 1: title_in_question
        f1 = len(question_tokens & title_tokens) / (len(title_tokens) + 1e-9)

        # Feature 2: question_length (normalized)
        f2 = min(len(question_tokens) / 50.0, 1.0)  # normalize to ~[0, 1]

        # Feature 3: section_length (normalized)
        section_tokens = set(re.findall(r'\w+', section_text.lower()))
        f3 = min(len(section_tokens) / 200.0, 1.0)  # normalize to ~[0, 1]

        return np.array([f1, f2, f3], dtype=np.float32)

    def update(
        self, question: str, title: str, section_text: str, reward: float
    ) -> None:
        """Update model with observed (features, reward) pair.

        Args:
            question: question text
            title: section title
            section_text: section body
            reward: binary reward (1.0 if gold, 0.0 otherwise)
        """
        x = self._extract_features(question, title, section_text)

        # Update global design matrix and response
        self.A_global += np.outer(x, x)
        self.b_global += reward * x
        self.t += 1

        # Track per-arm statistics (optional: useful for debugging)
        arm_key = title
        if arm_key not in self.arms:
            self.arms[arm_key] = ArmState(
                A=np.eye(self.d),
                b=np.zeros(self.d),
            )

        arm_state = self.arms[arm_key]
        arm_state.A += np.outer(x, x)
        arm_state.b += reward * x
        arm_state.count += 1

    def train(self, dataset: list[dict[str, Any]]) -> None:
        """Train on full dataset.

        Args:
            dataset: list of HotpotQA samples
        """
        for example in dataset:
            question = example["question"]
            gold_titles = set(example["supporting_facts"]["title"])

            titles = example["context"]["title"]
            sentences_list = example["context"]["sentences"]

            for title, sentences in zip(titles, sentences_list):
                section_text = " ".join(sentences)
                reward = 1.0 if title in gold_titles else 0.0
                self.update(question, title, section_text, reward)

    def score(
        self, question: str, title: str, section_text: str
    ) -> float:
        """Compute LinUCB score (exploitation + exploration).

        Score = w^T * x + alpha * sqrt(x^T * A^{-1} * x)

        Args:
            question: question text
            title: section title
            section_text: section body

        Returns:
            UCB score for this arm
        """
        x = self._extract_features(question, title, section_text)

        # Exploit: estimated expected reward
        try:
            w = np.linalg.solve(self.A_global, self.b_global)
        except np.linalg.LinAlgError:
            # Singular matrix (shouldn't happen if t > 0)
            w = np.zeros(self.d)

        exploit = np.dot(w, x)

        # Explore: uncertainty bonus
        try:
            A_inv = np.linalg.inv(self.A_global)
        except np.linalg.LinAlgError:
            A_inv = np.eye(self.d)  # fallback

        uncertainty = np.sqrt(np.dot(x, A_inv @ x))
        explore = self.alpha * uncertainty

        return float(exploit + explore)

    def select(
        self,
        question: str,
        titles: list[str],
        sentences_list: list[list[str]],
        k: int,
    ) -> list[str]:
        """Select top-k sections via LinUCB scores.

        Args:
            question: question text
            titles: list of section titles
            sentences_list: list of sentence lists
            k: number of sections to select

        Returns:
            list of top-k selected titles (in descending score order)
        """
        scores = []
        for title, sentences in zip(titles, sentences_list):
            section_text = " ".join(sentences)
            score = self.score(question, title, section_text)
            scores.append((title, score))

        # Sort by score descending, return top-k titles
        scores.sort(key=lambda x: x[1], reverse=True)
        return [title for title, _ in scores[:k]]

    def save(self, path: str) -> None:
        """Serialize model to JSON.

        Args:
            path: output file path
        """
        state = {
            "d": self.d,
            "alpha": self.alpha,
            "t": self.t,
            "A_global": self.A_global.tolist(),
            "b_global": self.b_global.tolist(),
            "arms": {
                title: {
                    "A": arm.A.tolist(),
                    "b": arm.b.tolist(),
                    "count": arm.count,
                }
                for title, arm in self.arms.items()
            },
        }

        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def load(self, path: str) -> None:
        """Deserialize model from JSON.

        Args:
            path: input file path
        """
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        self.d = state["d"]
        self.alpha = state["alpha"]
        self.t = state["t"]
        self.A_global = np.array(state["A_global"], dtype=np.float32)
        self.b_global = np.array(state["b_global"], dtype=np.float32)

        self.arms = {}
        for title, arm_dict in state.get("arms", {}).items():
            self.arms[title] = ArmState(
                A=np.array(arm_dict["A"], dtype=np.float32),
                b=np.array(arm_dict["b"], dtype=np.float32),
                count=arm_dict["count"],
            )
