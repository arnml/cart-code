"""LinUCB-Semantic (d=4) reranker: adds dense embeddings to lexical features.

Extends LinUCB with a 4th feature: cosine similarity between question and section
embeddings. This bridges the gap between symbolic (token overlap) and semantic
(dense similarity) understanding.

Feature vector: [token_overlap, title_in_question, section_length, dense_similarity]
- token_overlap: Jaccard similarity of question and section tokens
- title_in_question: Fraction of title tokens appearing in question
- section_length: Normalized count of section sentences
- dense_similarity: Cosine similarity of embeddings (pre-computed)

Usage:
    reranker = LinUCBSemantic(d=4, alpha=0.5)
    reranker.train(dataset, embeddings)
    reranker.save("path/to/model.json")

    reranker.load("path/to/model.json")
    scores = reranker.score(question, title, section_text, embeddings)
    selected = reranker.select(question, titles, sentences_list, k, embeddings)
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
    """Per-arm (section) statistics for LinUCB-Semantic."""

    A: np.ndarray  # d x d design matrix (XtX)
    b: np.ndarray  # d-length vector (Xty)
    count: int = 0  # number of times this arm was pulled


class LinUCBSemantic:
    """Contextual linear bandit reranker with semantic embeddings (d=4).

    Learns a linear model: reward ≈ w^T * features(question, section, embeddings)
    Selects arms with highest UCB1 score: w^T*x + alpha*sqrt(x^T*(A^-1)*x)

    Feature dimension: d=4
    - f1: token_overlap (lexical: Jaccard similarity)
    - f2: title_in_question (lexical: title-question match)
    - f3: section_length (structural: normalized section size)
    - f4: dense_similarity (semantic: embedding cosine similarity)
    """

    def __init__(self, d: int = 4, alpha: float = 0.5):
        """Initialize LinUCB-Semantic reranker.

        Args:
            d: feature dimension (must be 4 for semantic features)
            alpha: exploration bonus coefficient (smaller for larger feature space)
        """
        if d != 4:
            raise ValueError(
                f"LinUCBSemantic requires d=4 (token_overlap, title_in_question, "
                f"section_length, dense_similarity), got d={d}"
            )

        self.d = d
        self.alpha = alpha
        self.A_global = np.eye(d, dtype=np.float32)  # global design matrix
        self.b_global = np.zeros(d, dtype=np.float32)  # global response vector
        self.arms: dict[str, ArmState] = {}  # per-arm statistics
        self.t = 0  # total training observations

    def _extract_features(
        self,
        question: str,
        title: str,
        section_text: str,
        q_embedding: np.ndarray | None = None,
        s_embedding: np.ndarray | None = None,
    ) -> np.ndarray:
        """Extract 4-element feature vector for question-section pair.

        Features:
        1. token_overlap: Jaccard similarity (question tokens & section tokens)
        2. title_in_question: Fraction of title tokens in question
        3. section_length: Normalized count of section sentences
        4. dense_similarity: Cosine similarity of embeddings (dot product, L2-normalized)

        Args:
            question: question text
            title: section title
            section_text: section body
            q_embedding: pre-computed question embedding (shape: 384,)
            s_embedding: pre-computed section embedding (shape: 384,)

        Returns:
            feature vector [f1, f2, f3, f4] as float32 array
        """
        # Feature 1: token_overlap (Jaccard similarity)
        question_tokens = set(re.findall(r'\w+', question.lower()))
        section_tokens = set(re.findall(r'\w+', section_text.lower()))

        intersection = len(question_tokens & section_tokens)
        union = len(question_tokens | section_tokens)
        f1 = intersection / (union + 1e-9)

        # Feature 2: title_in_question
        title_tokens = set(re.findall(r'\w+', title.lower()))
        f2 = len(question_tokens & title_tokens) / (len(title_tokens) + 1e-9)

        # Feature 3: section_length (normalized)
        section_sentences = section_text.split('.')
        f3 = min(len(section_sentences) / 10.0, 1.0)

        # Feature 4: dense_similarity (cosine of L2-normalized embeddings)
        if q_embedding is not None and s_embedding is not None:
            # Embeddings are pre-normalized, so cosine similarity = dot product
            f4 = float(np.dot(q_embedding, s_embedding))
        else:
            # Fallback if embeddings not provided (e.g., during loading)
            f4 = 0.5

        return np.array([f1, f2, f3, f4], dtype=np.float32)

    def update(
        self,
        question: str,
        title: str,
        section_text: str,
        reward: float,
        q_embedding: np.ndarray | None = None,
        s_embedding: np.ndarray | None = None,
    ) -> None:
        """Update model with observed (features, reward) pair.

        Args:
            question: question text
            title: section title
            section_text: section body
            reward: binary reward (1.0 if gold, 0.0 otherwise)
            q_embedding: question embedding (shape: 384,)
            s_embedding: section embedding (shape: 384,)
        """
        x = self._extract_features(
            question,
            title,
            section_text,
            q_embedding=q_embedding,
            s_embedding=s_embedding,
        )

        # Update global design matrix and response
        self.A_global += np.outer(x, x)
        self.b_global += reward * x
        self.t += 1

        # Track per-arm statistics (optional: useful for debugging)
        arm_key = title
        if arm_key not in self.arms:
            self.arms[arm_key] = ArmState(
                A=np.eye(self.d, dtype=np.float32),
                b=np.zeros(self.d, dtype=np.float32),
            )

        arm_state = self.arms[arm_key]
        arm_state.A += np.outer(x, x)
        arm_state.b += reward * x
        arm_state.count += 1

    def train(
        self,
        dataset: list[dict[str, Any]],
        embeddings: dict[str, Any],
    ) -> None:
        """Train on full dataset with pre-computed embeddings.

        Args:
            dataset: list of HotpotQA samples
            embeddings: dict with keys "questions" and "sections" (numpy dicts)
        """
        for example_idx, example in enumerate(dataset):
            question = example["question"]
            gold_titles = set(example["supporting_facts"]["title"])

            titles = example["context"]["title"]
            sentences_list = example["context"]["sentences"]

            # Get question embedding
            q_embedding = embeddings["questions"].get(example_idx)
            if q_embedding is None:
                raise KeyError(f"Question embedding missing for example {example_idx}")

            for rank, (title, sentences) in enumerate(zip(titles, sentences_list)):
                section_text = " ".join(sentences)
                reward = 1.0 if title in gold_titles else 0.0

                # Get section embedding
                s_embedding = embeddings["sections"].get((example_idx, rank))
                if s_embedding is None:
                    raise KeyError(
                        f"Section embedding missing for ({example_idx}, {rank})"
                    )

                self.update(
                    question,
                    title,
                    section_text,
                    reward,
                    q_embedding=q_embedding,
                    s_embedding=s_embedding,
                )

    def score(
        self,
        question: str,
        title: str,
        section_text: str,
        q_embedding: np.ndarray | None = None,
        s_embedding: np.ndarray | None = None,
    ) -> float:
        """Compute LinUCB score (exploitation + exploration).

        Score = w^T * x + alpha * sqrt(x^T * A^{-1} * x)

        Args:
            question: question text
            title: section title
            section_text: section body
            q_embedding: question embedding (shape: 384,)
            s_embedding: section embedding (shape: 384,)

        Returns:
            UCB score for this arm
        """
        x = self._extract_features(
            question,
            title,
            section_text,
            q_embedding=q_embedding,
            s_embedding=s_embedding,
        )

        # Exploit: estimated expected reward (ridge regression)
        try:
            w = np.linalg.solve(self.A_global, self.b_global)
        except np.linalg.LinAlgError:
            # Singular matrix (shouldn't happen if t > 0)
            w = np.zeros(self.d, dtype=np.float32)

        exploit = float(np.dot(w, x))

        # Explore: uncertainty bonus
        try:
            A_inv = np.linalg.inv(self.A_global)
        except np.linalg.LinAlgError:
            A_inv = np.eye(self.d, dtype=np.float32)  # fallback

        uncertainty = np.sqrt(np.dot(x, A_inv @ x))
        explore = self.alpha * float(uncertainty)

        return exploit + explore

    def select(
        self,
        question: str,
        titles: list[str],
        sentences_list: list[list[str]],
        k: int,
        q_embedding: np.ndarray | None = None,
        s_embeddings: dict[int, np.ndarray] | None = None,
    ) -> list[str]:
        """Select top-k sections via LinUCB scores.

        Args:
            question: question text
            titles: list of section titles
            sentences_list: list of sentence lists
            k: number of sections to select
            q_embedding: question embedding (shape: 384,)
            s_embeddings: dict mapping rank -> section embedding

        Returns:
            list of top-k selected titles (in descending score order)
        """
        scores = []
        for rank, (title, sentences) in enumerate(zip(titles, sentences_list)):
            section_text = " ".join(sentences)
            s_embedding = None
            if s_embeddings is not None:
                s_embedding = s_embeddings.get(rank)

            score = self.score(
                question,
                title,
                section_text,
                q_embedding=q_embedding,
                s_embedding=s_embedding,
            )
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
