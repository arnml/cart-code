"""Helpers for MuSiQue-Ans experiments.

The dataset already exposes paragraph-level context, so this module focuses on
normalizing that structure into the same string-list interface used by the
other experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

DATASET_NAME = "dgslibisey/MuSiQue"
CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "musique_ans"


def get_paragraphs(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the paragraph list from a MuSiQue-Ans sample."""
    paragraphs = sample.get("paragraphs", [])
    return list(paragraphs)


def flatten_context(paragraphs: list[dict[str, Any]]) -> list[str]:
    """Convert paragraph dicts to readable strings."""
    flattened: list[str] = []
    for paragraph in paragraphs:
        title = str(paragraph.get("title", "")).strip()
        paragraph_text = str(paragraph.get("paragraph_text", "")).strip()
        flattened.append(f"{title}. {paragraph_text}" if title else paragraph_text)
    return flattened


def get_question(sample: dict[str, Any]) -> str:
    return str(sample.get("question", "")).strip()


def get_answer(sample: dict[str, Any]) -> str:
    return str(sample.get("answer", "")).strip()


def get_answer_aliases(sample: dict[str, Any]) -> list[str]:
    aliases = sample.get("answer_aliases", [])
    return [str(alias).strip() for alias in aliases]


def get_support_titles(sample: dict[str, Any]) -> list[str]:
    return [
        str(paragraph.get("title", "")).strip()
        for paragraph in get_paragraphs(sample)
        if paragraph.get("is_supporting")
    ]


def build_always_think_prompt(question: str) -> str:
    """Prompt for the no-context fallback."""
    return f"""Answer the multi-hop question.

Rules:
- Output only the final answer.
- If the answer is yes, no or noanswer, output exactly: yes or no or noanswer.
- Otherwise output a short span or entity name only.
- Do not include any explanation.
- Do not repeat the question.

Question: {question}

Answer:"""


def build_retrieval_prompt(question: str, paragraphs: list[str]) -> str:
    """Prompt for retrieval-augmented answering."""
    context_str = "\n\n".join([f"[{i + 1}] {p}" for i, p in enumerate(paragraphs)])
    return f"""Answer the multi-hop question using only the provided context.

Rules:
- Output only the final answer.
- If the answer is yes, no or noanswer, output exactly: yes or no or noanswer.
- Otherwise output a short span or entity name only.
- Do not include any explanation.
- Do not repeat the question.

Context:
{context_str}

Question: {question}

Answer:"""
