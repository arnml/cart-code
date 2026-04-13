"""Helpers for 2WikiMultiHopQA experiments.

The goal is to mirror the HotpotQA experiment interface while keeping the
dataset-specific parsing in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

DATASET_NAME = "framolfese/2WikiMultihopQA"
CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "2wikimultihopqa"


def get_raw_context(sample: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical 2Wiki context payload."""
    metadata = sample.get("metadata", {})
    context = metadata.get("context", {})
    if context:
        return context
    return sample.get("context", {})


def flatten_context(context: dict[str, Any]) -> list[str]:
    """Convert the 2Wiki context to readable paragraph strings."""
    titles = context.get("title", [])
    sentences_list = context.get("sentences", context.get("content", []))

    paragraphs: list[str] = []
    for title, paragraph_lines in zip(titles, sentences_list):
        title_text = str(title).strip()
        if isinstance(paragraph_lines, list):
            sentences = (
                str(sentence).strip()
                for sentence in paragraph_lines
                if sentence and str(sentence).strip()
            )
            sent_text = " ".join(sentences)
        else:
            sent_text = str(paragraph_lines).strip()
        paragraphs.append(f"{title_text}. {sent_text}" if sent_text else title_text)
    return paragraphs


def get_question(sample: dict[str, Any]) -> str:
    return str(sample.get("question", "")).strip()


def get_answer(sample: dict[str, Any]) -> str:
    answers = sample.get("golden_answers", [])
    if answers:
        return str(answers[0]).strip()
    return str(sample.get("answer", "")).strip()


def get_support_titles(sample: dict[str, Any]) -> list[str]:
    metadata = sample.get("metadata", {})
    supporting_facts = metadata.get("supporting_facts", {})
    return [str(title).strip() for title in supporting_facts.get("title", [])]


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
