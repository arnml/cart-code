"""Shared preprocessing helpers for HotpotQA experiments."""

from __future__ import annotations

from typing import Any


def flatten_context(context: dict[str, Any]) -> list[str]:
    """Convert HotpotQA context to a list of readable paragraphs."""
    titles = context.get("title", [])
    sentences_list = context.get("sentences", [])

    paragraphs: list[str] = []
    for title, sentences in zip(titles, sentences_list, strict=True):
        title = str(title).strip()
        sent_text = " ".join(s.strip() for s in sentences if s and s.strip())
        para_text = f"{title}. {sent_text}" if sent_text else title
        paragraphs.append(para_text)
    return paragraphs


def build_always_think_prompt(question: str) -> str:
    """Build the prompt for the no-retrieval fallback."""
    return f"""Answer the HotpotQA question.

Rules:
- Output only the final answer.
- If the answer is yes, no or noanswer, output exactly: yes or no or noanswer.
- Otherwise output a short span or entity name only.
- Do not include any explanation.
- Do not repeat the question.

Question: {question}

Answer:"""


def build_retrieval_prompt(question: str, paragraphs: list[str]) -> str:
    """Build the prompt for retrieval-augmented answering."""
    context_str = "\n\n".join([f"[{i + 1}] {p}" for i, p in enumerate(paragraphs)])
    return f"""Answer the HotpotQA question using only the provided context.

Rules:
- Output only the final answer.
- If the answer is yes, no or noanswer, output exactly: yes or no or noanswer.
- Otherwise output a short span or entity name only.
- Do not include any explanation.
- Do not repeat the question.
- If multiple positions/titles are mentioned, output only the one that
  directly answers the question.

Context:
{context_str}

Question: {question}

Answer:"""
