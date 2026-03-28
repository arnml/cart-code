"""Evaluation utilities: F1, exact match, cost calculation, CSV loading."""

import csv
import math
import re
import string
from collections import Counter
from pathlib import Path

# Per-1K-token pricing as (input_usd, output_usd); divide by 1000 to get per-token cost.
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5.4-mini": (0.00075, 0.0045),
    "gpt-4o-mini": (0.00015, 0.0006),
    "claude": (0.001, 0.005),  # Haiku / Claude family
}


def normalize_answer(s: str) -> str:
    """Normalize answer for evaluation (HotpotQA standard)."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def extract_answer(content: str) -> str:
    """Extract the final answer from a model response with an 'Answer:' label."""
    return content.split("Answer:")[-1].strip() if "Answer:" in content else content.strip()


def f1_score(prediction: str, ground_truth: str | list) -> float:
    """Calculate token-level F1 score (HotpotQA standard)."""
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0] if ground_truth else ""

    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str | list) -> int:
    """Calculate exact match (0 or 1) after normalization."""
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0] if ground_truth else ""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def cost_usd(input_tokens: int, output_tokens: int, model: str = "gpt-4o-mini") -> float:
    """
    Estimate cost in USD.

    Pricing (per 1M tokens):
    - gpt-4o-mini:  $0.15 input / $0.60 output
    - gpt-5.4-mini: $0.75 input / $4.50 output
    - claude/haiku: $1.00 input / $5.00 output
    """
    if model in ("gpt-5.4-mini-2026-03-17", "gpt-5.4-mini"):
        in_rate, out_rate = _MODEL_PRICING["gpt-5.4-mini"]
    elif "haiku" in model.lower() or "claude" in model.lower():
        in_rate, out_rate = _MODEL_PRICING["claude"]
    else:
        in_rate, out_rate = _MODEL_PRICING["gpt-4o-mini"]
    return (input_tokens * in_rate + output_tokens * out_rate) / 1000


def efficiency(f1: float, total_tokens: int) -> float:
    """F1 / log(1 + tokens) — quality per unit cost."""
    if total_tokens <= 0:
        return 0.0
    return f1 / math.log(1 + total_tokens)


def load_csv_safe(csv_path: Path) -> list[dict]:
    """Load a CSV file, falling back to latin-1 encoding if UTF-8 fails."""
    try:
        with open(csv_path, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except UnicodeDecodeError:
        with open(csv_path, encoding="latin-1") as f:
            return list(csv.DictReader(f))
