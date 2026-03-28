"""Evaluation utilities: F1, exact match, cost calculation."""

import re
import string
import math
from collections import Counter


def normalize_answer(s: str) -> str:
    """Normalize answer for evaluation (HotpotQA standard)."""
    s = s.lower()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(s.split())


def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Calculate token-level F1 score.
    Works with both string and list answers.
    """
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0] if ground_truth else ""

    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    if not pred_tokens or not gt_tokens:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> int:
    """Calculate exact match (0 or 1) after normalization."""
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0] if ground_truth else ""

    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def cost_usd(input_tokens: int, output_tokens: int, model: str = "gpt-4o-mini") -> float:
    """
    Estimate cost in USD for different models.

    Models:
    - gpt-4o-mini: $0.15 per 1M input, $0.60 per 1M output
    - gpt-5.4-mini: $0.75 per 1M input, $4.50 per 1M output
    - claude-haiku-4-5: $1.00 per 1M input, $5.00 per 1M output
    """
    if model == "gpt-5.4-mini-2026-03-17" or model == "gpt-5.4-mini":
        # GPT-5.4-mini pricing
        return (input_tokens * 0.00075 + output_tokens * 0.0045) / 1000
    elif "haiku" in model.lower() or "claude" in model.lower():
        # Claude Haiku 4.5 pricing: $1.00/$5.00 per 1M
        return (input_tokens * 0.001 + output_tokens * 0.005) / 1000
    else:
        # Default to GPT-4o-mini pricing
        return (input_tokens * 0.00015 + output_tokens * 0.0006) / 1000


def efficiency(f1: float, total_tokens: int) -> float:
    """
    Efficiency metric: F1 / log(1 + tokens)
    Captures quality per unit cost.
    """
    if total_tokens <= 0:
        return 0.0
    return f1 / math.log(1 + total_tokens)
