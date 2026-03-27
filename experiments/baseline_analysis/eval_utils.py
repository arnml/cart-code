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


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    """
    Estimate cost in USD for GPT-4o-mini.
    Pricing: $0.15 per 1M input, $0.60 per 1M output
    """
    return (input_tokens * 0.00015 + output_tokens * 0.0006) / 1000


def efficiency(f1: float, total_tokens: int) -> float:
    """
    Efficiency metric: F1 / log(1 + tokens)
    Captures quality per unit cost.
    """
    if total_tokens <= 0:
        return 0.0
    return f1 / math.log(1 + total_tokens)
