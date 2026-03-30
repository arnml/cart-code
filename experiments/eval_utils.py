"""
Evaluation utilities following official HotpotQA v1 metrics.

Includes F1, EM, precision, recall, and efficiency metrics.
Reference: https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
"""

import re
import string
from collections import Counter
from dataclasses import dataclass


@dataclass
class EvalMetrics:
    """Container for evaluation results."""
    em: float  # Exact match
    f1: float  # F1 score
    precision: float  # Token-level precision
    recall: float  # Token-level recall


def normalize_answer(s: str) -> str:
    """
    Normalize answer for evaluation (HotpotQA standard).

    Following official HotpotQA v1 evaluation script exactly.
    """
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """
    Compute token-level F1, precision, recall (HotpotQA standard).

    Returns:
        (f1, precision, recall) - Tuple of three floats [0, 1]
        Returns (0, 0, 0) if yes/no/noanswer mismatch (strict evaluation)

    Special handling:
        - If ground_truth is "yes"/"no"/"noanswer", prediction must match exactly
        - Handles empty predictions gracefully
    """
    normalized_pred = normalize_answer(prediction)
    normalized_gt = normalize_answer(ground_truth)

    # HotpotQA special rule: yes/no/noanswer must match exactly
    if normalized_gt in ['yes', 'no', 'noanswer']:
        if normalized_pred != normalized_gt:
            return 0.0, 0.0, 0.0

    # Tokenize
    pred_tokens = normalized_pred.split()
    gt_tokens = normalized_gt.split()

    # Handle empty predictions
    if not pred_tokens or not gt_tokens:
        return 0.0, 0.0, 0.0

    # Compute overlap using multiset (Counter)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0, 0.0, 0.0

    # Precision & Recall
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)

    # F1
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """
    Compute exact match (0 or 1) after normalization.

    Returns:
        True if normalized prediction == normalized ground_truth, False otherwise
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def evaluate_sample(
    prediction: str,
    ground_truth: str,
) -> EvalMetrics:
    """
    Evaluate a single prediction-ground_truth pair.

    Args:
        prediction: Generated answer
        ground_truth: Reference answer

    Returns:
        EvalMetrics dataclass with EM, F1, precision, recall
    """
    # Exact Match
    em = float(exact_match_score(prediction, ground_truth))

    # F1, Precision, Recall
    f1, prec, recall = f1_score(prediction, ground_truth)

    return EvalMetrics(
        em=em,
        f1=f1,
        precision=prec,
        recall=recall,
    )


# ============================================================================
# Price-aware cost computation (for use in evaluation)
# ============================================================================

_MODEL_PRICING = {
    "gpt-5.4-mini": (0.00075, 0.0045),
    "gpt-4o-mini": (0.00015, 0.0006),
    "claude-haiku": (0.001, 0.005),
    "claude-sonnet": (0.003, 0.015),
    "claude-opus": (0.015, 0.045),
}

def cost_usd(
    input_tokens: int,
    output_tokens: int,
    model: str = "claude-haiku",
) -> float:
    """
    Estimate LLM generation cost in USD based on 2026 pricing tiers.

    Note: Only includes LLM API costs (input + output tokens).
    Embedding costs (if using retrieval) are cached and calculated separately.

    Args:
        input_tokens: Number of input tokens to the LLM
        output_tokens: Number of output tokens from the LLM
        model: Model name (claude-haiku, claude-sonnet, gpt-4o-mini, etc.)

    Returns:
        Total LLM cost in USD
    """
    model_lower = model.lower()

    # Find matching pricing
    in_rate, out_rate = _MODEL_PRICING.get("claude-haiku", (0.001, 0.005))

    for model_key in _MODEL_PRICING:
        if model_key in model_lower:
            in_rate, out_rate = _MODEL_PRICING[model_key]
            break

    return (input_tokens * in_rate + output_tokens * out_rate) / 1000


# ============================================================================
# Example usage in paper/experiments
# ============================================================================

if __name__ == "__main__":
    # Example: Single sample evaluation
    pred = "Paris is the capital of France"
    gt = "Paris"

    metrics = evaluate_sample(
        prediction=pred,
        ground_truth=gt,
    )

    print("Single Sample Evaluation:")
    print(f"  EM: {metrics.em}")
    print(f"  F1: {metrics.f1:.4f}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall: {metrics.recall:.4f}")
