"""
Evaluation utilities following official HotpotQA v1 metrics.

Includes F1, EM, precision, recall, and efficiency metrics.
Reference: https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
"""

import math
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Optional


@dataclass
class EvalMetrics:
    """Container for evaluation results."""
    em: float  # Exact match
    f1: float  # F1 score
    precision: float  # Token-level precision
    recall: float  # Token-level recall
    token_efficiency: Optional[float] = None  # F1 / log(1 + input_tokens + output_tokens) — quality per compute
    cost_efficiency: Optional[float] = None  # F1 / cost_usd — quality per dollar


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
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
) -> EvalMetrics:
    """
    Evaluate a single prediction-ground_truth pair.

    Args:
        prediction: Generated answer
        ground_truth: Reference answer
        input_tokens: Total input tokens used (for efficiency metric)
        output_tokens: Total output tokens used (for efficiency metric)
        cost_usd: Total cost in USD (for cost efficiency metric)

    Returns:
        EvalMetrics dataclass with EM, F1, precision, recall, efficiency scores
    """
    # Exact Match
    em = float(exact_match_score(prediction, ground_truth))

    # F1, Precision, Recall
    f1, prec, recall = f1_score(prediction, ground_truth)

    # Token Efficiency: F1 / log(1 + total_tokens) — quality per compute unit
    total_tokens = input_tokens + output_tokens
    token_efficiency = None
    if total_tokens >= 0:
        token_efficiency = f1 / math.log(1 + total_tokens) if total_tokens > 0 else f1

    # Cost Efficiency: F1 / cost_usd — quality per dollar spent
    cost_efficiency = None
    if cost_usd > 0:
        cost_efficiency = f1 / cost_usd

    return EvalMetrics(
        em=em,
        f1=f1,
        precision=prec,
        recall=recall,
        token_efficiency=token_efficiency,
        cost_efficiency=cost_efficiency,
    )


def aggregate_metrics(
    predictions: list[str],
    ground_truths: list[str],
    input_tokens: Optional[list[int]] = None,
    output_tokens: Optional[list[int]] = None,
    costs_usd: Optional[list[float]] = None,
) -> dict:
    """
    Aggregate metrics across multiple predictions.

    Args:
        predictions: List of generated answers
        ground_truths: List of reference answers
        input_tokens: List of input token counts per sample (optional)
        output_tokens: List of output token counts per sample (optional)
        costs_usd: List of costs in USD per sample (optional)

    Returns:
        Dictionary with aggregated metrics:
            - em_mean, f1_mean, precision_mean, recall_mean
            - token_efficiency_mean (if tokens provided) — quality per compute
            - cost_efficiency_mean (if costs provided) — quality per dollar
            - Also includes _sum variants for each metric
    """
    assert len(predictions) == len(ground_truths), "Mismatched lengths"

    if input_tokens is None:
        input_tokens = [0] * len(predictions)
    if output_tokens is None:
        output_tokens = [0] * len(predictions)
    if costs_usd is None:
        costs_usd = [0.0] * len(predictions)

    metrics_list = [
        evaluate_sample(
            pred, gt, inp, out, cost
        )
        for pred, gt, inp, out, cost in zip(
            predictions, ground_truths, input_tokens, output_tokens, costs_usd
        )
    ]

    # Aggregate
    n = len(metrics_list)
    agg = {
        'count': n,
        'em_sum': sum(m.em for m in metrics_list),
        'f1_sum': sum(m.f1 for m in metrics_list),
        'precision_sum': sum(m.precision for m in metrics_list),
        'recall_sum': sum(m.recall for m in metrics_list),
        'em_mean': sum(m.em for m in metrics_list) / n,
        'f1_mean': sum(m.f1 for m in metrics_list) / n,
        'precision_mean': sum(m.precision for m in metrics_list) / n,
        'recall_mean': sum(m.recall for m in metrics_list) / n,
    }

    # Token Efficiency metrics (if available)
    if any(m.token_efficiency is not None for m in metrics_list):
        token_effs = [m.token_efficiency for m in metrics_list if m.token_efficiency is not None]
        if token_effs:
            agg['token_efficiency_mean'] = sum(token_effs) / len(token_effs)
            agg['token_efficiency_sum'] = sum(token_effs)

    # Cost efficiency (if available)
    if any(m.cost_efficiency is not None for m in metrics_list):
        cost_effs = [m.cost_efficiency for m in metrics_list if m.cost_efficiency is not None]
        if cost_effs:
            agg['cost_efficiency_mean'] = sum(cost_effs) / len(cost_effs)
            agg['cost_efficiency_sum'] = sum(cost_effs)

    return agg


def format_metrics(metrics: dict) -> str:
    """
    Format aggregated metrics for printing.

    Args:
        metrics: Dictionary from aggregate_metrics()

    Returns:
        Formatted string for display
    """
    lines = [f"Metrics (n={metrics.get('count', '?')}):"]

    if 'em_mean' in metrics:
        lines.append(f"  EM:        {metrics['em_mean']:.4f}")
    if 'f1_mean' in metrics:
        lines.append(f"  F1:        {metrics['f1_mean']:.4f}")
    if 'precision_mean' in metrics:
        lines.append(f"  Precision: {metrics['precision_mean']:.4f}")
    if 'recall_mean' in metrics:
        lines.append(f"  Recall:    {metrics['recall_mean']:.4f}")
    if 'token_efficiency_mean' in metrics:
        lines.append(f"  Token-Eff (F1/log(tok)): {metrics['token_efficiency_mean']:.4f}")
    if 'cost_efficiency_mean' in metrics:
        lines.append(f"  Cost-Eff (F1/$):         {metrics['cost_efficiency_mean']:.2f}")

    return '\n'.join(lines)


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
    # Example 1: Single sample evaluation
    pred = "Paris is the capital of France"
    gt = "Paris"

    metrics = evaluate_sample(
        prediction=pred,
        ground_truth=gt,
        input_tokens=500,
        output_tokens=50,
        cost_usd=cost_usd(500, 50, model="claude-haiku"),
    )

    print("Single Sample Evaluation:")
    print(f"  EM: {metrics.em}")
    print(f"  F1: {metrics.f1:.4f}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall: {metrics.recall:.4f}")
    print(f"  Token-Eff (F1/log(tok)): {metrics.token_efficiency:.4f}")
    print(f"  Cost-Eff (F1/$): {metrics.cost_efficiency:.2f}")
    print()

    # Example 2: Batch evaluation
    predictions = [
        "Paris is the capital of France",
        "London is the capital of England",
        "Berlin is the capital of Germany",
    ]
    ground_truths = [
        "Paris",
        "London",
        "Berlin is the capital",
    ]
    input_tokens_list = [500, 500, 500]
    output_tokens_list = [50, 50, 75]
    costs_list = [
        cost_usd(inp, out, model="claude-haiku")
        for inp, out in zip(input_tokens_list, output_tokens_list)
    ]

    agg = aggregate_metrics(
        predictions,
        ground_truths,
        input_tokens=input_tokens_list,
        output_tokens=output_tokens_list,
        costs_usd=costs_list,
    )

    print(format_metrics(agg))
