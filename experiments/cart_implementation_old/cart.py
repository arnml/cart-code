"""CART ablation variants: base, noise-gated, and full UCB-Cost policy."""

from .llm import generate, think
from .policy import UCBCostPolicy
from .retrieval import retrieve_and_filter, embed_and_rank, adaptive_k


def _result(method: str, answer: str, inp: int, out: int, docs: int, routed: str, **extra) -> dict:
    return {
        "method": method,
        "answer": answer,
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": inp + out,
        "llm_calls": 1,
        "docs_retrieved": docs,
        "routed_to": routed,
        **extra,
    }


def cart_base(question: str, paragraphs: list[str], model: str) -> dict:
    """Ablation: adaptive-k only. No noise gate, no UCB."""
    docs, scores = embed_and_rank(question, paragraphs)
    k = adaptive_k(scores)
    # cart_base uses exactly k documents as per Taguchi et al. 2025
    context = "\n\n".join(docs[:k]) or "No context."
    ans, inp, out = generate(question, context, model)
    return _result("cart_base", ans, inp, out, k, "retrieve")


def cart_noise(question: str, paragraphs: list[str], model: str) -> dict:
    """Ablation: adaptive-k + noise gate. No UCB."""
    docs, _ = retrieve_and_filter(question, paragraphs)
    if docs:
        ans, inp, out = generate(question, "\n\n".join(docs), model)
        routed = "retrieve"
    else:
        ans, inp, out = think(question, model)
        routed = "think_fallback"
    return _result("cart_noise", ans, inp, out, len(docs), routed)


def cart_full(
    question: str,
    paragraphs: list[str],
    model: str,
    lambda_cost: float = 1.0,
    policy: UCBCostPolicy | None = None,
) -> dict:
    """CART-full: adaptive-k + noise gate + stateful UCB-Cost policy.

    Implements Algorithm 1 from the paper: iterative decision loop.
    """
    if policy is None:
        policy = UCBCostPolicy(lambda_cost=lambda_cost)
    else:
        lambda_cost = policy.lambda_cost

    docs: list[str] = []
    scores: list[float] = []

    # Algorithm 1: Loop for up to 3 steps to find the best action
    for _ in range(3):
        action = policy.select()

        if action == "think":
            # For think, we don't have a similarity-based reward proxy at this stage.
            # The actual reward will be updated by the caller using F1 to allow adaptation.
            ans, inp, out = think(question, model)
            return _result("cart_full", ans, inp, out, 0, "think", lambda_cost=lambda_cost)

        if action in ("retrieve", "tool") or not docs:
            docs, scores = retrieve_and_filter(question, paragraphs)
            # If we found documents, we proceed to generation.
            # The policy update for retrieval happens here with a similarity proxy.
            reward = sum(scores) / max(len(scores), 1) if scores else 0.0
            policy.update(action, reward)
            if docs:
                break

    if docs:
        ans, inp, out = generate(question, "\n\n".join(docs), model)
        routed = "retrieve"
    else:
        ans, inp, out = think(question, model)
        routed = "think_fallback"

    return _result("cart_full", ans, inp, out, len(docs), routed, lambda_cost=lambda_cost)
