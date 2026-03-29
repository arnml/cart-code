"""CART ablation variants: base, noise-gated, and full UCB-Cost policy."""

from .llm import generate, think
from .policy import UCBCostPolicy
from .retrieval import adaptive_k, embed_and_rank, retrieve_and_filter


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
) -> dict:
    """CART-full: adaptive-k + noise gate + UCB-Cost policy. Main method."""
    policy = UCBCostPolicy(lambda_cost=lambda_cost)
    # Retrieve once — inputs are fixed, repeated calls would be pure waste.
    docs, scores = retrieve_and_filter(question, paragraphs)
    reward = sum(scores) / max(len(scores), 1) if scores else 0.0

    for _ in range(3):
        action = policy.select()
        if action == "think" and docs:
            policy.update("think", 0.4)
            ans, inp, out = think(question, model)
            return _result("cart_full", ans, inp, out, 0, "think", lambda_cost=lambda_cost)
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
