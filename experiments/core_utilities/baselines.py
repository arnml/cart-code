"""Baseline method implementations for HotpotQA evaluation.

Includes:
- always_think: No retrieval, pure reasoning
- retrieval_k3/k5/k10: Dense retrieval by embedding similarity
"""

from experiments.core_utilities.embedding_utils import retrieve_top_k


def _flatten_context(context: list) -> list[str]:
    """Convert HotpotQA context format to list of strings.

    Combines title and sentences into natural-reading paragraphs.
    Handles empty sentences, messy spacing, and edge cases robustly.

    Args:
        context: List of [title, sentences] pairs from HotpotQA

    Returns:
        List of paragraph strings (title. sentences joined)
    """
    paragraphs = []
    for title, sentences in context:
        title = str(title).strip()
        # Filter and clean sentences
        sent_text = " ".join(s.strip() for s in sentences if s and s.strip())
        # Combine with period for natural reading in embeddings
        para_text = f"{title}. {sent_text}" if sent_text else title
        paragraphs.append(para_text)
    return paragraphs


def always_think(sample: dict, model: str) -> str:
    """No retrieval — just ask the LLM the question.

    This is a baseline of pure reasoning without access to supporting documents.

    Args:
        sample: HotpotQA sample with "question" key
        model: Model name (passed for API routing, not used here)

    Returns:
        LLM-generated answer
    """
    question = sample["question"]
    prompt = f"Answer the question: {question}"

    # Placeholder: implement by calling LLM client with prompt
    answer = call_llm(prompt, model)
    return answer


def retrieval_k3(sample: dict, model: str) -> str:
    """Retrieve top-3 paragraphs by embedding similarity, then ask LLM.

    Args:
        sample: HotpotQA sample
        model: Model name

    Returns:
        LLM-generated answer based on retrieved context
    """
    return _retrieval_kn(sample, model, k=3)


def retrieval_k5(sample: dict, model: str) -> str:
    """Retrieve top-5 paragraphs by embedding similarity, then ask LLM.

    Args:
        sample: HotpotQA sample
        model: Model name

    Returns:
        LLM-generated answer based on retrieved context
    """
    return _retrieval_kn(sample, model, k=5)


def retrieval_k10(sample: dict, model: str) -> str:
    """Retrieve top-10 paragraphs by embedding similarity, then ask LLM.

    Args:
        sample: HotpotQA sample
        model: Model name

    Returns:
        LLM-generated answer based on retrieved context
    """
    return _retrieval_kn(sample, model, k=10)


def _retrieval_kn(sample: dict, model: str, k: int) -> str:
    """Generic retrieval-based method.

    Retrieves top-k paragraphs by embedding similarity to the question,
    then asks the LLM to answer based on those paragraphs.

    Note: supporting_facts from the sample are ignored here and only used
    during evaluation for reference.

    Args:
        sample: HotpotQA sample with "question" and "context" keys
        model: LLM model name
        k: Number of paragraphs to retrieve

    Returns:
        LLM-generated answer based on retrieved context

    Raises:
        KeyError: If LLM model not in EMBEDDING_CONFIG
    """
    question = sample["question"]
    context = sample["context"]

    from experiments.core_utilities.baselines_config import LLM_TO_EMBEDDING

    # Flatten context to list of paragraph strings
    # Note: We use titles + paragraph text for retrieval, not supporting_facts
    paragraphs = _flatten_context(context)

    # Get embedding config for this LLM model (includes provider, model, max_tokens)
    emb_config = LLM_TO_EMBEDDING[model]

    # Retrieve top-k paragraphs by embedding similarity to the question
    top_paragraphs, _ = retrieve_top_k(
        question=question,
        paragraphs=paragraphs,
        k=k,
        provider=emb_config["provider"],
        embedding_model=emb_config["embedding_model"],
        token_budget=emb_config["max_tokens"],
    )

    # Build prompt with retrieved context
    context_str = "\n\n".join(
        [f"[{i+1}] {p}" for i, p in enumerate(top_paragraphs)]
    )
    prompt = f"""Using the context below, answer the question:

Context:
{context_str}

Question: {question}

Answer:"""

    # Call LLM with context-augmented prompt
    answer = call_llm(prompt, model)
    return answer


def get_method(method_name: str):
    """Return the method function by name.

    Args:
        method_name: One of "always_think", "retrieval_k3", "retrieval_k5", "retrieval_k10"

    Returns:
        Callable method(sample, model) -> str

    Raises:
        ValueError: If method_name is not recognized
    """
    methods = {
        "always_think": always_think,
        "retrieval_k3": retrieval_k3,
        "retrieval_k5": retrieval_k5,
        "retrieval_k10": retrieval_k10,
    }

    if method_name not in methods:
        raise ValueError(
            f"Unknown method: {method_name}. Available: {list(methods.keys())}"
        )

    return methods[method_name]


def call_llm(prompt: str, model: str) -> str:
    """Call LLM and return generated answer.

    Placeholder: Implement with actual API calls (OpenAI, Anthropic, etc.)
    Must track:
        - input_tokens: Total tokens in prompt
        - output_tokens: Total tokens in response
        - cost_usd: Total cost for this call

    Args:
        prompt: Full prompt to send to LLM
        model: Model name (e.g., "gpt-4o-mini", "claude-sonnet-4-6")

    Returns:
        Generated answer string

    Raises:
        NotImplementedError: Until you implement with actual API client
    """
    raise NotImplementedError(
        "call_llm() must be implemented with actual API calls (OpenAI SDK, Anthropic SDK, etc.)"
    )
