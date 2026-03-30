"""Baseline method implementations for HotpotQA evaluation.

Includes:
- always_think: No retrieval, pure reasoning
- retrieval_k3/k5/k10: Dense retrieval by embedding similarity
"""

from experiments.embedding_utils import retrieve_top_k


def _flatten_context(context: dict) -> list[str]:
    """Convert HotpotQA context format to list of strings.

    Combines title and sentences into natural-reading paragraphs.
    Handles empty sentences, messy spacing, and edge cases robustly.

    Args:
        context: Dict with "title" and "sentences" keys from HotpotQA
                 Both are lists of equal length (one paragraph per index)

    Returns:
        List of paragraph strings (title. sentences joined)
    """
    titles = context.get("title", [])
    sentences_list = context.get("sentences", [])

    paragraphs = []
    for title, sentences in zip(titles, sentences_list):
        title = str(title).strip()
        # Filter and clean sentences
        sent_text = " ".join(s.strip() for s in sentences if s and s.strip())
        # Combine with period for natural reading in embeddings
        para_text = f"{title}. {sent_text}" if sent_text else title
        paragraphs.append(para_text)
    return paragraphs


def always_think(sample: dict, model: str) -> tuple[str, int, int, float]:
    """No retrieval — just ask the LLM the question.

    This is a baseline of pure reasoning without access to supporting documents.

    Args:
        sample: HotpotQA sample with "question" key
        model: Model name (passed for API routing)

    Returns:
        Tuple of (answer, input_tokens, output_tokens, cost_usd)
    """
    question = sample["question"]
    prompt = f"""Answer the HotpotQA question.

Rules:
- Output only the final answer.
- If the answer is yes or no, output exactly: yes or no.
- Otherwise output a short span or entity name only.
- Do not include any explanation.
- Do not repeat the question.

Question: {question}

Answer:"""

    # Call LLM and get answer + tokens + cost
    answer, input_tokens, output_tokens, cost_usd = call_llm(prompt, model)
    return answer, input_tokens, output_tokens, cost_usd


def retrieval_k3(sample: dict, model: str) -> tuple[str, int, int, float]:
    """Retrieve top-3 paragraphs by embedding similarity, then ask LLM.

    Args:
        sample: HotpotQA sample
        model: Model name

    Returns:
        Tuple of (answer, input_tokens, output_tokens, cost_usd)
    """
    return _retrieval_kn(sample, model, k=3)


def retrieval_k5(sample: dict, model: str) -> tuple[str, int, int, float]:
    """Retrieve top-5 paragraphs by embedding similarity, then ask LLM.

    Args:
        sample: HotpotQA sample
        model: Model name

    Returns:
        Tuple of (answer, input_tokens, output_tokens, cost_usd)
    """
    return _retrieval_kn(sample, model, k=5)


def retrieval_k10(sample: dict, model: str) -> tuple[str, int, int, float]:
    """Retrieve top-10 paragraphs by embedding similarity, then ask LLM.

    Args:
        sample: HotpotQA sample
        model: Model name

    Returns:
        Tuple of (answer, input_tokens, output_tokens, cost_usd)
    """
    return _retrieval_kn(sample, model, k=10)


def _retrieval_kn(sample: dict, model: str, k: int) -> tuple[str, int, int, float]:
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
        Tuple of (answer, input_tokens, output_tokens, cost_usd)

    Raises:
        KeyError: If LLM model not in EMBEDDING_CONFIG
    """
    question = sample["question"]
    context = sample["context"]

    from experiments.baselines_config import LLM_TO_EMBEDDING

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
    prompt = f"""Answer the HotpotQA question using only the provided context.

Rules:
- Output only the final answer.
- If the answer is yes or no, output exactly: yes or no.
- Otherwise output a short span or entity name only.
- Do not include any explanation.
- Do not repeat the question.
- If multiple positions/titles are mentioned, output only the one that directly answers the question.

Context:
{context_str}

Question: {question}

Answer:"""

    # Call LLM with context-augmented prompt and get tokens + cost
    answer, input_tokens, output_tokens, cost_usd = call_llm(prompt, model)
    return answer, input_tokens, output_tokens, cost_usd


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


def call_llm(prompt: str, model: str) -> tuple[str, int, int, float]:
    """Call LLM and return answer + token counts + cost.

    Routes to OpenAI or Anthropic based on model configuration.

    Args:
        prompt: Full prompt to send to LLM
        model: Model name (e.g., "gpt-4o-mini", "claude-sonnet-4-6")

    Returns:
        Tuple of (answer, input_tokens, output_tokens, cost_usd)

    Raises:
        KeyError: If model not recognized
        Exception: If API call fails (auth, rate limit, etc.)
    """
    from experiments.baselines_config import LLM_CONFIG

    if model not in LLM_CONFIG:
        raise KeyError(f"Unknown model: {model}")

    config = LLM_CONFIG[model]
    provider = config["provider"]

    if provider == "openai":
        from experiments.llm_openai import call_openai
        result = call_openai(prompt, model)
    elif provider == "anthropic":
        from experiments.llm_anthropic import call_anthropic
        result = call_anthropic(prompt, model)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return (
        result["answer"],
        result["input_tokens"],
        result["output_tokens"],
        result["cost_usd"],
    )
