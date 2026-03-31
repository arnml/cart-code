"""Shared helpers for HotpotQA baseline evaluation.

Includes:
- flatten_context: HotpotQA context normalization
- build_always_think_prompt / build_retrieval_prompt: Prompt templates
- call_llm: Provider routing and model invocation
"""


def flatten_context(context: dict) -> list[str]:
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


def build_always_think_prompt(question: str) -> str:
    """Build the prompt for the no-retrieval baseline."""
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
    context_str = "\n\n".join([f"[{i+1}] {p}" for i, p in enumerate(paragraphs)])
    return f"""Answer the HotpotQA question using only the provided context.

Rules:
- Output only the final answer.
- If the answer is yes, no or noanswer, output exactly: yes or no or noanswer.
- Otherwise output a short span or entity name only.
- Do not include any explanation.
- Do not repeat the question.
- If multiple positions/titles are mentioned, output only the one that directly answers the question.

Context:
{context_str}

Question: {question}

Answer:"""


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
