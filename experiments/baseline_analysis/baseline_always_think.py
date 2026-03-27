"""Baseline 2: Always-Think (chain-of-thought, no retrieval)."""

from openai import OpenAI

client = OpenAI()


def always_think(question: str) -> dict:
    """
    Pure reasoning baseline: answer using chain-of-thought without retrieval.

    Args:
        question: The question to answer

    Returns:
        Dict with method, answer, token counts, etc.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer questions using step-by-step reasoning."
        },
        {
            "role": "user",
            "content": f"""Question: {question}

Think step by step, then provide a concise answer.

Format your response as:
Reasoning: [your reasoning steps]
Answer: [the final answer]"""
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
        max_tokens=300
    )

    content = response.choices[0].message.content
    # Extract answer after "Answer:" label
    answer = content.split("Answer:")[-1].strip() if "Answer:" in content else content.strip()

    return {
        "method": "always_think",
        "answer": answer,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "llm_calls": 1
    }
