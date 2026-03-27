"""Load and prepare HotpotQA dataset."""

import random
from datasets import load_dataset


def get_sample(n: int = 50, seed: int = 42):
    """
    Load n random samples from HotpotQA validation set.

    Args:
        n: Number of samples to load
        seed: Random seed for reproducibility

    Returns:
        List of sample dicts
    """
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    random.seed(seed)
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    return [ds[i] for i in indices]


def extract_paragraphs(sample: dict) -> list[str]:
    """
    Convert HotpotQA context format to list of paragraph strings.

    HotpotQA format:
        sample['context']['title']: list of document titles
        sample['context']['sentences']: list of lists (sentences per doc)

    Returns:
        List of "[Title] sentence1 sentence2 ..." strings
    """
    paragraphs = []

    titles = sample['context']['title']
    sentences_lists = sample['context']['sentences']

    for title, sentences in zip(titles, sentences_lists):
        text = ' '.join(sentences)
        paragraphs.append(f"[{title}] {text}")

    return paragraphs
