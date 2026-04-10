"""Diagnostic: analyze feature signal in HotpotQA training data.

Confirms that question-section features (token overlap, title match, NER)
have predictive signal for distinguishing gold from non-gold sections.

Usage:
    uv run python -m experiments.diagnostic_linucb_features
"""

import re
from collections import defaultdict
from experiments.cache_dataset import load_dataset_cached
from experiments.baselines_config import DATASET_CONFIG


def tokenize(text: str) -> set[str]:
    """Lowercase, split on word boundaries."""
    return set(re.findall(r'\w+', text.lower()))


def analyze_feature_signal(train_data: list[dict], n: int = 5000) -> None:
    """
    For each section in each example, compute features and label (gold/not).
    Report mean feature value for gold vs non-gold sections.
    """
    results = defaultdict(lambda: {"gold": [], "non_gold": []})

    print(f"Analyzing {min(n, len(train_data))} training examples...")

    for example in train_data[:n]:
        question_tokens = tokenize(example["question"])
        gold_titles = set(example["supporting_facts"]["title"])

        # HotpotQA context format: dict with "title" list and "sentences" list
        titles = example["context"]["title"]
        sentences_list = example["context"]["sentences"]

        for title, sentences in zip(titles, sentences_list):
            is_gold = title in gold_titles
            section_text = " ".join(sentences)
            section_tokens = tokenize(section_text)
            title_tokens = tokenize(title)

            features = {
                "token_overlap": len(question_tokens & section_tokens) / (len(question_tokens) + 1),
                "title_in_question": len(question_tokens & title_tokens) / (len(title_tokens) + 1),
                "section_length": len(sentences),
                "title_length": len(title_tokens),
            }

            bucket = "gold" if is_gold else "non_gold"
            for fname, fval in features.items():
                results[fname][bucket].append(fval)

    # Print results
    print("\n" + "=" * 70)
    print(f"{'Feature':<25} {'Gold mean':>12} {'Non-gold mean':>16} {'Ratio':>10}")
    print("=" * 70)

    for fname in sorted(results.keys()):
        buckets = results[fname]
        if not buckets["gold"] or not buckets["non_gold"]:
            continue

        gold_mean = sum(buckets["gold"]) / len(buckets["gold"])
        non_gold_mean = sum(buckets["non_gold"]) / len(buckets["non_gold"])
        ratio = gold_mean / (non_gold_mean + 1e-9)

        status = "[SIGNAL]" if ratio >= 1.5 else "[WEAK]" if ratio >= 1.2 else "[NONE]"
        print(
            f"{fname:<25} {gold_mean:>12.4f} {non_gold_mean:>16.4f} "
            f"{ratio:>9.2f}x  {status}"
        )

    print("=" * 70)
    print("\nInterpretation:")
    print("  [SIGNAL]:  Ratio >= 1.5x  -> feature is predictive, include in LinUCB")
    print("  [WEAK]:    Ratio 1.2-1.5x -> limited signal, optional")
    print("  [NONE]:    Ratio < 1.2x   -> no signal, skip")
    print()


if __name__ == "__main__":
    print("Loading HotpotQA train split...")
    train = load_dataset_cached(
        dataset_name=DATASET_CONFIG["dataset_name"],
        subset=DATASET_CONFIG["subset"],
        split="train",
        n_rows=5000,
    )
    print(f"Loaded {len(train)} training samples\n")

    analyze_feature_signal(train, n=5000)
