"""Pre-compute and cache question/section embeddings using sentence-transformers.

Generates embeddings for all questions and sections in train/validation splits,
then caches them as numpy archives (.npz) for fast inference-time lookup.

Usage from root:
    uv run python -m experiments.precompute_embeddings train experiments/cache/train_embeddings.npz
    uv run python -m experiments.precompute_embeddings validation experiments/cache/val_embeddings.npz

For testing on small sample:
    uv run python -m experiments.precompute_embeddings train experiments/cache/test_embeddings.npz 100
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from experiments.baselines_config import DATASET_CONFIG
from experiments.cache_dataset import load_dataset_cached

# Embedding model: optimized for question-to-passage retrieval
EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"
BATCH_SIZE = 512


def precompute_embeddings(
    split: str,
    output_path: str,
    n_rows: int | None = None,
) -> None:
    """Pre-compute embeddings for all questions and sections.

    Args:
        split: Dataset split ("train" or "validation")
        output_path: Path to save embeddings .npz file
        n_rows: Maximum number of examples to process (None = all)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    print("\n" + "=" * 70)
    print(f"Pre-computing Embeddings: {split.upper()} split")
    print("=" * 70)

    # Load dataset
    print(f"\nLoading {split} dataset...")
    dataset = load_dataset_cached(
        dataset_name=DATASET_CONFIG["dataset_name"],
        subset=DATASET_CONFIG["subset"],
        split=split,
        n_rows=n_rows,
    )
    print(f"Loaded {len(dataset)} samples")

    # Load embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {embedding_dim}")

    # Collect all unique texts
    print("\nExtracting questions and sections...")
    questions: dict[int, str] = {}  # example_idx -> question text
    sections: dict[tuple[int, int], str] = {}  # (example_idx, section_idx) -> section text

    for i, example in enumerate(dataset):
        questions[i] = example["question"]

        titles = example["context"]["title"]
        sentences_list = example["context"]["sentences"]

        for rank, (title, sentences) in enumerate(zip(titles, sentences_list)):
            # Section text = title + body
            section_text = title + " " + " ".join(sentences)
            sections[(i, rank)] = section_text

        if (i + 1) % 10000 == 0:
            print(f"  Extracted {i + 1}/{len(dataset)} samples...")

    print(f"Total questions: {len(questions)}")
    print(f"Total sections: {len(sections)}")

    # Batch encode questions
    print(f"\nEncoding {len(questions)} questions...")
    q_ids = sorted(questions.keys())
    q_texts = [questions[i] for i in q_ids]
    q_embeddings = model.encode(
        q_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    q_emb_map = {q_ids[i]: q_embeddings[i] for i in range(len(q_ids))}
    print(f"Encoded questions shape: {q_embeddings.shape}")

    # Batch encode sections
    print(f"\nEncoding {len(sections)} sections...")
    s_ids = sorted(sections.keys())
    s_texts = [sections[k] for k in s_ids]
    s_embeddings = model.encode(
        s_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    s_emb_map = {s_ids[i]: s_embeddings[i] for i in range(len(s_ids))}
    print(f"Encoded sections shape: {s_embeddings.shape}")

    # Save to .npz
    print(f"\nSaving embeddings to {output_path}...")
    result = {
        "questions": q_emb_map,
        "sections": s_emb_map,
        "metadata": {
            "split": split,
            "n_samples": len(dataset),
            "n_questions": len(q_emb_map),
            "n_sections": len(s_emb_map),
            "embedding_dim": embedding_dim,
            "model": EMBEDDING_MODEL,
        },
    }

    np.savez_compressed(output_path, **result)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved to {output_path} ({file_size_mb:.1f} MB)")

    print("\n" + "=" * 70)
    print("Pre-computation complete!")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: uv run python -m experiments.precompute_embeddings "
            "<split> <output_path> [n_rows]"
        )
        print("\nExamples:")
        print("  uv run python -m experiments.precompute_embeddings train cache/train_embeddings.npz")
        print("  uv run python -m experiments.precompute_embeddings validation cache/val_embeddings.npz")
        print("  uv run python -m experiments.precompute_embeddings train cache/test_embeddings.npz 100")
        sys.exit(1)

    split = sys.argv[1]
    output_path = sys.argv[2]
    n_rows = int(sys.argv[3]) if len(sys.argv) >= 4 else None

    if split not in ("train", "validation"):
        print(f"Error: split must be 'train' or 'validation', got '{split}'")
        sys.exit(1)

    precompute_embeddings(split, output_path, n_rows)
