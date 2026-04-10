"""Train LinUCB-Semantic (d=4) reranker on HotpotQA training split with embeddings.

Usage:
    uv run python -m experiments.train_linucb_semantic [--embeddings-path <path>] [--output-path <path>]

Examples:
    uv run python -m experiments.train_linucb_semantic
    uv run python -m experiments.train_linucb_semantic --embeddings-path cache/train_embeddings.npz --output-path cache/linucb_semantic_model.json
"""

import sys
from pathlib import Path

import numpy as np

from experiments.baselines_config import DATASET_CONFIG
from experiments.cache_dataset import load_dataset_cached
from experiments.linucb_semantic import LinUCBSemantic


def train_linucb_semantic(
    embeddings_path: str = "experiments/cache/train_embeddings.npz",
    output_path: str = "experiments/cache/linucb_semantic_model.json",
) -> None:
    """Train LinUCB-Semantic on full HotpotQA training split.

    Args:
        embeddings_path: path to pre-computed train embeddings (NPZ file)
        output_path: where to save the trained model
    """
    print("\n" + "=" * 70)
    print("LinUCB-Semantic (d=4) Reranker Training")
    print("=" * 70)

    # Load embeddings
    embeddings_path = Path(embeddings_path)
    print(f"\nLoading embeddings from {embeddings_path}...")
    if not embeddings_path.exists():
        print(f"Error: Embeddings file not found: {embeddings_path}")
        print("Please run: uv run python -m experiments.precompute_embeddings train <output_path>")
        sys.exit(1)

    embeddings_archive = np.load(embeddings_path, allow_pickle=True)
    embeddings = {key: embeddings_archive[key].item() for key in embeddings_archive.files}

    print(f"Loaded embeddings archive with keys: {list(embeddings.keys())}")
    if "metadata" in embeddings:
        metadata = embeddings.pop("metadata")
        print(f"  Split: {metadata['split']}")
        print(f"  Questions: {metadata['n_questions']}")
        print(f"  Sections: {metadata['n_sections']}")
        print(f"  Embedding dimension: {metadata['embedding_dim']}")

    # Load training dataset
    print("\nLoading HotpotQA training split...")
    train = load_dataset_cached(
        dataset_name=DATASET_CONFIG["dataset_name"],
        subset=DATASET_CONFIG["subset"],
        split="train",
        n_rows=None,  # use all
    )
    print(f"Loaded {len(train)} training samples")

    # Validate embeddings match dataset
    if len(embeddings["questions"]) != len(train):
        print(
            f"Error: Embedding count ({len(embeddings['questions'])}) "
            f"does not match dataset size ({len(train)})"
        )
        sys.exit(1)

    # Initialize reranker
    print("\nInitializing LinUCB-Semantic reranker (d=4, alpha=0.5)...")
    reranker = LinUCBSemantic(d=4, alpha=0.5)

    # Train
    print("Training on dataset...")
    reranker.train(train, embeddings)

    print(f"\nTraining complete:")
    print(f"  Total observations: {reranker.t:,}")
    print(f"  Unique sections (arms): {len(reranker.arms):,}")
    print(f"  Global design matrix A shape: {reranker.A_global.shape}")
    print(f"  Global response vector b shape: {reranker.b_global.shape}")

    # Save model
    output_path = Path(output_path)
    print(f"\nSaving model to {output_path}...")
    reranker.save(str(output_path))
    print(f"Model saved: {output_path.stat().st_size:,} bytes")

    print("=" * 70)


if __name__ == "__main__":
    embeddings_path = "experiments/cache/train_embeddings.npz"
    output_path = "experiments/cache/linucb_semantic_model.json"

    # Parse arguments
    if len(sys.argv) > 1:
        i = 1
        while i < len(sys.argv):
            if sys.argv[i] == "--embeddings-path" and i + 1 < len(sys.argv):
                embeddings_path = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--output-path" and i + 1 < len(sys.argv):
                output_path = sys.argv[i + 1]
                i += 2
            else:
                i += 1

    train_linucb_semantic(embeddings_path, output_path)
