"""Train LinUCB reranker on HotpotQA training split.

Usage:
    uv run python -m experiments.train_linucb [--output-path <path>]
"""

import sys
from pathlib import Path

from experiments.baselines_config import DATASET_CONFIG
from experiments.cache_dataset import load_dataset_cached
from experiments.linucb import LinUCBReranker


def train_linucb(output_path: str = "experiments/cache/linucb_model.json") -> None:
    """Train LinUCB on full HotpotQA training split.

    Args:
        output_path: where to save the trained model
    """
    print("\n" + "=" * 70)
    print("LinUCB Reranker Training")
    print("=" * 70)

    # Load training dataset
    print("\nLoading HotpotQA training split...")
    train = load_dataset_cached(
        dataset_name=DATASET_CONFIG["dataset_name"],
        subset=DATASET_CONFIG["subset"],
        split="train",
        n_rows=None,  # use all
    )
    print(f"Loaded {len(train)} training samples")

    # Initialize reranker
    print("\nInitializing LinUCB reranker (d=3, alpha=1.0)...")
    reranker = LinUCBReranker(d=3, alpha=1.0)

    # Train
    print("Training on dataset...")
    reranker.train(train)

    print(f"\nTraining complete:")
    print(f"  Total observations: {reranker.t:,}")
    print(f"  Unique sections (arms): {len(reranker.arms):,}")

    # Compute reward statistics
    total_gold = sum(1 for arm in reranker.arms.values() for _ in range(arm.count))
    # This is inefficient but works for diagnostics
    gold_counts = 0
    for arm in reranker.arms.values():
        # We'd need to re-compute rewards to get exact stats
        # For now, just report observation count
        pass

    print(f"  Global design matrix A shape: {reranker.A_global.shape}")
    print(f"  Global response vector b shape: {reranker.b_global.shape}")

    # Save model
    print(f"\nSaving model to {output_path}...")
    reranker.save(output_path)
    print(f"Model saved: {Path(output_path).stat().st_size:,} bytes")

    print("=" * 70)


if __name__ == "__main__":
    output_path = "experiments/cache/linucb_model.json"

    if len(sys.argv) > 1:
        if sys.argv[1] == "--output-path" and len(sys.argv) > 2:
            output_path = sys.argv[2]

    train_linucb(output_path)
