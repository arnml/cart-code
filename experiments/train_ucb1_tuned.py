"""Train the UCB1-TUNED reranker on HotpotQA train split.

This script is run once to build a scoreboard of title utilities, which is then
loaded and used during inference. Training is fast (no LLM calls, just Welford
variance updates).

Usage from root:
    uv run python -m experiments.train_ucb1_tuned
    uv run python -m experiments.train_ucb1_tuned --output-path custom/path.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiments.baselines_config import DATASET_CONFIG
from experiments.cache_dataset import load_dataset_cached
from experiments.ucb1_tuned import UCB1TunedReranker

DEFAULT_OUTPUT_PATH = (
    Path(__file__).parent / "cache" / "ucb1_scoreboard.json"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train UCB1-TUNED reranker on HotpotQA train split."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Path to save the trained scoreboard (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    """Load train split, train reranker, save scoreboard."""
    args = parse_args()
    output_path = Path(args.output_path)

    print("\n" + "=" * 70)
    print("UCB1-TUNED: Training Phase")
    print("=" * 70)

    # Load train split
    print("\nLoading HotpotQA train split...")
    try:
        ds = load_dataset_cached(
            dataset_name=DATASET_CONFIG["dataset_name"],
            subset=DATASET_CONFIG["subset"],
            split="train",
            n_rows=None,  # Use full train split
        )
    except NotImplementedError as e:
        print(f"Error: {e}")
        print("Please download the train split first:")
        print("  uv run python -m experiments.download_dataset --split train")
        sys.exit(1)

    print(f"Loaded {len(ds)} training samples")

    # Create and train reranker
    print("\nTraining reranker...")
    reranker = UCB1TunedReranker(num_ranks=10)
    reranker.train(ds)

    # Print stats
    print("\nTraining Complete!")
    print(f"  Total observations: {reranker.t:,}")
    print(f"  Rank positions: {reranker.num_ranks}")

    # Gold rate by rank position
    print("\n  Gold rate by rank position:")
    print(f"  {'Rank':>4} | {'Count':>8} | {'Gold Rate':>10} | {'Variance':>10}")
    print("  " + "-" * 45)
    for rank in range(reranker.num_ranks):
        stats = reranker.scoreboard[rank]
        variance = stats.M / (stats.count - 1) if stats.count > 1 else 0.0
        print(
            f"  {rank:>4} | {stats.count:>8} | {stats.mean:>10.4f} | {variance:>10.4f}"
        )

    # Save scoreboard
    print(f"\nSaving scoreboard to {output_path}...")
    reranker.save(str(output_path))
    print(f"Scoreboard saved successfully!")

    print("\n" + "=" * 70)
    print("Next step: Run inference with the trained scoreboard")
    print(f"  uv run python -m experiments.run_ucb1_tuned gpt-4o-mini 100")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
