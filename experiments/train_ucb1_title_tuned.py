"""Train the legacy UCB1-TUNED title reranker on HotpotQA train split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiments.baselines_config import DATASET_CONFIG
from experiments.cache_dataset import load_dataset_cached
from experiments.ucb1_title_tuned import UCB1TitleTunedReranker

DEFAULT_OUTPUT_PATH = Path(__file__).parent / "cache" / "ucb1_title_scoreboard.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train legacy UCB1-TUNED title reranker on HotpotQA train split."
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
    print("Legacy UCB1-TUNED (title-based): Training Phase")
    print("=" * 70)

    print("\nLoading HotpotQA train split...")
    try:
        ds = load_dataset_cached(
            dataset_name=DATASET_CONFIG["dataset_name"],
            subset=DATASET_CONFIG["subset"],
            split="train",
            n_rows=None,
        )
    except NotImplementedError as e:
        print(f"Error: {e}")
        print("Please download the train split first:")
        print("  uv run python -m experiments.download_dataset --split train")
        sys.exit(1)

    print(f"Loaded {len(ds)} training samples")

    print("\nTraining reranker...")
    reranker = UCB1TitleTunedReranker(optimistic_init=1.0)
    reranker.train(ds)

    print("\nTraining Complete!")
    print(f"  Total observations: {reranker.t:,}")
    print(f"  Unique titles: {len(reranker.scoreboard):,}")

    print(f"\nSaving scoreboard to {output_path}...")
    reranker.save(str(output_path))
    print("Scoreboard saved successfully!")

    print("\n" + "=" * 70)
    print("Next step: Run legacy title-based inference with the trained scoreboard")
    print("  uv run python -m experiments.run_ucb1_title_tuned gpt-4o-mini 100")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
