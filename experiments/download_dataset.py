"""Download and persist HotpotQA distractor splits locally.

This uses Hugging Face Datasets' Arrow-based `save_to_disk()` format, which is
much faster to reload than re-downloading or parsing JSON on every run.

Usage:
    uv run python -m experiments.download_dataset
    uv run python -m experiments.download_dataset --split train
    uv run python -m experiments.download_dataset --split validation --force
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DATASET_NAME = "hotpot_qa"
SUBSET = "distractor"


def get_cache_dir(split: str) -> Path:
    """Get the cache directory for a specific split."""
    return Path(__file__).parent / "cache" / DATASET_NAME / SUBSET / split


def download_dataset(split: str = "validation", force: bool = False) -> Path:
    """Download the dataset split and save it to the local cache directory."""
    from datasets import load_dataset

    cache_dir = get_cache_dir(split)

    if cache_dir.exists():
        if not force:
            print(f"Dataset already cached at {cache_dir}")
            return cache_dir
        shutil.rmtree(cache_dir)

    print(f"Downloading {DATASET_NAME}/{SUBSET}/{split} from Hugging Face...")
    ds = load_dataset(DATASET_NAME, SUBSET, split=split)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(cache_dir)
    print(f"Saved {len(ds)} rows to {cache_dir}")
    return cache_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download HotpotQA splits and store them locally."
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation"],
        help="Which split to download (default: validation).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete any existing local snapshot before downloading again.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_dataset(split=args.split, force=args.force)


if __name__ == "__main__":
    main()
