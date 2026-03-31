"""Download and persist the HotpotQA validation split locally.

This uses Hugging Face Datasets' Arrow-based `save_to_disk()` format, which is
much faster to reload than re-downloading or parsing JSON on every run.

Usage:
    uv run python -m experiments.download_dataset
    uv run python -m experiments.download_dataset --force
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DATASET_NAME = "hotpot_qa"
SUBSET = "distractor"
SPLIT = "validation"
CACHE_DIR = Path(__file__).parent / "cache" / DATASET_NAME / SUBSET / SPLIT


def download_dataset(force: bool = False) -> Path:
    """Download the dataset split and save it to the local cache directory."""
    from datasets import load_dataset

    if CACHE_DIR.exists():
        if not force:
            print(f"Dataset already cached at {CACHE_DIR}")
            return CACHE_DIR
        shutil.rmtree(CACHE_DIR)

    print(f"Downloading {DATASET_NAME}/{SUBSET}/{SPLIT} from Hugging Face...")
    ds = load_dataset(DATASET_NAME, SUBSET, split=SPLIT)
    CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(CACHE_DIR)
    print(f"Saved {len(ds)} rows to {CACHE_DIR}")
    return CACHE_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download HotpotQA validation and store it locally."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete any existing local snapshot before downloading again.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_dataset(force=args.force)


if __name__ == "__main__":
    main()
