"""Download and cache MuSiQue-Ans locally.

Usage:
    uv run python -m experiments.musique_ans.download_dataset
    uv run python -m experiments.musique_ans.download_dataset --split train
    uv run python -m experiments.musique_ans.download_dataset --split all --force
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from datasets import load_dataset

from .utils import CACHE_DIR, DATASET_NAME

SPLITS = ("train", "validation")


def get_cache_dir(split: str) -> Path:
    return CACHE_DIR / split


def download_split(split: str, force: bool = False) -> Path:
    cache_dir = get_cache_dir(split)

    if cache_dir.exists():
        if not force:
            print(f"Dataset already cached at {cache_dir}")
            return cache_dir
        shutil.rmtree(cache_dir)

    print(f"Downloading {DATASET_NAME}/{split} from Hugging Face...")
    ds = load_dataset(DATASET_NAME, split=split)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(cache_dir)
    print(f"Saved {len(ds)} rows to {cache_dir}")
    return cache_dir


def download_dataset(split: str = "all", force: bool = False) -> list[Path]:
    if split == "all":
        return [download_split(one_split, force=force) for one_split in SPLITS]
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    return [download_split(split, force=force)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and cache MuSiQue-Ans.")
    parser.add_argument(
        "--split",
        default="all",
        choices=["train", "validation", "all"],
        help="Which split to download (default: all).",
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

