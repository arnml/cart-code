"""Dataset caching utilities to avoid re-loading and re-processing large datasets.

Provides persistent, disk-based caching for HotpotQA and other datasets.
Supports partial loads and deterministic random sampling.
"""

import json
import random
from pathlib import Path
from typing import Optional

# Cache root for datasets
_CACHE_ROOT = Path(__file__).parent / "cache" / "dataset_cache"


def _cache_path(dataset_name: str, subset: str, split: str) -> Path:
    """Return path to the cache file for this dataset config.

    Args:
        dataset_name: Dataset name (e.g., "hotpot_qa")
        subset: Dataset subset (e.g., "distractor")
        split: Dataset split (e.g., "validation")

    Returns:
        Path to cache file
    """
    return _CACHE_ROOT / dataset_name / subset / f"{split}.json"


def load_dataset_cached(
    dataset_name: str,
    subset: str,
    split: str,
    n_rows: Optional[int] = None,
    force_reload: bool = False,
) -> list[dict]:
    """Load dataset with caching support.

    First checks if the full dataset is cached on disk. If not, loads from Hugging Face
    and caches it. Then returns a deterministic random sample of n_rows (or all if n_rows is None).

    Args:
        dataset_name: Dataset name (e.g., "hotpot_qa")
        subset: Dataset subset (e.g., "distractor")
        split: Dataset split (e.g., "validation")
        n_rows: Optional limit on number of rows to return. If None, returns all.
                Uses deterministic random sampling (seed=42) for reproducibility.
        force_reload: If True, ignore cache and reload from Hugging Face

    Returns:
        List of dictionaries (rows) from the dataset

    Example:
        >>> ds = load_dataset_cached("hotpot_qa", "distractor", "validation", n_rows=100)
        >>> len(ds)  # <= 100 (or full dataset size if < 100)
        >>> # Same 100 records returned on every call with n_rows=100
    """
    from datasets import load_dataset

    cache_file = _cache_path(dataset_name, subset, split)

    # Try to load from cache
    if not force_reload and cache_file.exists():
        print(f"Loading {dataset_name}/{subset}/{split} from cache...")
        with open(cache_file) as f:
            data = json.load(f)
        rows = data.get("rows", [])
        print(f"Loaded {len(rows)} rows from cache")
    else:
        # Load from Hugging Face and cache
        print(f"Loading {dataset_name}/{subset}/{split} from Hugging Face...")
        ds = load_dataset(dataset_name, subset, split=split)
        rows = [dict(row) for row in ds]
        print(f"Loaded {len(rows)} rows from Hugging Face")

        # Save to cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "dataset_name": dataset_name,
            "subset": subset,
            "split": split,
            "total_rows": len(rows),
            "rows": rows,
        }
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
        print(f"Cached to {cache_file}")

    # Return deterministic random sample
    if n_rows is not None and n_rows > 0:
        n_rows = min(n_rows, len(rows))
        random.seed(42)  # Fixed seed for reproducibility
        sample_indices = random.sample(range(len(rows)), n_rows)
        return [rows[i] for i in sample_indices]
    return rows


def get_row(
    dataset_name: str,
    subset: str,
    split: str,
    row_idx: int,
) -> dict:
    """Get a single row from cached dataset.

    Loads the full cached dataset and returns one row by index.

    Args:
        dataset_name: Dataset name
        subset: Dataset subset
        split: Dataset split
        row_idx: Row index (0-based)

    Returns:
        Single row as dictionary

    Raises:
        IndexError: If row_idx is out of bounds
    """
    ds = load_dataset_cached(dataset_name, subset, split)
    if row_idx < 0 or row_idx >= len(ds):
        raise IndexError(f"Row index {row_idx} out of bounds for dataset size {len(ds)}")
    return ds[row_idx]


def clear_cache(
    dataset_name: str,
    subset: Optional[str] = None,
    split: Optional[str] = None,
) -> None:
    """Clear dataset cache selectively.

    Args:
        dataset_name: Dataset name to clear
        subset: Optional subset to clear. If None, clears all subsets of dataset.
        split: Optional split to clear. If None, clears all splits.

    Example:
        >>> clear_cache("hotpot_qa")  # Clear all hotpot_qa caches
        >>> clear_cache("hotpot_qa", "distractor", "validation")  # Clear specific
    """
    import shutil

    if subset is None:
        # Clear entire dataset
        cache_dir = _CACHE_ROOT / dataset_name
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"Cleared cache for {dataset_name}")
    elif split is None:
        # Clear specific subset
        cache_dir = _CACHE_ROOT / dataset_name / subset
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"Cleared cache for {dataset_name}/{subset}")
    else:
        # Clear specific file
        cache_file = _cache_path(dataset_name, subset, split)
        if cache_file.exists():
            cache_file.unlink()
            print(f"Cleared cache for {dataset_name}/{subset}/{split}")


if __name__ == "__main__":
    """Test dataset caching."""

    print("Testing dataset caching...\n")

    # Load a small subset
    ds = load_dataset_cached("hotpot_qa", "distractor", "validation", n_rows=5)
    print(f"\nLoaded {len(ds)} rows")
    print(f"First row keys: {list(ds[0].keys())}\n")

    # Load again (should hit cache)
    ds2 = load_dataset_cached("hotpot_qa", "distractor", "validation", n_rows=5)
    print(f"Loaded again: {len(ds2)} rows (from cache)\n")

    # Get single row
    row = get_row("hotpot_qa", "distractor", "validation", 0)
    print(f"Single row question: {row['question'][:50]}...\n")

    print("Test complete!")
