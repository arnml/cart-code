"""Local HotpotQA snapshot loader.

The HotpotQA distractor train and validation splits are used by the experiments.
This module reads Arrow snapshots saved by ``experiments.download_dataset`` and
returns plain Python dictionaries for the callers that expect them.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

DATASET_NAME = "hotpot_qa"
SUBSET = "distractor"
ALLOWED_SPLITS = ("train", "validation")


def _validate_config(dataset_name: str, subset: str, split: str) -> None:
    """Ensure the caller is requesting a supported local dataset snapshot."""
    if dataset_name != DATASET_NAME:
        raise NotImplementedError(f"Only {DATASET_NAME} is supported.")
    if subset != SUBSET:
        raise NotImplementedError(f"Only {SUBSET} is supported.")
    if split not in ALLOWED_SPLITS:
        raise NotImplementedError(
            f"Only {ALLOWED_SPLITS} splits are supported. "
            f"Run `uv run python -m experiments.download_dataset --split {split}` "
            f"to download the {split} split first."
        )


def _sample_rows(ds, n_rows: Optional[int]) -> list[dict]:
    """Return either all rows or a deterministic sample from the dataset."""
    if n_rows is None or n_rows <= 0:
        return [dict(row) for row in ds]

    n_rows = min(n_rows, len(ds))
    rng = random.Random(42)
    indices = rng.sample(range(len(ds)), n_rows)
    return [dict(row) for row in ds.select(indices)]


def load_dataset_cached(
    dataset_name: str,
    subset: str,
    split: str,
    n_rows: Optional[int] = None,
    force_reload: bool = False,
) -> list[dict]:
    """Load a local HotpotQA snapshot and return an optional sample.

    The dataset is expected to exist at:
    ``experiments/cache/hotpot_qa/distractor/{split}``.

    Args:
        dataset_name: Dataset name. Only ``hotpot_qa`` is supported.
        subset: Dataset subset. Only ``distractor`` is supported.
        split: Dataset split. ``train`` or ``validation`` are supported.
        n_rows: Optional limit on the number of rows to return.
        force_reload: Retained for compatibility with the old API. The loader is
            local-only, so this flag has no effect.

    Returns:
        A list of row dictionaries.
    """
    del force_reload  # Local snapshot loader; there is no remote refresh path.

    _validate_config(dataset_name, subset, split)

    snapshot_dir = Path(__file__).parent / "cache" / dataset_name / subset / split
    if not snapshot_dir.exists():
        raise FileNotFoundError(
            "Local dataset snapshot not found at "
            f"{snapshot_dir}. Run `uv run python -m experiments.download_dataset --split {split}` "
            "first."
        )

    from datasets import load_from_disk

    ds = load_from_disk(snapshot_dir)
    return _sample_rows(ds, n_rows)
