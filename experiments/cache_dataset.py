"""Local HotpotQA snapshot loader.

Only the HotpotQA distractor validation split is used by the experiments, so
this module is intentionally small. It reads the Arrow snapshot saved by
``experiments.download_dataset`` and returns plain Python dictionaries for the
callers that expect them.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

DATASET_NAME = "hotpot_qa"
SUBSET = "distractor"
SPLIT = "validation"
SNAPSHOT_DIR = Path(__file__).parent / "cache" / DATASET_NAME / SUBSET / SPLIT


def _validate_config(dataset_name: str, subset: str, split: str) -> None:
    """Ensure the caller is requesting the local dataset snapshot we ship."""
    if (dataset_name, subset, split) != (DATASET_NAME, SUBSET, SPLIT):
        raise NotImplementedError(
            "This repository only ships a local snapshot for "
            f"{DATASET_NAME}/{SUBSET}/{SPLIT}."
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
    """Load the local HotpotQA snapshot and return an optional sample.

    The dataset is expected to exist at:
    ``experiments/cache/hotpot_qa/distractor/validation``.

    Args:
        dataset_name: Dataset name. Only ``hotpot_qa`` is supported.
        subset: Dataset subset. Only ``distractor`` is supported.
        split: Dataset split. Only ``validation`` is supported.
        n_rows: Optional limit on the number of rows to return.
        force_reload: Retained for compatibility with the old API. The loader is
            local-only, so this flag has no effect.

    Returns:
        A list of row dictionaries.
    """
    del force_reload  # Local snapshot loader; there is no remote refresh path.

    _validate_config(dataset_name, subset, split)

    if not SNAPSHOT_DIR.exists():
        raise FileNotFoundError(
            "Local dataset snapshot not found at "
            f"{SNAPSHOT_DIR}. Run `uv run python -m experiments.download_dataset` "
            "first."
        )

    from datasets import load_from_disk

    ds = load_from_disk(SNAPSHOT_DIR)
    return _sample_rows(ds, n_rows)
