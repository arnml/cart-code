from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency: `datasets`.\n"
        "Activate the project venv and re-run:\n"
        "  .\\.venv\\Scripts\\Activate.ps1\n"
        "  python experiments\\download_dataset.py\n"
    ) from e


def _first_available_split(ds, preferred=("train", "validation", "test")) -> str:
    for name in preferred:
        if name in ds:
            return name
    # Fallback: first split in whatever order HF provides.
    return next(iter(ds.keys()))


def _pick_first_present(example: dict, candidates: list[str]):
    for k in candidates:
        if k in example and example[k] is not None:
            return example[k]
    return None


def main() -> None:
    # Windows terminals often use a legacy code page (cp1252) which can crash when printing JSON
    # containing non-ASCII characters. Prefer utf-8 output when possible.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # Downloads & caches the dataset locally.
    ds = load_dataset("hotpot_qa", "distractor")
    print("Dataset loaded:", ds)

    split = _first_available_split(ds)
    print(f"Using split: {split}")

    # Inspect schema once (helps understand how to extract question/answers/contexts).
    ds_split = ds[split]
    sample = ds_split[0]
    print("Example keys:", list(sample.keys()))

    # `Dataset[:10]` returns a dict of columns (e.g. {"question": [...], ...}),
    # but iterating over it yields the *column names* (strings).
    # Use `select` so each iteration gives a row dict.
    n = min(10, getattr(ds_split, "num_rows", 10))
    examples_ds = ds_split.select(range(n))

    extracted = []
    for i, ex in enumerate(examples_ds):
        question = _pick_first_present(ex, ["question", "q"])

        # HotpotQA answers can be a string, list of strings, or sometimes a dict/list under different keys.
        answers = _pick_first_present(ex, ["answer", "answers", "gold_answer", "supporting_answers"])
        contexts = _pick_first_present(
            ex,
            [
                "contexts",
                "context",
                "paragraphs",
                "supporting_contexts",
                "evidence",
            ],
        )

        extracted.append(
            {
                "idx": i,
                "question": question,
                "answers": answers,
                "contexts": contexts,
            }
        )

    print("\nFirst 10 examples (extracted fields):")
    # Use ensure_ascii=True for console printing to avoid encoding issues in older terminals.
    print(json.dumps(extracted, ensure_ascii=True, indent=2))

    out_path = Path(__file__).with_name("hotpotqa_distractor_examples.json")
    out_path.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

