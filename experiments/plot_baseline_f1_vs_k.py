"""Plot baseline F1 vs k for each model.

This script reads the per-question baseline CSVs in
experiments/results/baseline/ and creates a 2x2 seaborn figure with one
subfigure per model.

The plotting convention treats ``always_think`` as ``k=0`` (no retrieval).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiments.baselines_config import MODELS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "baseline"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "baseline"
    / "baseline_f1_vs_k.pdf"
)
DEFAULT_OUTPUT_POINTS = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "baseline"
    / "baseline_f1_vs_k_points.pdf"
)

MODEL_LABELS = {
    "gpt-4o-mini": "GPT-4o-mini",
    "gpt-5.4-mini": "GPT-5.4-mini",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
}

K_ORDER = [0, 3, 5, 10]
KNOWN_METHODS = {"always_think", "retrieval_k3", "retrieval_k5", "retrieval_k10"}


def method_to_k(method: str) -> int:
    """Map a baseline method name to its retrieval depth k."""
    if method == "always_think":
        return 0

    prefix = "retrieval_k"
    if method.startswith(prefix):
        suffix = method.removeprefix(prefix)
        if suffix.isdigit():
            return int(suffix)

    raise ValueError(f"Unknown baseline method: {method}")


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load and normalize all baseline result CSVs."""
    frames: list[pd.DataFrame] = []

    for model in MODELS:
        csv_path = results_dir / f"baseline_{model}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing results file: {csv_path}")

        df = read_csv_flexible(csv_path)
        df["model_key"] = model
        df["model"] = MODEL_LABELS.get(model, model)
        df["k"] = df["method"].map(method_to_k)

        unknown_methods = sorted(set(df["method"]) - KNOWN_METHODS)
        if unknown_methods:
            raise ValueError(f"Unexpected methods in {csv_path}: {', '.join(unknown_methods)}")

        frames.append(df[["question_id", "method", "model_key", "model", "k", "f1"]])

    data = pd.concat(frames, ignore_index=True)
    data["k"] = data["k"].astype(int)
    return data


def read_csv_flexible(csv_path: Path) -> pd.DataFrame:
    """Read a CSV using the most likely encoding for this repo's results files."""
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    return pd.read_csv(csv_path, encoding="latin-1")


def print_count_table(data: pd.DataFrame) -> None:
    """Print a compact count table for sanity checking."""
    table = (
        data.groupby(["model", "k"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=[MODEL_LABELS[m] for m in MODELS], columns=K_ORDER, fill_value=0)
    )

    print("\nRows per model and k:")
    print(table.to_string())


def plot_baseline_f1_vs_k(
    data: pd.DataFrame,
    output_path: Path,
    errorbar: str | tuple[str, int] | None,
    show_points: bool = False,
) -> None:
    """Create and save the faceted F1-vs-k figure."""
    sns.set_theme(style="whitegrid", context="paper")
    palette = sns.color_palette("colorblind", n_colors=len(MODELS))

    g = sns.relplot(
        data=data,
        x="k",
        y="f1",
        col="model",
        col_order=[MODEL_LABELS[m] for m in MODELS],
        col_wrap=2,
        kind="line",
        estimator="mean",
        errorbar=errorbar,
        n_boot=1000,
        seed=42,
        marker="o",
        dashes=False,
        linewidth=2.0,
        height=2.7,
        aspect=1.35,
        facet_kws={"sharex": True, "sharey": True},
    )

    g.set_axis_labels("k", "F1")
    g.set_titles("{col_name}")

    rng = np.random.default_rng(42)
    for ax, model_key, color in zip(g.axes.flat, MODELS, palette, strict=True):
        model_label = MODEL_LABELS[model_key]
        model_data = data[data["model"] == model_label]

        if show_points:
            x_jitter = model_data["k"].to_numpy(dtype=float) + rng.uniform(
                -0.18,
                0.18,
                size=len(model_data),
            )
            ax.scatter(
                x_jitter,
                model_data["f1"],
                s=18,
                alpha=0.28,
                color=color,
                edgecolors="none",
                rasterized=True,
                zorder=1,
            )

        if ax.lines:
            ax.lines[0].set_color(color)
            ax.lines[0].set_markerfacecolor(color)
            ax.lines[0].set_markeredgecolor(color)

        ax.set_xticks(K_ORDER)
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", alpha=0.3)
        ax.grid(False, axis="x")
        sns.despine(ax=ax, top=True, right=True)

        for line in ax.lines:
            line.set_zorder(3)
            line.set_linewidth(2.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.figure.tight_layout()
    g.figure.savefig(output_path, bbox_inches="tight")
    plt.close(g.figure)

    print(f"Saved figure: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot baseline F1 vs k for each model.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing baseline_{model}.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output figure path.",
    )
    parser.add_argument(
        "--points",
        action="store_true",
        help="Overlay raw per-question F1 points at each k.",
    )
    parser.add_argument(
        "--errorbar",
        choices=["ci", "sd", "none"],
        default="ci",
        help="Uncertainty shown around the mean F1.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    data = load_results(args.results_dir)
    print_count_table(data)
    if args.points:
        errorbar: str | tuple[str, int] | None = None
    elif args.errorbar == "ci":
        errorbar = ("ci", 95)
    elif args.errorbar == "sd":
        errorbar = "sd"
    else:
        errorbar = None

    output_path = args.output
    if args.points and output_path == DEFAULT_OUTPUT:
        output_path = DEFAULT_OUTPUT_POINTS

    plot_baseline_f1_vs_k(data, output_path, errorbar, show_points=args.points)


if __name__ == "__main__":
    main()
