"""Plot baseline F1 by method for each model.

This script reads the per-question baseline CSVs in
experiments/results/baseline/ and creates a 2x2 categorical figure with one
subfigure per model.

Each panel mirrors the adaptive-k comparison style: jittered raw F1 points,
an overlaid mean marker, and optional uncertainty bars for the fixed baseline
methods ``always_think``, ``retrieval_k3``, ``retrieval_k5``, and
``retrieval_k10``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiments.baselines_config import METHODS, MODELS

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

METHOD_DISPLAY_NAMES = {
    "always_think": "Thinking only",
    "retrieval_k3": "Retrieval k=3",
    "retrieval_k5": "Retrieval k=5",
    "retrieval_k10": "Retrieval k=10",
}

METHOD_AXIS_LABELS = {
    "always_think": "Thinking only",
    "retrieval_k3": "Retrieval\nk=3",
    "retrieval_k5": "Retrieval\nk=5",
    "retrieval_k10": "Retrieval\nk=10",
}

METHOD_ORDER = list(METHODS)
METHOD_LABEL_ORDER = [METHOD_DISPLAY_NAMES[method] for method in METHOD_ORDER]
METHOD_AXIS_ORDER = [METHOD_AXIS_LABELS[method] for method in METHOD_ORDER]

K_ORDER = [0, 3, 5, 10]
KNOWN_METHODS = set(METHOD_ORDER)


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
        df["method_label"] = df["method"].map(METHOD_DISPLAY_NAMES)

        unknown_methods = sorted(set(df["method"]) - KNOWN_METHODS)
        if unknown_methods:
            raise ValueError(f"Unexpected methods in {csv_path}: {', '.join(unknown_methods)}")

        frames.append(df[["question_id", "method", "method_label", "model_key", "model", "f1"]])

    data = pd.concat(frames, ignore_index=True)
    data["method_label"] = pd.Categorical(
        data["method_label"], categories=METHOD_LABEL_ORDER, ordered=True
    )
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
        data.groupby(["model", "method_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=[MODEL_LABELS[m] for m in MODELS], columns=METHOD_LABEL_ORDER, fill_value=0)
    )

    print("\nRows per model and method:")
    print(table.to_string())


def summarize_by_method(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate F1 statistics by model and method."""
    summary = (
        data.groupby(["model_key", "model", "method"], as_index=False)
        .agg(count=("f1", "size"), mean_f1=("f1", "mean"), std_f1=("f1", "std"))
        .sort_values(["model_key", "method"])
    )
    summary["method_label"] = summary["method"].map(METHOD_DISPLAY_NAMES)
    summary["method"] = pd.Categorical(summary["method"], categories=METHOD_ORDER, ordered=True)
    summary["method_label"] = pd.Categorical(
        summary["method_label"], categories=METHOD_LABEL_ORDER, ordered=True
    )
    return summary


def resolve_errorbar_yerr(
    errorbar: str | None,
    count: int,
    std_f1: float,
) -> float | None:
    """Convert the requested uncertainty mode into a y-error value."""
    if errorbar == "ci":
        if count > 1 and pd.notna(std_f1):
            return 1.96 * std_f1 / np.sqrt(count)
        return 0.0

    if errorbar == "sd":
        return std_f1 if pd.notna(std_f1) else 0.0

    return None


def add_method_artists(
    ax: plt.Axes,
    idx: int,
    method_color: str,
    method_points: np.ndarray,
    mean_f1: float,
    count: int,
    std_f1: float,
    errorbar: str | None,
    rng: np.random.Generator,
) -> None:
    """Draw the raw points, uncertainty bar, and mean marker for one method."""
    x_jitter = np.full(len(method_points), idx, dtype=float) + rng.uniform(
        -0.18,
        0.18,
        size=len(method_points),
    )
    ax.scatter(
        x_jitter,
        method_points,
        s=18,
        alpha=0.28,
        color=method_color,
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )

    if count <= 0 or pd.isna(mean_f1):
        return

    yerr = resolve_errorbar_yerr(errorbar, count, std_f1)
    if yerr is not None:
        ax.errorbar(
            idx,
            mean_f1,
            yerr=yerr,
            fmt="none",
            ecolor=method_color,
            elinewidth=1.2,
            capsize=4,
            zorder=3,
        )

    ax.scatter(
        idx,
        mean_f1,
        s=80,
        color=method_color,
        edgecolors="black",
        linewidths=0.8,
        zorder=4,
    )


def style_panel_axis(ax: plt.Axes, panel_idx: int, model_label: str) -> None:
    """Apply the shared styling for one model facet."""
    row_idx, col_idx = divmod(panel_idx, 2)
    ax.set_title(model_label)
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels(METHOD_AXIS_ORDER)
    ax.set_xlim(-0.5, len(METHOD_ORDER) - 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Method" if row_idx == 1 else "")
    ax.set_ylabel("F1" if col_idx == 0 else "")
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(False, axis="x")
    sns.despine(ax=ax, top=True, right=True)


def plot_model_panel(
    ax: plt.Axes,
    panel_idx: int,
    model_key: str,
    data: pd.DataFrame,
    summary: pd.DataFrame,
    palette: list[tuple[float, float, float]],
    rng: np.random.Generator,
    errorbar: str | None,
) -> None:
    """Plot one model facet with the fixed baseline methods."""
    model_label = MODEL_LABELS[model_key]
    model_data = data[data["model_key"] == model_key]
    model_summary = summary[summary["model_key"] == model_key].set_index("method").reindex(
        METHOD_ORDER
    )

    for idx, method_key in enumerate(METHOD_ORDER):
        row = model_summary.loc[method_key]
        method_points = model_data[model_data["method"] == method_key]["f1"].to_numpy(
            dtype=float
        )
        add_method_artists(
            ax=ax,
            idx=idx,
            method_color=palette[idx],
            method_points=method_points,
            mean_f1=float(row["mean_f1"]) if pd.notna(row["mean_f1"]) else np.nan,
            count=int(row["count"]) if pd.notna(row["count"]) else 0,
            std_f1=float(row["std_f1"]) if pd.notna(row["std_f1"]) else np.nan,
            errorbar=errorbar,
            rng=rng,
        )

    style_panel_axis(ax, panel_idx, model_label)


def plot_baseline_f1_vs_k(
    data: pd.DataFrame,
    output_path: Path,
    errorbar: str | None,
) -> None:
    """Create and save the faceted F1-by-method figure."""
    sns.set_theme(style="whitegrid", context="paper")
    palette = sns.color_palette("colorblind", n_colors=len(METHOD_ORDER))
    summary = summarize_by_method(data)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    rng = np.random.default_rng(42)

    for panel_idx, (ax, model_key) in enumerate(zip(axes_flat, MODELS, strict=True)):
        plot_model_panel(
            ax=ax,
            panel_idx=panel_idx,
            model_key=model_key,
            data=data,
            summary=summary,
            palette=palette,
            rng=rng,
            errorbar=errorbar,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot baseline F1 by method for each model.")
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
        help="Use the legacy points-named output path.",
    )
    parser.add_argument(
        "--errorbar",
        choices=["ci", "sd", "none"],
        default="none",
        help="Uncertainty shown around the mean F1.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    data = load_results(args.results_dir)
    print_count_table(data)
    errorbar = None if args.errorbar == "none" else args.errorbar

    output_path = args.output
    if args.points and output_path == DEFAULT_OUTPUT:
        output_path = DEFAULT_OUTPUT_POINTS

    plot_baseline_f1_vs_k(data, output_path, errorbar)


if __name__ == "__main__":
    main()
