"""Plot baseline methods vs adaptive-k F1 by method.

This script compares the fixed-k baseline methods against the adaptive-k run
for a single model. The x-axis shows one column per method:
Thinking only, retrieval k=3, retrieval k=5, retrieval k=10, and adaptive-k.
Each column shows the raw F1 values jittered horizontally, with the mean F1
overlaid as a point. The adaptive-k rows are aggregated into one column; the
per-row ``k_star`` values are intentionally ignored on the plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiments.baselines_config import METHODS
from experiments.plot_baseline_f1_vs_k import MODEL_LABELS, read_csv_flexible

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_KEY = "gpt-5.4-mini"
BASELINE_RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "baseline"
ADAPTIVE_RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "cart"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "results" / "cart"
ADAPTIVE_METHOD = "adaptive_k"
METHOD_DISPLAY_NAMES = {
    "always_think": "Thinking only",
    "retrieval_k3": "Retrieval k=3",
    "retrieval_k5": "Retrieval k=5",
    "retrieval_k10": "Retrieval k=10",
    "adaptive_k": "Adaptive-k",
}
METHOD_AXIS_LABELS = {
    "always_think": "Thinking only",
    "retrieval_k3": "Retrieval\nk=3",
    "retrieval_k5": "Retrieval\nk=5",
    "retrieval_k10": "Retrieval\nk=10",
    "adaptive_k": "Adaptive-k\n(k*)",
}
METHOD_ORDER = list(METHODS) + [ADAPTIVE_METHOD]
METHOD_LABEL_ORDER = [METHOD_DISPLAY_NAMES[method] for method in METHOD_ORDER]
METHOD_AXIS_ORDER = [METHOD_AXIS_LABELS[method] for method in METHOD_ORDER]
KNOWN_BASELINE_METHODS = set(METHODS)


def baseline_csv_path(model_key: str) -> Path:
    """Return the fixed baseline CSV path for a model."""
    return BASELINE_RESULTS_DIR / f"baseline_{model_key}.csv"


def adaptive_csv_path(model_key: str) -> Path:
    """Return the fixed adaptive-k CSV path for a model."""
    return ADAPTIVE_RESULTS_DIR / f"results_adaptive_k_{model_key}.csv"


def require_columns(csv_path: Path, df: pd.DataFrame, required: set[str]) -> None:
    """Validate that the CSV has the expected columns."""
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {', '.join(missing)}")


def load_baseline_results(csv_path: Path) -> pd.DataFrame:
    """Load the fixed-k baseline CSV and normalize it for plotting."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing results file: {csv_path}")

    df = read_csv_flexible(csv_path)
    require_columns(csv_path, df, {"question_id", "method", "f1"})

    unknown_methods = sorted(set(df["method"]) - KNOWN_BASELINE_METHODS)
    if unknown_methods:
        raise ValueError(f"Unexpected methods in {csv_path}: {', '.join(unknown_methods)}")

    df = df.copy()
    df["f1"] = pd.to_numeric(df["f1"], errors="raise")
    return df[["question_id", "method", "f1"]]


def load_adaptive_results(csv_path: Path) -> pd.DataFrame:
    """Load the adaptive-k CSV and normalize it for plotting."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing results file: {csv_path}")

    df = read_csv_flexible(csv_path)
    require_columns(csv_path, df, {"question_id", "method", "f1", "k_star"})

    unknown_methods = sorted(set(df["method"]) - {ADAPTIVE_METHOD})
    if unknown_methods:
        raise ValueError(f"Unexpected methods in {csv_path}: {', '.join(unknown_methods)}")

    df = df.copy()
    df["k_star"] = pd.to_numeric(df["k_star"], errors="raise").astype(int)
    df["f1"] = pd.to_numeric(df["f1"], errors="raise")
    return df[["question_id", "method", "k_star", "f1"]]


def load_results(model_key: str) -> tuple[pd.DataFrame, str]:
    """Load, normalize, and combine both result sets."""
    baseline_csv = baseline_csv_path(model_key)
    adaptive_csv = adaptive_csv_path(model_key)
    model_label = MODEL_LABELS.get(model_key, model_key)

    baseline = load_baseline_results(baseline_csv)
    baseline["model_key"] = model_key
    baseline["model"] = model_label
    baseline["method_label"] = baseline["method"].map(METHOD_DISPLAY_NAMES)

    adaptive = load_adaptive_results(adaptive_csv)
    adaptive["model_key"] = model_key
    adaptive["model"] = model_label
    adaptive["method_label"] = adaptive["method"].map(METHOD_DISPLAY_NAMES)

    data = pd.concat([baseline, adaptive], ignore_index=True)
    data["method_label"] = pd.Categorical(
        data["method_label"], categories=METHOD_LABEL_ORDER, ordered=True
    )
    data["f1"] = pd.to_numeric(data["f1"], errors="raise")
    return data, model_label


def print_summary_table(data: pd.DataFrame) -> None:
    """Print a compact summary table for sanity checking."""
    table = (
        data.groupby("method_label", as_index=False)
        .agg(count=("f1", "size"), mean_f1=("f1", "mean"), std_f1=("f1", "std"))
        .set_index("method_label")
        .reindex(METHOD_LABEL_ORDER)
    )

    print("\nRows per method:")
    print(table.round({"mean_f1": 3, "std_f1": 3}).to_string())


def default_output_path(model_key: str) -> Path:
    """Build the default output path for the current model."""
    return DEFAULT_OUTPUT_DIR / f"result_adaptive_k_{model_key}.pdf"


def plot_f1_by_method(
    data: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
    model_label: str,
    errorbar: str | tuple[str, int] | None,
) -> None:
    """Create and save the comparison figure."""
    sns.set_theme(style="whitegrid", context="paper")
    palette = sns.color_palette("colorblind", n_colors=len(METHOD_ORDER))
    fig, ax = plt.subplots(figsize=(9.4, 5.1))
    rng = np.random.default_rng(42)

    for idx, method_key in enumerate(METHOD_ORDER):
        method_data = data[data["method"] == method_key]["f1"].to_numpy(dtype=float)
        x_jitter = np.full(len(method_data), idx, dtype=float) + rng.uniform(
            -0.18,
            0.18,
            size=len(method_data),
        )
        ax.scatter(
            x_jitter,
            method_data,
            s=18,
            alpha=0.28,
            color=palette[idx],
            edgecolors="none",
            rasterized=True,
            zorder=1,
        )

        method_label = METHOD_DISPLAY_NAMES[method_key]
        mean_f1 = float(summary.loc[method_label, "mean_f1"])
        std_f1 = float(summary.loc[method_label, "std_f1"])
        n = int(summary.loc[method_label, "count"])
        if errorbar == "ci" and n > 0:
            yerr = 1.96 * std_f1 / np.sqrt(n) if n > 1 else 0.0
        elif errorbar == "sd":
            yerr = std_f1
        else:
            yerr = None

        if yerr is not None:
            ax.errorbar(
                idx,
                mean_f1,
                yerr=yerr,
                fmt="none",
                ecolor=palette[idx],
                elinewidth=1.2,
                capsize=4,
                zorder=3,
            )

        ax.scatter(
            idx,
            mean_f1,
            s=80,
            color=palette[idx],
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )

    ax.set_title(model_label)
    ax.set_xlabel("Method")
    ax.set_ylabel("F1")

    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels(METHOD_AXIS_ORDER)
    ax.set_xlim(-0.5, len(METHOD_ORDER) - 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(False, axis="x")
    sns.despine(ax=ax, top=True, right=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Plot baseline methods vs adaptive-k F1 by method for a single model."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_MODEL_KEY,
        help=f"Model name used in the fixed result filenames (default: {DEFAULT_MODEL_KEY}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path. Defaults to a model-specific PDF in experiments/results/cart/.",
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
    data, model_label = load_results(args.model)
    model_key = args.model

    summary = (
        data.groupby("method_label", as_index=False)
        .agg(count=("f1", "size"), mean_f1=("f1", "mean"), std_f1=("f1", "std"))
        .set_index("method_label")
        .reindex(METHOD_LABEL_ORDER)
    )

    print_summary_table(data)

    if args.errorbar == "ci":
        errorbar: str | tuple[str, int] | None = "ci"
    elif args.errorbar == "sd":
        errorbar = "sd"
    else:
        errorbar = None

    output_path = args.output or default_output_path(model_key)
    plot_f1_by_method(data, summary, output_path, model_label, errorbar)


if __name__ == "__main__":
    main()
