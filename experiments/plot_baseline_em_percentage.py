"""Plot EM=0 vs EM=1 percentages for each model and retrieval depth.

This script reads the per-question baseline CSVs in
experiments/results/baseline/ and creates a 2x2 figure with one subfigure
per model.

Each subplot shows stacked percentage bars for EM=1 and EM=0 at
``k in {0, 3, 5, 10}``, with ``always_think`` mapped to ``k=0``.
The mean EM rate is drawn as a line over the bars.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from experiments.baselines_config import MODELS
from experiments.plot_baseline_f1_vs_k import (
    DEFAULT_RESULTS_DIR,
    K_ORDER,
    MODEL_LABELS,
    method_to_k,
    read_csv_flexible,
)

DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "baseline_em_percentage_vs_k.pdf"

EM1_COLOR = "#4C78A8"
EM0_COLOR = "#D9D9D9"
EM_LINE_COLOR = "#1F1F1F"


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
        df["em"] = pd.to_numeric(df["em"], errors="raise")
        frames.append(df[["question_id", "method", "model_key", "model", "k", "em"]])

    data = pd.concat(frames, ignore_index=True)
    data["k"] = data["k"].astype(int)
    data["em"] = data["em"].astype(int)
    return data


def summarize_em(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize EM outcomes as percentages by model and k."""
    summary = (
        data.groupby(["model_key", "model", "k"], as_index=False)
        .agg(n=("em", "size"), em1_pct=("em", "mean"))
        .sort_values(["model_key", "k"])
    )
    summary["em1_pct"] = summary["em1_pct"] * 100
    summary["em0_pct"] = 100 - summary["em1_pct"]
    return summary


def print_summary(summary: pd.DataFrame) -> None:
    """Print a compact summary table for sanity checking."""
    n_table = (
        summary.pivot(index="model", columns="k", values="n")
        .reindex(index=[MODEL_LABELS[m] for m in MODELS], columns=K_ORDER, fill_value=0)
        .fillna(0)
    )
    em1_table = (
        summary.pivot(index="model", columns="k", values="em1_pct")
        .reindex(index=[MODEL_LABELS[m] for m in MODELS], columns=K_ORDER, fill_value=0)
        .fillna(0)
    )

    print("\nRows per model and k:")
    print(n_table.to_string())
    print("\nEM=1 percentage per model and k:")
    print(em1_table.round(1).to_string())


def plot_em_percentage(summary: pd.DataFrame, output_path: Path) -> None:
    """Create and save the faceted stacked bar figure."""
    sns.set_theme(style="whitegrid", context="paper")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    x_positions = np.arange(len(K_ORDER))
    bar_width = 0.72

    for ax, model_key in zip(axes_flat, MODELS, strict=True):
        model_label = MODEL_LABELS[model_key]
        model_summary = summary[summary["model_key"] == model_key].set_index("k").reindex(
            K_ORDER
        )

        em1 = model_summary["em1_pct"].to_numpy()
        em0 = model_summary["em0_pct"].to_numpy()

        ax.bar(
            x_positions,
            em1,
            width=bar_width,
            color=EM1_COLOR,
            edgecolor="white",
            linewidth=0.9,
            label="EM=1",
            zorder=2,
        )
        ax.bar(
            x_positions,
            em0,
            width=bar_width,
            bottom=em1,
            color=EM0_COLOR,
            edgecolor="white",
            linewidth=0.9,
            label="EM=0",
            zorder=3,
        )
        ax.plot(
            x_positions,
            em1,
            color=EM_LINE_COLOR,
            marker="o",
            markersize=4.5,
            markerfacecolor="white",
            markeredgecolor=EM_LINE_COLOR,
            markeredgewidth=1.0,
            linewidth=1.6,
            linestyle="--",
            label="Mean EM",
            zorder=4,
        )

        ax.set_title(model_label)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(k) for k in K_ORDER])
        ax.set_xlabel("k")
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.grid(True, axis="y", alpha=0.3)
        ax.grid(False, axis="x")
        sns.despine(ax=ax, top=True, right=True)

    axes[0, 0].set_ylabel("Percentage of questions")
    axes[1, 0].set_ylabel("Percentage of questions")

    legend_handles = [
        Patch(facecolor=EM1_COLOR, edgecolor="white", label="EM=1"),
        Patch(facecolor=EM0_COLOR, edgecolor="white", label="EM=0"),
        Line2D(
            [0],
            [0],
            color=EM_LINE_COLOR,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=EM_LINE_COLOR,
            linestyle="--",
            linewidth=1.6,
            label="Mean EM",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot EM percentages for each model and k.")
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
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    data = load_results(args.results_dir)
    summary = summarize_em(data)
    print_summary(summary)
    plot_em_percentage(summary, args.output)


if __name__ == "__main__":
    main()
