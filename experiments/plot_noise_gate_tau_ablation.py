"""Plot the noise-gate tau ablation as a dual-axis line chart."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from experiments.plot_baseline_f1_vs_k import read_csv_flexible

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_OUTPUT = REPO_ROOT / "paper" / "images" / "noise_gate_tau_ablation.pdf"


def load_tau_sweep(model: str) -> pd.DataFrame:
    """Load and aggregate the noise-gate tau sweep."""
    csv_path = REPO_ROOT / "experiments" / "results" / "cart" / f"results_noise_gate_{model}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing results file: {csv_path}")

    df = read_csv_flexible(csv_path)
    required = {"threshold", "f1", "input_tokens", "output_tokens"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {', '.join(sorted(missing))}")

    df = df.copy()
    df["threshold"] = pd.to_numeric(df["threshold"], errors="raise")
    df["f1"] = pd.to_numeric(df["f1"], errors="raise")
    df["input_tokens"] = pd.to_numeric(df["input_tokens"], errors="raise")
    df["output_tokens"] = pd.to_numeric(df["output_tokens"], errors="raise")
    df["total_tokens"] = df["input_tokens"] + df["output_tokens"]

    grouped = (
        df.groupby("threshold", as_index=False)
        .agg(
            f1=("f1", "mean"),
            total_tokens=("total_tokens", "mean"),
        )
        .sort_values("threshold")
    )
    return grouped


def plot_tau_ablation(data: pd.DataFrame, output_path: Path) -> None:
    """Render the dual-axis tau ablation figure."""
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax_f1 = plt.subplots(figsize=(7.6, 4.8))
    ax_tok = ax_f1.twinx()

    color_f1 = "#1f77b4"
    color_tok = "#d95f02"
    thresholds = data["threshold"].to_numpy(dtype=float)
    f1 = data["f1"].to_numpy(dtype=float)
    tokens = data["total_tokens"].to_numpy(dtype=float)

    ax_f1.plot(
        thresholds,
        f1,
        marker="o",
        markersize=6,
        linewidth=2.0,
        color=color_f1,
        label="F1",
        zorder=3,
    )
    ax_tok.plot(
        thresholds,
        tokens,
        marker="s",
        markersize=6,
        linewidth=2.0,
        color=color_tok,
        label="Mean total tokens",
        zorder=2,
    )

    recommended = data.loc[data["threshold"] == 0.3]
    if not recommended.empty:
        row = recommended.iloc[0]
        ax_f1.scatter([row["threshold"]], [row["f1"]], s=100, color=color_f1, edgecolors="black", zorder=4)
        ax_tok.scatter([row["threshold"]], [row["total_tokens"]], s=100, color=color_tok, edgecolors="black", zorder=4)
        ax_f1.annotate(
            r"recommended $\tau=0.3$",
            (row["threshold"], row["f1"]),
            xytext=(8, -18),
            textcoords="offset points",
            fontsize=8.5,
            color="black",
        )

    ax_f1.set_xlabel(r"Cosine threshold $\tau$")
    ax_f1.set_ylabel("F1", color=color_f1)
    ax_tok.set_ylabel("Mean total tokens", color=color_tok)
    ax_f1.tick_params(axis="y", colors=color_f1)
    ax_tok.tick_params(axis="y", colors=color_tok)
    ax_f1.set_xlim(float(thresholds.min()) - 0.02, float(thresholds.max()) + 0.02)
    ax_f1.set_ylim(0.58, 0.80)
    ax_tok.set_ylim(350, 1360)
    ax_f1.set_title("Noise-gate tau ablation on HotpotQA distractor")
    ax_f1.grid(True, axis="y", alpha=0.25)
    ax_f1.grid(False, axis="x")
    sns.despine(ax=ax_f1, top=True, right=False)

    lines_f1, labels_f1 = ax_f1.get_legend_handles_labels()
    lines_tok, labels_tok = ax_tok.get_legend_handles_labels()
    ax_f1.legend(lines_f1 + lines_tok, labels_f1 + labels_tok, frameon=False, loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot the noise-gate tau ablation.")
    parser.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_MODEL,
        help=f"Model name used in the result filename (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PDF path for the figure.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    data = load_tau_sweep(args.model)
    plot_tau_ablation(data, args.output)


if __name__ == "__main__":
    main()
