"""Plot gold-support frequency by BM25 rank position for HotpotQA.

This figure supports the UCB1 analysis by showing that rank position is nearly
uniformly predictive of supporting-fact presence on the training split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCOREBOARD = REPO_ROOT / "experiments" / "cache" / "ucb1_scoreboard.json"
DEFAULT_OUTPUT = REPO_ROOT / "paper" / "images" / "rank_position_gold_frequency.pdf"


def load_rank_rates(scoreboard_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load rank means/counts from the cached UCB1 scoreboard."""
    if not scoreboard_path.exists():
        raise FileNotFoundError(f"Missing scoreboard: {scoreboard_path}")

    with open(scoreboard_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scoreboard = data.get("scoreboard", {})
    ranks = sorted((int(rank_str) for rank_str in scoreboard.keys()))
    means = []
    counts = []
    for rank in ranks:
        stats = scoreboard[str(rank)]
        means.append(float(stats["mean"]))
        counts.append(int(stats["count"]))

    means_arr = np.array(means, dtype=float)
    counts_arr = np.array(counts, dtype=float)
    stderr = np.sqrt(means_arr * (1.0 - means_arr) / counts_arr)
    return np.array(ranks, dtype=int), means_arr * 100.0, stderr * 100.0


def plot_rank_frequency(ranks: np.ndarray, rates: np.ndarray, stderr: np.ndarray, output_path: Path) -> None:
    """Render the bar chart."""
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    bars = ax.bar(
        ranks,
        rates,
        color="#4d4d4d",
        edgecolor="black",
        linewidth=0.8,
        width=0.72,
        zorder=3,
    )
    ax.errorbar(
        ranks,
        rates,
        yerr=1.96 * stderr,
        fmt="none",
        ecolor="#1f77b4",
        elinewidth=1.1,
        capsize=3,
        zorder=4,
        label="95% CI",
    )

    mean_rate = float(np.mean(rates))
    ax.axhline(mean_rate, linestyle="--", color="#d95f02", linewidth=1.4, label=f"Mean {mean_rate:.1f}%")

    for bar, rate in zip(bars, rates, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.05,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("BM25 rank position")
    ax.set_ylabel("Supporting-fact rate (%)")
    ax.set_title("HotpotQA distractor: gold-support frequency by rank")
    ax.set_xticks(ranks)
    ax.set_ylim(18.7, 21.2)
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")
    sns.despine(ax=ax, top=True, right=True)
    ax.legend(frameon=False, loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot rank-position gold-support frequency.")
    parser.add_argument(
        "--scoreboard",
        type=Path,
        default=DEFAULT_SCOREBOARD,
        help=f"Path to the cached scoreboard JSON (default: {DEFAULT_SCOREBOARD}).",
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
    ranks, rates, stderr = load_rank_rates(args.scoreboard)
    plot_rank_frequency(ranks, rates, stderr, args.output)


if __name__ == "__main__":
    main()
