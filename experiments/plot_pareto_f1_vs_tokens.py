"""Plot F1 vs. total tokens for the main retrieval strategies.

The figure matches the paper's central Pareto scatter plot:
- fixed retrieval baselines
- adaptive-k
- UCB1-TUNED
- LinUCB
- noise-gate thresholds

The output is written to paper/images/pareto_f1_vs_tokens.pdf by default.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiments.plot_baseline_f1_vs_k import read_csv_flexible

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_OUTPUT = REPO_ROOT / "paper" / "images" / "pareto_f1_vs_tokens.pdf"


@dataclass(frozen=True)
class Point:
    label: str
    family: str
    f1: float
    tokens: float
    marker: str


def _aggregate_point(df: pd.DataFrame, label: str, family: str, marker: str) -> Point:
    """Aggregate a per-question CSV into one plotted point."""
    df = df.copy()
    df["f1"] = pd.to_numeric(df["f1"], errors="raise")
    total_tokens = pd.to_numeric(df["input_tokens"], errors="raise") + pd.to_numeric(
        df["output_tokens"], errors="raise"
    )
    return Point(
        label=label,
        family=family,
        f1=float(df["f1"].mean()),
        tokens=float(total_tokens.mean()),
        marker=marker,
    )


def _load_points(model: str) -> list[Point]:
    """Load the main strategy points for the requested model."""
    baseline_path = REPO_ROOT / "experiments" / "results" / "baseline" / f"baseline_{model}.csv"
    adaptive_path = (
        REPO_ROOT / "experiments" / "results" / "cart" / f"results_adaptive_k_{model}.csv"
    )
    ucb1_path = REPO_ROOT / "experiments" / "results" / "ucb1" / f"ucb1_{model}.csv"
    linucb_path = REPO_ROOT / "experiments" / "results" / "linucb" / f"linucb_{model}.csv"
    noise_gate_path = REPO_ROOT / "experiments" / "results" / "cart" / f"results_noise_gate_{model}.csv"

    for path in [baseline_path, adaptive_path, ucb1_path, linucb_path, noise_gate_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing results file: {path}")

    baseline = read_csv_flexible(baseline_path)
    required = {"method", "f1", "input_tokens", "output_tokens"}
    missing = required - set(baseline.columns)
    if missing:
        raise ValueError(f"Missing columns in {baseline_path}: {', '.join(sorted(missing))}")

    baseline_points: list[Point] = []
    for method, label in [
        ("always_think", "k=0"),
        ("retrieval_k5", "k=5"),
        ("retrieval_k10", "k=10"),
    ]:
        subset = baseline[baseline["method"] == method]
        if subset.empty:
            raise ValueError(f"No rows found for {method} in {baseline_path}")
        baseline_points.append(_aggregate_point(subset, label, "Baseline", "o"))

    adaptive = read_csv_flexible(adaptive_path)
    adaptive_required = {"method", "f1", "input_tokens", "output_tokens"}
    missing = adaptive_required - set(adaptive.columns)
    if missing:
        raise ValueError(f"Missing columns in {adaptive_path}: {', '.join(sorted(missing))}")
    adaptive_points = [_aggregate_point(adaptive, "Adaptive-k", "Learned", "s")]

    ucb1 = read_csv_flexible(ucb1_path)
    ucb1_required = {"method", "k", "f1", "input_tokens", "output_tokens"}
    missing = ucb1_required - set(ucb1.columns)
    if missing:
        raise ValueError(f"Missing columns in {ucb1_path}: {', '.join(sorted(missing))}")
    ucb1_k5 = ucb1[pd.to_numeric(ucb1["k"], errors="raise") == 5]
    if ucb1_k5.empty:
        raise ValueError(f"No k=5 rows found in {ucb1_path}")
    ucb1_points = [_aggregate_point(ucb1_k5, "UCB1-TUNED", "Learned", "s")]

    linucb = read_csv_flexible(linucb_path)
    linucb_required = {"method", "k", "f1", "input_tokens", "output_tokens"}
    missing = linucb_required - set(linucb.columns)
    if missing:
        raise ValueError(f"Missing columns in {linucb_path}: {', '.join(sorted(missing))}")
    linucb_k5 = linucb[pd.to_numeric(linucb["k"], errors="raise") == 5]
    if linucb_k5.empty:
        raise ValueError(f"No k=5 rows found in {linucb_path}")
    linucb_points = [_aggregate_point(linucb_k5, "LinUCB", "Learned", "s")]

    noise_gate = read_csv_flexible(noise_gate_path)
    noise_gate_required = {"threshold", "f1", "input_tokens", "output_tokens"}
    missing = noise_gate_required - set(noise_gate.columns)
    if missing:
        raise ValueError(f"Missing columns in {noise_gate_path}: {', '.join(sorted(missing))}")
    noise_gate["threshold"] = pd.to_numeric(noise_gate["threshold"], errors="raise")
    noise_gate_points: list[Point] = []
    for tau, label in [(0.2, "tau=0.2"), (0.3, "tau=0.3")]:
        subset = noise_gate[np.isclose(noise_gate["threshold"], tau)]
        if subset.empty:
            raise ValueError(f"No tau={tau} rows found in {noise_gate_path}")
        noise_gate_points.append(_aggregate_point(subset, label, "Noise-Gate", "D"))

    return baseline_points + adaptive_points + ucb1_points + linucb_points + noise_gate_points


def _pareto_frontier(points: list[Point]) -> list[Point]:
    """Return non-dominated points sorted from low tokens to high tokens."""
    ordered = sorted(points, key=lambda p: (p.tokens, -p.f1, p.label))
    frontier: list[Point] = []
    best_f1 = -np.inf
    for point in ordered:
        if point.f1 > best_f1 + 1e-12:
            frontier.append(point)
            best_f1 = point.f1
    return frontier


def plot_pareto(points: list[Point], output_path: Path) -> None:
    """Render the Pareto scatter plot."""
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(8.2, 5.4))

    palette = {
        "Baseline": "#4d4d4d",
        "Learned": "#1f77b4",
        "Noise-Gate": "#d95f02",
    }

    reference = next(point for point in points if point.label == "k=10")
    ax.axhspan(
        reference.f1 - 0.05,
        reference.f1 + 0.05,
        color="0.5",
        alpha=0.12,
        label=r"$\pm$0.05 F1 band",
        zorder=0,
    )

    frontier = _pareto_frontier(points)
    frontier_x = [p.tokens for p in frontier]
    frontier_y = [p.f1 for p in frontier]
    ax.plot(
        frontier_x,
        frontier_y,
        linestyle="--",
        color="0.25",
        linewidth=1.4,
        label="Pareto frontier",
        zorder=2,
    )

    offsets = {
        "k=0": (6, -10),
        "k=5": (6, 8),
        "k=10": (-28, -2),
        "Adaptive-k": (-4, -16),
        "UCB1-TUNED": (6, -16),
        "LinUCB": (6, -16),
        "tau=0.2": (6, 8),
        "tau=0.3": (6, -18),
    }

    family_sizes = {
        "Baseline": 90,
        "Learned": 95,
        "Noise-Gate": 120,
    }

    for point in points:
        ax.scatter(
            point.tokens,
            point.f1,
            s=family_sizes[point.family],
            marker=point.marker,
            color=palette[point.family],
            edgecolors="black",
            linewidths=0.8,
            zorder=3 if point.label == "tau=0.3" else 2.5,
        )
        dx, dy = offsets[point.label]
        ax.annotate(
            point.label,
            (point.tokens, point.f1),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.5,
            color="black",
        )

    ax.scatter(
        reference.tokens,
        reference.f1,
        s=165,
        marker="*",
        color=palette["Baseline"],
        edgecolors="black",
        linewidths=0.8,
        zorder=4,
    )

    ax.set_xlabel("Mean total tokens")
    ax.set_ylabel("F1")
    ax.set_title("HotpotQA distractor: quality-cost frontier")
    ax.set_xlim(70, max(p.tokens for p in points) * 1.03)
    ax.set_ylim(0.32, 0.84)
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")
    sns.despine(ax=ax, top=True, right=True)

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    dedup_handles = []
    dedup_labels = []
    for handle, label in zip(handles, labels, strict=True):
        if label in seen:
            continue
        seen.add(label)
        dedup_handles.append(handle)
        dedup_labels.append(label)
    ax.legend(dedup_handles, dedup_labels, frameon=False, loc="lower right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot the main Pareto frontier figure.")
    parser.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_MODEL,
        help=f"Model name used in the result filenames (default: {DEFAULT_MODEL}).",
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
    points = _load_points(args.model)
    plot_pareto(points, args.output)


if __name__ == "__main__":
    main()
