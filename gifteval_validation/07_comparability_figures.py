#!/usr/bin/env python3

from pathlib import Path
from _paths import ARTIFACT_DIR, FIGURE_DIR

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILE = ARTIFACT_DIR / "raw_comparison_long.csv"
SUMMARY_FILE = ARTIFACT_DIR / "comparability_by_model_metric.csv"
RANKING_FILE = ARTIFACT_DIR / "comparability_rankings.csv"


# ============================================================
# LOAD
# ============================================================

raw = pd.read_csv(RAW_FILE)
summary = pd.read_csv(SUMMARY_FILE)
rankings = pd.read_csv(RANKING_FILE)


# ============================================================
# FIGURE 1 — SCATTER GIFT-EVAL vs BENCHOPT
# ============================================================
#
# We focus on MAE, MSE and sMAPE because these are the metrics
# showing the strongest agreement.
# ============================================================

scatter_metrics = [
    "MAE",
    "MSE",
    "sMAPE",
]

for metric in scatter_metrics:

    df = raw[
        raw["metric"] == metric
    ].copy()

    df = df[
        np.isfinite(df["benchopt_value"])
        & np.isfinite(df["gifteval_value"])
    ]

    if df.empty:
        continue

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    for solver in sorted(
        df["solver_family"].unique()
    ):

        subset = df[
            df["solver_family"] == solver
        ]

        ax.scatter(
            subset["gifteval_value"],
            subset["benchopt_value"],
            label=solver,
            alpha=0.7,
        )

    all_values = np.concatenate(
        [
            df["gifteval_value"].to_numpy(),
            df["benchopt_value"].to_numpy(),
        ]
    )

    finite = all_values[
        np.isfinite(all_values)
    ]

    if len(finite):

        lower = np.min(finite)
        upper = np.max(finite)

        ax.plot(
            [lower, upper],
            [lower, upper],
            linestyle="--",
            linewidth=1.5,
            label="Perfect agreement",
        )

    ax.set_xlabel(
        f"GIFT-Eval {metric}"
    )

    ax.set_ylabel(
        f"BenchOpt {metric}"
    )

    ax.set_title(
        f"BenchOpt vs GIFT-Eval — {metric}"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    path = (
        FIGURE_DIR
        / f"scatter_{metric.lower()}.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# FIGURE 2 — CORRELATION BY MODEL × METRIC
# ============================================================

corr = summary.copy()

pivot = corr.pivot(
    index="solver_family",
    columns="metric",
    values="correlation",
)

metrics_order = [
    metric
    for metric in [
        "MAE",
        "MSE",
        "MASE",
        "sMAPE",
        "WQL",
    ]
    if metric in pivot.columns
]

pivot = pivot[
    metrics_order
]

fig, ax = plt.subplots(
    figsize=(10, 6)
)

x = np.arange(
    len(pivot.index)
)

n_metrics = len(metrics_order)

width = 0.8 / max(
    n_metrics,
    1,
)

for idx, metric in enumerate(
    metrics_order
):

    offset = (
        idx
        - (n_metrics - 1) / 2
    ) * width

    ax.bar(
        x + offset,
        pivot[metric],
        width=width,
        label=metric,
    )

ax.axhline(
    0.90,
    linestyle="--",
    linewidth=1,
)

ax.set_xticks(x)

ax.set_xticklabels(
    pivot.index
)

ax.set_ylabel(
    "Correlation with GIFT-Eval"
)

ax.set_title(
    "Dataset-level correlation — BenchOpt vs GIFT-Eval"
)

ax.set_ylim(
    -0.1,
    1.05,
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25,
)

fig.tight_layout()

corr_path = (
    FIGURE_DIR
    / "correlation_by_model_metric.png"
)

fig.savefig(
    corr_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 3 — RANK COMPARISON
# ============================================================

rank_metrics = [
    "MAE",
    "MSE",
    "sMAPE",
    "WQL",
    "MASE",
]

rank_metrics = [
    metric
    for metric in rank_metrics
    if metric in set(rankings["metric"])
]

for metric in rank_metrics:

    df = rankings[
        rankings["metric"] == metric
    ].copy()

    if df.empty:
        continue

    df = df.sort_values(
        "gifteval_rank"
    )

    y = np.arange(
        len(df)
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.scatter(
        df["gifteval_rank"],
        y,
        s=100,
        label="GIFT-Eval",
    )

    ax.scatter(
        df["benchopt_rank"],
        y,
        s=100,
        marker="x",
        label="BenchOpt",
    )

    for idx, row in df.iterrows():

        y_pos = list(
            df.index
        ).index(idx)

        ax.plot(
            [
                row["gifteval_rank"],
                row["benchopt_rank"],
            ],
            [
                y_pos,
                y_pos,
            ],
            linewidth=1.5,
        )

    ax.set_yticks(y)

    ax.set_yticklabels(
        df["solver_family"]
    )

    ax.set_xlabel(
        "Rank"
    )

    ax.set_title(
        f"Model ranking — {metric}"
    )

    ax.set_xticks(
        sorted(
            set(
                df["gifteval_rank"]
            )
            |
            set(
                df["benchopt_rank"]
            )
        )
    )

    ax.invert_yaxis()

    ax.legend()

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    fig.tight_layout()

    path = (
        FIGURE_DIR
        / f"ranking_{metric.lower()}.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# FIGURE 4 — MEDIAN RELATIVE DIFFERENCE
# ============================================================

diff = summary.copy()

pivot_diff = diff.pivot(
    index="solver_family",
    columns="metric",
    values="median_relative_difference_percent",
)

pivot_diff = pivot_diff[
    [
        metric
        for metric in metrics_order
        if metric in pivot_diff.columns
    ]
]

fig, ax = plt.subplots(
    figsize=(10, 6)
)

x = np.arange(
    len(pivot_diff.index)
)

n_metrics = len(
    pivot_diff.columns
)

width = 0.8 / max(
    n_metrics,
    1,
)

for idx, metric in enumerate(
    pivot_diff.columns
):

    offset = (
        idx
        - (n_metrics - 1) / 2
    ) * width

    ax.bar(
        x + offset,
        pivot_diff[metric],
        width=width,
        label=metric,
    )

ax.set_xticks(x)

ax.set_xticklabels(
    pivot_diff.index
)

ax.set_ylabel(
    "Median relative difference (%)"
)

ax.set_title(
    "Median numerical difference — BenchOpt vs GIFT-Eval"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25,
)

fig.tight_layout()

diff_path = (
    FIGURE_DIR
    / "median_relative_difference.png"
)

fig.savefig(
    diff_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 90)
print("GIFT-EVAL COMPARABILITY FIGURES")
print("=" * 90)

print()
print("Figures generated in:")
print(FIGURE_DIR)

print()

for path in sorted(
    FIGURE_DIR.glob("*.png")
):
    print(" -", path)

print()
print("=" * 90)
