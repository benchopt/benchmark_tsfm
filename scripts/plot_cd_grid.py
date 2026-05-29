"""Render a grid of CD diagrams — one subplot per objective_* metric — for a
benchopt-schema parquet, and save it as a single PNG.

Uses scikit_posthocs.critical_difference_diagram for the drawing, keeps only
the minimal preprocessing logic (pivot, biclique trim for sparse benchmarks).

Usage:
    python scripts/plot_cd_grid.py <parquet> [--out PATH] [--filter QUERY]
                                  [--ncols N] [--top-k N]
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, rankdata, studentized_range


def short_solver(name: str) -> str:
    return re.sub(r"\[.*?\]$", "", str(name)).strip()


def discover_metrics(df: pd.DataFrame) -> list[str]:
    """Return all numeric objective_* columns that have any non-NaN values."""
    cols = []
    for c in df.columns:
        if not c.startswith("objective_") or c == "objective_name":
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].notna().sum() == 0:
            continue
        cols.append(c)
    return cols


def greedy_biclique(mat: pd.DataFrame, min_solvers: int = 3,
                    min_datasets: int = 5) -> pd.DataFrame:
    """Iteratively drop the row or column with the most NaNs until the block is
    full or one of the size floors is hit. Greedy approx for max-biclique."""
    work = mat.copy()
    while work.isna().any().any():
        if work.shape[0] <= min_solvers or work.shape[1] <= min_datasets:
            break
        row_nan = work.isna().sum(axis=1)
        col_nan = work.isna().sum(axis=0)
        if row_nan.max() >= col_nan.max():
            work = work.drop(index=row_nan.idxmax())
        else:
            work = work.drop(columns=col_nan.idxmax())
    return work


def prepare_matrix(df: pd.DataFrame, metric: str, top_k: int | None = None,
                   ) -> tuple[pd.DataFrame, str]:
    """Pivot to (solver × dataset), strict-clean, then greedy biclique if needed.

    Returns the cleaned matrix and a one-line note for the subplot caption.
    """
    pivot = df.pivot_table(index="solver_name", columns="dataset_name",
                           values=metric, aggfunc="mean")
    k0, n0 = pivot.shape

    # Strict: drop datasets with any NaN, drop all-tied datasets
    complete = pivot.loc[:, pivot.notna().all(axis=0)]
    n_drop_nan = pivot.shape[1] - complete.shape[1]
    if complete.shape[1] > 0:
        var = complete.var(axis=0, skipna=False)
        complete = complete.loc[:, var > 0]

    note_parts = [f"started {k0}×{n0}"]
    if complete.shape[1] == 0:
        # Sparse benchmark — fall back to greedy biclique on the original.
        var = pivot.var(axis=0, skipna=False)
        # Keep only solvers with at least one non-NaN value
        pivot = pivot.dropna(how="all", axis=0)
        trimmed = greedy_biclique(pivot)
        # Drop all-tied after trim
        if trimmed.shape[1] > 0:
            var = trimmed.var(axis=0, skipna=False)
            trimmed = trimmed.loc[:, var > 0]
        complete = trimmed
        note_parts.append(f"sparse → greedy {complete.shape[0]}×{complete.shape[1]}")
    else:
        note_parts.append(f"strict {complete.shape[0]}×{complete.shape[1]}")

    if top_k is not None and complete.shape[0] > top_k:
        # Keep the k solvers with best mean rank (proxy: lowest mean of the
        # cleaned metric since lower=better). This keeps the diagram readable.
        keep = complete.mean(axis=1).nsmallest(top_k).index
        complete = complete.loc[keep]
        note_parts.append(f"top-{top_k} solvers")

    return complete, ", ".join(note_parts)


def cd_for_metric(df: pd.DataFrame, metric: str, ax: plt.Axes,
                  alpha: float = 0.05, top_k: int | None = None) -> str:
    """Render a CD diagram on `ax`. Returns a one-line status for stdout."""
    sub = df[["solver_name", "dataset_name", metric]].dropna(subset=[metric])
    if sub.empty:
        ax.text(0.5, 0.5, f"{metric}\nno data", ha="center", va="center",
                fontsize=10)
        ax.axis("off")
        return f"{metric}: no data"

    mat, note = prepare_matrix(sub, metric, top_k=top_k)
    k, n = mat.shape

    if k < 3 or n < 2:
        ax.text(0.5, 0.5,
                f"{metric}\nk={k}, N={n}\ninsufficient\n({note})",
                ha="center", va="center", fontsize=9)
        ax.axis("off")
        return f"{metric}: insufficient (k={k}, N={n})"

    # Rank within each dataset (lower=better → rank 1 = best)
    ranks_arr = np.vstack([rankdata(mat[c].values, method="average")
                           for c in mat.columns]).T
    ranks_df = pd.DataFrame(ranks_arr, index=mat.index, columns=mat.columns)
    mean_ranks = ranks_df.mean(axis=1).rename(short_solver)
    mean_ranks.index = [short_solver(s) for s in mean_ranks.index]

    # Friedman test
    chi2, pval = friedmanchisquare(*[mat.iloc[i].values for i in range(k)])

    # Nemenyi pairwise p-values: scikit_posthocs expects (N × k), so we
    # transpose ranks_df.
    nemenyi_input = ranks_df.T.copy()
    nemenyi_input.columns = mean_ranks.index
    sig = sp.posthoc_nemenyi_friedman(nemenyi_input.values)
    sig.index = mean_ranks.index
    sig.columns = mean_ranks.index

    # Critical difference (for the title — scikit_posthocs draws its own
    # crossbars from sig_matrix, so this is informational only).
    # q_α is the studentized range statistic divided by √2 at infinite df.
    # Demšar (2006) tabulates k≤30; for larger k, scipy.stats.studentized_range
    # extends the table.
    q_alpha = studentized_range.ppf(1.0 - alpha, k, np.inf) / math.sqrt(2.0)
    cd = q_alpha * math.sqrt(k * (k + 1) / (6.0 * n))

    plt.sca(ax)
    sp.critical_difference_diagram(
        ranks=mean_ranks,
        sig_matrix=sig,
        ax=ax,
        alpha=alpha,
        text_h_margin=0.005,
    )
    title = (f"{metric.removeprefix('objective_')}  |  "
             f"k={k}, N={n}, χ²={chi2:.1f}, p={pval:.1e}  |  "
             f"CD={cd:.2f}")
    ax.set_title(title, fontsize=10)
    return f"{metric}: k={k} N={n} chi2={chi2:.2f} p={pval:.2e} CD={cd:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--filter", default=None,
                        help="pandas query, e.g. \"p_dataset_problem_type=='binary'\"")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--ncols", type=int, default=2,
                        help="Columns in the subplot grid (default 2)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="If a metric leaves more than N solvers, keep the "
                             "top-N by mean metric (default: keep all)")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    if args.filter:
        before = len(df)
        df = df.query(args.filter)
        print(f"[info] filter '{args.filter}': {before} → {len(df)} rows")

    metrics = discover_metrics(df)
    if not metrics:
        print("[error] no numeric objective_* columns with data", file=sys.stderr)
        return 2
    print(f"[info] {len(metrics)} metrics found: {metrics}")

    ncols = args.ncols
    nrows = math.ceil(len(metrics) / ncols)
    fig_w = 8.0 * ncols
    fig_h = 4.5 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                             squeeze=False)
    axes_flat = axes.flatten()

    for i, metric in enumerate(metrics):
        status = cd_for_metric(df, metric, axes_flat[i],
                               alpha=args.alpha, top_k=args.top_k)
        print(f"  {status}")
    # Hide unused axes
    for j in range(len(metrics), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        f"Critical Difference Diagrams — {args.parquet.stem}",
        fontsize=14, y=1.0,
    )
    fig.tight_layout()
    out = args.out or args.parquet.with_name(f"cd_grid_{args.parquet.stem}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
