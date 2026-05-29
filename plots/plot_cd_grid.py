"""Critical Difference diagrams for a benchopt-schema parquet.

Two front-ends share the same Demšar (2006) computation:

1. **CLI** — bundles every numeric ``objective_*`` column into a single
   subplot grid PNG. Run from the repo root::

       python plots/plot_cd_grid.py <parquet> [--out PATH] [--filter QUERY]
                                    [--ncols N] [--top-k N]

2. **benchopt custom plot** — the ``Plot`` class at the bottom of this file
   makes the CD diagram appear under the "Chart type" dropdown of the
   benchopt HTML report (``benchopt plot . --html``). One subplot per
   metric, switched via the ``objective_column`` selector.

Uses ``scipy.stats.friedmanchisquare`` + ``scikit_posthocs.posthoc_nemenyi_friedman``
+ ``scikit_posthocs.critical_difference_diagram``. For k>30 the critical
difference is computed from ``scipy.stats.studentized_range`` instead of
the tabulated Demšar values.
"""

from __future__ import annotations

import argparse
import io
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

from benchopt import BasePlot

sys.path.insert(0, str(Path(__file__).parent))
from elo import HIGHER_IS_BETTER  # noqa: E402


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
        # Keep the k solvers with the best mean rank.
        # For higher-is-better metrics use nlargest; lower-is-better nsmallest.
        means = complete.mean(axis=1)
        if metric in HIGHER_IS_BETTER:
            keep = means.nlargest(top_k).index
        else:
            keep = means.nsmallest(top_k).index
        complete = complete.loc[keep]
        note_parts.append(f"top-{top_k} solvers")

    return complete, ", ".join(note_parts)


GLOBAL_KEY = "global"


def _cd_global(df: pd.DataFrame, ax: plt.Axes, alpha: float = 0.05) -> str:
    """Render the global CD diagram across every numeric ``objective_*`` metric.

    For each metric we build per-dataset ranks (respecting HIGHER_IS_BETTER),
    intersect the solver set across metrics, then concatenate the (k × N_metric)
    rank matrices along axis 1. Each (metric, dataset) pair becomes one
    Friedman "block" of size k. Mean rank per solver, Nemenyi pairwise p-values,
    and the Demšar CD overlay are computed on that stacked rank matrix.
    """
    metrics = discover_metrics(df)
    if not metrics:
        ax.text(0.5, 0.5, "global\nno objective_* columns",
                ha="center", va="center", fontsize=10)
        ax.axis("off")
        return "global: no metrics"

    per_metric_ranks: list[pd.DataFrame] = []
    used_metrics: list[str] = []
    for metric in metrics:
        sub = df[["solver_name", "dataset_name", metric]].dropna(subset=[metric])
        if sub.empty:
            continue
        mat, _note = prepare_matrix(sub, metric, top_k=None)
        if mat.shape[0] < 2 or mat.shape[1] < 1:
            continue
        # Higher-is-better → negate so rank 1 == best in both regimes
        rank_input = -mat if metric in HIGHER_IS_BETTER else mat
        ranks_arr = np.vstack([
            rankdata(rank_input[c].values, method="average")
            for c in rank_input.columns
        ]).T
        ranks = pd.DataFrame(
            ranks_arr, index=mat.index,
            columns=[f"{metric}|{c}" for c in mat.columns],
        )
        per_metric_ranks.append(ranks)
        used_metrics.append(metric)

    if not per_metric_ranks:
        ax.text(0.5, 0.5, "global\nno usable metrics", ha="center",
                va="center", fontsize=10)
        ax.axis("off")
        return "global: no usable metrics"

    # Intersect solvers across all metrics so the stacked block matrix is
    # complete — Friedman requires complete blocks.
    common = set(per_metric_ranks[0].index)
    for r in per_metric_ranks[1:]:
        common &= set(r.index)
    common_sorted = sorted(common)

    if len(common_sorted) < 3:
        ax.text(0.5, 0.5,
                f"global\nonly {len(common_sorted)} solvers shared\n"
                f"across {len(used_metrics)} metrics",
                ha="center", va="center", fontsize=9)
        ax.axis("off")
        return f"global: only {len(common_sorted)} solvers in common"

    stacked = pd.concat([r.loc[common_sorted] for r in per_metric_ranks], axis=1)
    k, n_blocks = stacked.shape

    mean_ranks = stacked.mean(axis=1)
    mean_ranks.index = [short_solver(s) for s in mean_ranks.index]

    # Friedman on the stacked ranks. friedmanchisquare ranks within each
    # block; since we already supplied per-block ranks, the re-ranking is a
    # no-op (in expectation) and the chi^2 statistic comes out the same as
    # plugging into the closed-form formula.
    chi2, pval = friedmanchisquare(*[stacked.iloc[i].values for i in range(k)])

    sig = sp.posthoc_nemenyi_friedman(stacked.T.values)
    sig.index = mean_ranks.index
    sig.columns = mean_ranks.index

    q_alpha = studentized_range.ppf(1.0 - alpha, k, np.inf) / math.sqrt(2.0)
    cd = q_alpha * math.sqrt(k * (k + 1) / (6.0 * n_blocks))

    plt.sca(ax)
    sp.critical_difference_diagram(
        ranks=mean_ranks, sig_matrix=sig, ax=ax, alpha=alpha,
        text_h_margin=0.005,
    )

    # Same CD-bar overlay as the per-metric path
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    extra = 0.18 * (y_max - y_min)
    ax.set_ylim(y_min, y_max + extra)
    y_min, y_max = ax.get_ylim()
    bar_y   = y_max - 0.04 * (y_max - y_min)
    tick_h  = 0.015 * (y_max - y_min)
    bar_x0  = x_min + 0.03 * (x_max - x_min)
    bar_x1  = bar_x0 + cd
    cd_colour = "#555555"
    ax.plot([bar_x0, bar_x1], [bar_y, bar_y],
            color=cd_colour, linewidth=1.5, solid_capstyle="butt", zorder=5)
    for x in (bar_x0, bar_x1):
        ax.plot([x, x], [bar_y - tick_h, bar_y + tick_h],
                color=cd_colour, linewidth=1.5, zorder=5)
    ax.text((bar_x0 + bar_x1) / 2, bar_y + 2 * tick_h,
            f"CD={cd:.2f}", ha="center", va="bottom",
            fontsize=7, color=cd_colour, zorder=5)

    ax.set_title(
        f"CD diagram — GLOBAL "
        f"(avg rank across {len(used_metrics)} metrics, "
        f"{n_blocks} blocks, k={k})",
        fontsize=10,
    )
    return (f"global: k={k} blocks={n_blocks} "
            f"chi2={chi2:.2f} p={pval:.2e} CD={cd:.3f}")


def cd_for_metric(df: pd.DataFrame, metric: str, ax: plt.Axes,
                  alpha: float = 0.05, top_k: int | None = None) -> str:
    """Render a CD diagram on `ax`. Returns a one-line status for stdout."""
    if metric == GLOBAL_KEY:
        return _cd_global(df, ax, alpha=alpha)

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

    # Rank within each dataset.
    # For higher-is-better metrics, negate so rank 1 = highest value.
    rank_mat = -mat if metric in HIGHER_IS_BETTER else mat
    ranks_arr = np.vstack([rankdata(rank_mat[c].values, method="average")
                           for c in rank_mat.columns]).T
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

    # Expand the top of the y-axis to make room for the CD bar above the diagram
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    extra = 0.18 * (y_max - y_min)
    ax.set_ylim(y_min, y_max + extra)
    # Recompute limits after expansion
    y_min, y_max = ax.get_ylim()

    # --- CD reference bar (top-left corner, Demšar-style) ---
    bar_y   = y_max - 0.04 * (y_max - y_min)
    tick_h  = 0.015 * (y_max - y_min)
    bar_x0  = x_min + 0.03 * (x_max - x_min)
    bar_x1  = bar_x0 + cd
    cd_colour = "#555555"
    ax.plot([bar_x0, bar_x1], [bar_y, bar_y],
            color=cd_colour, linewidth=1.5, solid_capstyle="butt", zorder=5)
    for x in (bar_x0, bar_x1):
        ax.plot([x, x], [bar_y - tick_h, bar_y + tick_h],
                color=cd_colour, linewidth=1.5, zorder=5)
    ax.text((bar_x0 + bar_x1) / 2, bar_y + 2 * tick_h,
            f"CD={cd:.2f}", ha="center", va="bottom",
            fontsize=7, color=cd_colour, zorder=5)
    # -------------------------------------------------------

    title = (f"CD diagram for {metric.removeprefix('objective_').upper()} ")
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


# ---------------------------------------------------------------------------
# benchopt custom plot — picks up the same CD computation via the "Chart type"
# dropdown in the HTML report. One subplot per objective_* metric, selectable
# from the sidebar.
# ---------------------------------------------------------------------------

def _fig_to_array(fig) -> np.ndarray:
    """Render a matplotlib Figure to a (H, W, 3) float array in [0, 1].

    benchopt's image plot backend (`benchopt.plotting.image_utils._array_to_png_src`)
    requires a numpy/array-API object with values in [0, 1]; string data URIs
    are rejected despite what the docs imply.
    """
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


class Plot(BasePlot):
    """Critical Difference Diagram as a benchopt-native custom plot.

    Appears in the HTML report's "Chart type" dropdown. The
    ``objective_column`` option is auto-populated with every ``objective_*``
    column found in the parquet, plus a virtual ``"global"`` entry that
    averages per-dataset ranks across **all** metrics and runs Friedman +
    Nemenyi on the stacked block matrix.
    """

    name = "Critical Difference Diagram"
    type = "image"
    options = {
        "objective_column": ...,
    }

    def _get_all_plots(self, df):
        # Let benchopt do the standard ``...`` expansion first, then append
        # a synthetic "global" objective_column that the dropdown can select.
        # Both the options list (read by the HTML sidebar) and the plots dict
        # (looked up by `<plot_name>_<value>`) get the extra entry.
        plots, options = super()._get_all_plots(df)
        opts = list(options.get("objective_column", []))
        if GLOBAL_KEY in opts:
            return plots, options
        opts.append(GLOBAL_KEY)
        options["objective_column"] = opts

        data = self.get_metadata(df, objective_column=GLOBAL_KEY)
        data["type"] = self.type
        data["data"] = self.plot(df, objective_column=GLOBAL_KEY)
        key = "_".join([self._get_name(), GLOBAL_KEY])
        plots[key] = data
        return plots, options

    def plot(self, df, objective_column):
        fig, ax = plt.subplots(figsize=(10, 5))
        cd_for_metric(df, objective_column, ax)
        return [{"image": _fig_to_array(fig), "label": objective_column}]

    def get_metadata(self, df, objective_column):
        if objective_column == GLOBAL_KEY:
            title = ("Critical Difference Diagram — global "
                     "(mean rank across every objective_* metric)")
        else:
            title = f"Critical Difference Diagram — {objective_column}"
        return {"title": title, "ncols": 1}


if __name__ == "__main__":
    sys.exit(main())
