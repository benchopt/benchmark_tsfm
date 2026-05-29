"""General leaderboard — one table row per solver, columns:
  Rank | Solver | Global Elo | Forecasting Elo | Classification Elo | Anomaly Elo

Global Elo is computed by combining all three task types (task-aware pivot,
metrics normalised to lower-is-better). Per-task Elos are computed
independently on each task's datasets only.

All Elos use Bradley-Terry MLE + 200-round bootstrap 95% CI.
"""

from __future__ import annotations

import sys
import os

# elo.py lives in the same plots/ directory; add it to sys.path so we can
# import its shared helpers without duplicating code.
sys.path.insert(0, os.path.dirname(__file__))
from elo import (  # noqa: E402
    _elo_table, _short, ANCHOR_PREFERENCES,
    ELO_ANCHOR, N_BOOTSTRAP, is_higher_better,
)

import numpy as np
import pandas as pd

from benchopt import BasePlot


# Task → results column used for that task's Elo and metric cell. Direction
# comes from is_higher_better — never hardcoded here.
_TASK_METRICS: dict[str, str] = {
    "forecasting":       "objective_wql",
    "classification":    "objective_balanced_accuracy",
    "anomaly_detection": "objective_auc_pr",
}

# Task → human-readable column header label.
_TASK_LABELS: dict[str, str] = {
    "forecasting":       "rWQL",
    "classification":    "Bal. Acc",
    "anomaly_detection": "AUC-PR",
}

_LOW_COLOUR  = "#ca8a04"
_HIGH_COLOUR = "#16a34a"


def _metric_cell(mean_val: float) -> str:
    """Format a mean metric value for a table cell."""
    return f'<div style="font-weight:500">{mean_val:.4f}</div>'


def _elo_cell(elo: float, ci_low: float, ci_high: float) -> str:
    low_d  = ci_low  - elo
    high_d = ci_high - elo
    return (
        f'<div style="line-height:1.25">'
        f'<div style="font-weight:600">{elo:.0f}</div>'
        f'<div style="font-size:0.85em">'
        f'<span style="color:{_LOW_COLOUR}">{low_d:+.0f}</span>&nbsp;'
        f'<span style="color:{_HIGH_COLOUR}">{high_d:+.0f}</span>'
        f'</div></div>'
    )


def _build_task_aware_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Combine all three task types into one solvers×datasets pivot."""
    records: dict[str, dict[str, float]] = {}
    for dataset_name, grp in df.groupby("dataset_name"):
        candidates = []
        for col in _TASK_METRICS.values():
            if col in grp.columns and grp[col].notna().any():
                candidates.append(col)
        if len(candidates) != 1:
            continue
        col = candidates[0]
        rows = grp[["solver_name", col]].dropna(subset=[col])
        if rows["solver_name"].duplicated().any() or len(rows) < 2:
            continue
        values = dict(zip(rows["solver_name"], rows[col]))
        # Elo treats lower as better, so flip higher-is-better metrics.
        if is_higher_better(col):
            values = {s: -v for s, v in values.items()}
        records[str(dataset_name)] = values
    if not records:
        return pd.DataFrame()
    pivot = pd.DataFrame(records).T
    pivot = pivot.dropna(axis=1, how="any").dropna(axis=0, how="any")
    return pivot.T  # solvers × datasets


def _safe_elo(pivot: pd.DataFrame) -> pd.DataFrame | None:
    """Return _elo_table result or None if insufficient data."""
    if pivot.empty or pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return None
    table, _anchor = _elo_table(pivot)
    return table


def _mean_metric(df: pd.DataFrame, col: str) -> dict[str, float]:
    """Mean of ``col`` per solver (ignoring NaN rows), keyed by short name."""
    rows = df[["solver_name", col]].dropna(subset=[col]).copy()
    rows["solver_name"] = rows["solver_name"].map(_short)
    return rows.groupby("solver_name")[col].mean().to_dict()


def _relative_wql_gmean(df: pd.DataFrame, col: str) -> dict[str, float]:
    """Geometric mean of (solver_wql / seasonal_naive_wql) across datasets.

    For each dataset we divide every solver's WQL by the Seasonal Naive WQL
    on that same dataset, then take the geometric mean across datasets.
    Values < 1 = better than Seasonal Naive.
    """
    rows = df[["solver_name", "dataset_name", col]].dropna(subset=[col]).copy()
    rows["solver_name"] = rows["solver_name"].map(_short)

    # Find the seasonal naive row name (short)
    all_solvers = rows["solver_name"].unique().tolist()
    baseline = None
    for needle in ANCHOR_PREFERENCES:
        for s in all_solvers:
            if needle in s.lower():
                baseline = s
                break
        if baseline:
            break
    if baseline is None:
        # Fall back to plain mean if no naive baseline present
        return rows.groupby("solver_name")[col].mean().to_dict()

    pivot = rows.pivot_table(
        index="dataset_name", columns="solver_name", values=col, aggfunc="mean"
    )
    # Only keep datasets where baseline has a value
    pivot = pivot.dropna(subset=[baseline])
    if pivot.empty:
        return {}

    # Relative WQL per dataset
    rel = pivot.div(pivot[baseline], axis=0)

    # Geometric mean = exp(mean(log(rel))), skip non-positive values
    result = {}
    for solver in rel.columns:
        vals = rel[solver].dropna()
        vals = vals[vals > 0]
        if len(vals) == 0:
            continue
        result[solver] = float(np.exp(np.log(vals).mean()))
    return result


class Plot(BasePlot):
    """Global leaderboard: Global Elo + per-task Elo columns in one table."""

    name = "General leaderboard"
    type = "table"
    options = {}

    def plot(self, df):
        # --- Global (task-aware) Elo ---
        global_pivot = _build_task_aware_pivot(df)
        global_res = _safe_elo(global_pivot)

        if global_res is None:
            return [["(insufficient data)", "—", "—", "—", "—"]]

        # Mean metric values per solver per task
        mean_metrics: dict[str, dict[str, float]] = {}
        for task, col in _TASK_METRICS.items():
            if col not in df.columns:
                mean_metrics[task] = {}
            elif task == "forecasting":
                mean_metrics[task] = _relative_wql_gmean(df, col)
            else:
                mean_metrics[task] = _mean_metric(df, col)

        rows = []
        for rank, row in enumerate(global_res.itertuples(index=False), start=1):
            solver = row.solver
            global_cell = _elo_cell(row.elo, row.ci_low, row.ci_high)
            metric_cells = []
            for task in _TASK_METRICS:
                mean_val = mean_metrics[task].get(solver)
                if mean_val is None:
                    metric_cells.append("—")
                else:
                    metric_cells.append(_metric_cell(mean_val))
            rows.append([str(rank), solver, global_cell] + metric_cells)

        return rows

    def get_metadata(self, df):
        columns = ["Rank", "Solver", "Global Elo"] + [
            f"{_TASK_LABELS[task]} {'↑' if is_higher_better(col) else '↓'}"
            for task, col in _TASK_METRICS.items()
        ]
        return {
            "title": (
                f"General leaderboard — Global Elo + mean metrics per task "
                f"(bootstrap N={N_BOOTSTRAP}, anchored at {ELO_ANCHOR:.0f})"
            ),
            "columns": columns,
        }
