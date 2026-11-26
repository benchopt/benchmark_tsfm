"""General leaderboard — one table row per solver, columns:
  Rank | Solver | Global Elo | Forecasting Elo | Classification Elo | Anomaly Elo

Global Elo is computed by combining all three task types (task-aware pivot,
metrics normalised to lower-is-better). Per-task Elos are computed
independently on each task's datasets only.

All Elos use Bradley-Terry MLE + 200-round bootstrap 95% CI.
"""

from __future__ import annotations

import pandas as pd
from benchopt import BasePlot
from benchmark_utils.elo import ELO_ANCHOR, N_BOOTSTRAP, _elo_table
from benchmark_utils.metrics import is_higher_better
from benchmark_utils.plot_utils import (
    TASKS,
    elo_cell,
    mean_per_solver,
    relative_gmean,
    table_columns,
)

# Task → (results column, label) used for that task's Elo and metric cell:
# its primary metric. Direction comes from is_higher_better — never hardcoded.
_TASK_METRICS: dict[str, tuple[str, str]] = {
    key: metrics[0] for key, (_, metrics, _) in TASKS.items()
}


def _build_task_aware_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Combine all three task types into one solvers×datasets pivot.

    Only solvers that have results for ALL three tasks are included.
    """
    records: dict[str, dict[str, float]] = {}
    dataset_task: dict[str, str] = {}  # dataset → which task it belongs to
    for dataset_name, grp in df.groupby("dataset_name"):
        candidates = []
        for task_key, (col, _) in _TASK_METRICS.items():
            if col in grp.columns and grp[col].notna().any():
                candidates.append((task_key, col))
        if len(candidates) != 1:
            continue
        task, col = candidates[0]
        # Mean over repetitions, like the per-task leaderboards.
        values = grp.groupby("solver_name")[col].mean().dropna().to_dict()
        if len(values) < 2:
            continue
        # Elo treats lower as better, so flip higher-is-better metrics.
        if is_higher_better(col):
            values = {s: -v for s, v in values.items()}
        records[str(dataset_name)] = values
        dataset_task[str(dataset_name)] = task

    if not records:
        return pd.DataFrame()

    pivot = pd.DataFrame(records).T  # datasets × solvers
    pivot = pivot.dropna(axis=1, how="any").dropna(axis=0, how="any")

    # Filter: keep only solvers present in every task
    tasks_present = set(_TASK_METRICS.keys())
    solver_tasks: dict[str, set[str]] = {col: set() for col in pivot.columns}
    for dataset, task in dataset_task.items():
        if dataset in pivot.index:
            for solver in pivot.columns:
                if not pd.isna(pivot.loc[dataset, solver]):
                    solver_tasks[solver].add(task)

    all_tasks_solvers = [
        s for s, tasks in solver_tasks.items() if tasks >= tasks_present
    ]
    if not all_tasks_solvers:
        return pd.DataFrame()

    pivot = pivot[all_tasks_solvers]
    return pivot.T  # solvers × datasets


def _safe_elo(pivot: pd.DataFrame) -> pd.DataFrame | None:
    """Return _elo_table result or None if insufficient data."""
    if pivot.empty or pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return None
    table, _anchor = _elo_table(pivot)
    return table


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
            return [["(insufficient data)"] + ["—"] * (2 + len(_TASK_METRICS))]

        # Mean metric values per solver per task
        mean_metrics: dict[str, dict[str, float]] = {}
        for task, (col, _) in _TASK_METRICS.items():
            relative = TASKS[task][2]
            if col not in df.columns:
                mean_metrics[task] = {}
            elif relative:
                mean_metrics[task] = relative_gmean(df, col)
            else:
                mean_metrics[task] = mean_per_solver(df, col)

        rows = []
        for rank, row in enumerate(global_res.itertuples(index=False), start=1):
            solver = row.solver
            global_cell = elo_cell(row.elo, row.ci_low, row.ci_high)
            metric_cells = []
            for task in _TASK_METRICS:
                mean_val = mean_metrics[task].get(solver)
                if mean_val is None:
                    metric_cells.append("—")
                else:
                    metric_cells.append(mean_val)
            rows.append([str(rank), solver, global_cell] + metric_cells)

        return rows

    def get_metadata(self, df):
        columns = table_columns("Global Elo", _TASK_METRICS.values())
        return {
            "title": (
                f"General leaderboard — Global Elo + mean metrics per task "
                f"(bootstrap N={N_BOOTSTRAP}, anchored at {ELO_ANCHOR:.0f})"
            ),
            "columns": columns,
        }
