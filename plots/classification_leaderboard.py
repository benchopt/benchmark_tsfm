"""Classification leaderboard — one row per solver, one column per metric.

Columns: Rank | Solver | Bal. Acc ↑ | Accuracy ↑ | F1 (weighted) ↑

All values are mean across datasets. Sorted by Balanced Accuracy descending.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from elo import _short, is_higher_better  # noqa: E402

import pandas as pd

from benchopt import BasePlot


# (column, display label). Direction comes from is_higher_better — never
# hardcoded here.
_METRICS: list[tuple[str, str]] = [
    ("objective_balanced_accuracy", "Bal. Acc"),
    ("objective_accuracy",          "Accuracy"),
    ("objective_f1_weighted",       "F1 weighted"),
]


def _mean_per_solver(df: pd.DataFrame, col: str) -> dict[str, float]:
    rows = df[["solver_name", col]].dropna(subset=[col]).copy()
    rows["solver_name"] = rows["solver_name"].map(_short)
    return rows.groupby("solver_name")[col].mean().to_dict()


class Plot(BasePlot):
    """Classification leaderboard: all classification metrics, sorted by Bal. Acc."""

    name = "Classification leaderboard"
    type = "table"
    options = {}

    def plot(self, df):
        clf_cols = [c for c, _ in _METRICS if c in df.columns]
        if not clf_cols:
            return [["(no classification data)", *["—"] * len(_METRICS)]]

        metric_maps: dict[str, dict[str, float]] = {
            col: (_mean_per_solver(df, col) if col in df.columns else {})
            for col, _ in _METRICS
        }

        all_solvers: set[str] = set()
        for m in metric_maps.values():
            all_solvers.update(m.keys())

        # Sort by the first metric, best solver first (direction-aware).
        primary = _METRICS[0][0]
        primary_higher = is_higher_better(primary)
        worst = -float("inf") if primary_higher else float("inf")

        def sort_key(s: str) -> float:
            v = metric_maps[primary].get(s)
            return v if v is not None else worst

        sorted_solvers = sorted(all_solvers, key=sort_key, reverse=primary_higher)

        rows = []
        for rank, solver in enumerate(sorted_solvers, start=1):
            cells = []
            for col, _label in _METRICS:
                val = metric_maps[col].get(solver)
                cells.append(
                    f'<span style="font-weight:500">{val:.4f}</span>'
                    if val is not None else "—"
                )
            rows.append([str(rank), solver] + cells)

        return rows

    def get_metadata(self, df):
        columns = (
            ["Rank", "Solver"]
            + [f"{label} {'↑' if is_higher_better(col) else '↓'}"
               for col, label in _METRICS]
        )
        return {
            "title": "Classification leaderboard — mean metrics across datasets",
            "columns": columns,
        }
