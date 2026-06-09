"""Anomaly detection leaderboard — one row per solver, one column per metric.

Columns: Rank | Solver | Elo ↑ | AUC-PR ↑ | AUC-ROC ↑ | F1-PA ↑

Elo is computed via Bradley-Terry MLE on AUC-PR (negated internally since
higher-is-better), mean-anchored at 1000. All metric columns are mean across
datasets. Sorted by Elo descending.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from benchopt import BasePlot
from elo import _elo_table, _short, is_higher_better  # noqa: E402

_ELO_COL = "objective_auc_pr"

# (column, display label). Direction comes from is_higher_better — never
# hardcoded here.
_METRICS: list[tuple[str, str]] = [
    ("objective_auc_pr", "AUC-PR"),
    ("objective_auc_roc", "AUC-ROC"),
    ("objective_f1_pa", "F1-PA"),
]
_LOW_COLOUR = "#ca8a04"
_HIGH_COLOUR = "#16a34a"


def _elo_cell(elo: float, ci_low: float, ci_high: float) -> str:
    low_d = ci_low - elo
    high_d = ci_high - elo
    return (
        f'<div style="line-height:1.25">'
        f'<div style="font-weight:600">{elo:.0f}</div>'
        f'<div style="font-size:0.85em">'
        f'<span style="color:{_LOW_COLOUR}">{low_d:+.0f}</span>&nbsp;'
        f'<span style="color:{_HIGH_COLOUR}">{high_d:+.0f}</span>'
        f"</div></div>"
    )


def _build_elo(df: pd.DataFrame, col: str) -> pd.DataFrame | None:
    if col not in df.columns:
        return None
    rows = df[["solver_name", "dataset_name", col]].dropna(subset=[col])
    pivot = rows.pivot_table(
        index="solver_name", columns="dataset_name", values=col, aggfunc="mean"
    )
    pivot = pivot.loc[:, pivot.notna().all(axis=0)]
    if pivot.shape[1] > 0:
        pivot = pivot.loc[:, pivot.var(axis=0, skipna=False) > 0]
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return None
    if is_higher_better(col):
        pivot = -pivot
    table, _ = _elo_table(pivot)
    return table


def _mean_per_solver(df: pd.DataFrame, col: str) -> dict[str, float]:
    rows = df[["solver_name", col]].dropna(subset=[col]).copy()
    rows["solver_name"] = rows["solver_name"].map(_short)
    return rows.groupby("solver_name")[col].mean().to_dict()


class Plot(BasePlot):
    """Anomaly detection leaderboard: Elo + all metrics, sorted by Elo."""

    name = "Anomaly detection leaderboard"
    type = "table"
    options = {}

    def plot(self, df):
        ad_cols = [c for c, _ in _METRICS if c in df.columns]
        if not ad_cols:
            return [["(no anomaly detection data)", *["—"] * (len(_METRICS) + 1)]]

        elo_res = _build_elo(df, _ELO_COL)

        metric_maps = {
            col: (_mean_per_solver(df, col) if col in df.columns else {})
            for col, _ in _METRICS
        }

        all_solvers: set[str] = set()
        if elo_res is not None:
            all_solvers.update(elo_res["solver"])
        for m in metric_maps.values():
            all_solvers.update(m.keys())

        elo_lookup = (
            {
                row.solver: (row.elo, row.ci_low, row.ci_high)
                for row in elo_res.itertuples(index=False)
            }
            if elo_res is not None
            else {}
        )

        def sort_key(s: str):
            if s in elo_lookup:
                return -elo_lookup[s][0]  # negate → descending
            return metric_maps.get(_ELO_COL, {}).get(s, -float("inf"))

        sorted_solvers = sorted(all_solvers, key=sort_key)

        rows = []
        for rank, solver in enumerate(sorted_solvers, start=1):
            elo_entry = elo_lookup.get(solver)
            elo_cell = _elo_cell(*elo_entry) if elo_entry else "—"
            cells = []
            for col, _label in _METRICS:
                val = metric_maps[col].get(solver)
                cells.append(
                    f'<span style="font-weight:500">{val:.4f}</span>'
                    if val is not None
                    else "—"
                )
            rows.append([str(rank), solver, elo_cell] + cells)

        return rows

    def get_metadata(self, df):
        columns = ["Rank", "Solver", "Elo ↑"] + [
            f"{label} {'↑' if is_higher_better(col) else '↓'}"
            for col, label in _METRICS
        ]
        return {
            "title": "Anomaly detection leaderboard — Elo (BT-MLE on AUC-PR) + mean "
            "metrics across datasets",
            "columns": columns,
        }
