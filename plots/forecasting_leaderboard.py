"""Forecasting leaderboard — one row per solver, one column per metric.

Columns: Rank | Solver | Elo | rWQL ↓ | rSQL ↓ | rMASE ↓ | rWAPE ↓ | rMAE ↓ | rMSE ↓ | rSMAPE ↓

Elo is computed via Bradley-Terry MLE on WQL (lower-is-better), anchored at
Seasonal Naive = 1000.  All other metric columns are the geometric mean of
(solver / Seasonal Naive) across datasets (relative, lower = better).

Sorted by Elo descending.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from benchopt import BasePlot
from elo import ANCHOR_PREFERENCES, _elo_table, _short, is_higher_better  # noqa: E402

# Primary metric used for Elo ranking
_ELO_COL = "objective_wql"

_METRICS: list[tuple[str, str]] = [
    ("objective_wql", "rWQL"),
    ("objective_sql", "rSQL"),
    ("objective_mase", "rMASE"),
    ("objective_wape", "rWAPE"),
    ("objective_mae", "rMAE"),
    ("objective_mse", "rMSE"),
    ("objective_smape", "rSMAPE"),
]

_LOW_COLOUR = "#ca8a04"
_HIGH_COLOUR = "#16a34a"


def _find_baseline(solvers: list[str]) -> str | None:
    for needle in ANCHOR_PREFERENCES:
        for s in solvers:
            if needle in s.lower():
                return s
    return None


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
    """Build Elo table from a single metric column."""
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
    return table  # columns: solver, elo, ci_low, ci_high, games


def _relative_gmean(df: pd.DataFrame, col: str) -> dict[str, float] | None:
    """Geometric mean of (solver / baseline) per dataset for one metric."""
    if col not in df.columns:
        return None
    rows = df[["solver_name", "dataset_name", col]].dropna(subset=[col]).copy()
    rows["solver_name"] = rows["solver_name"].map(_short)
    baseline = _find_baseline(rows["solver_name"].unique().tolist())
    if baseline is None:
        return rows.groupby("solver_name")[col].mean().to_dict()
    pivot = rows.pivot_table(
        index="dataset_name", columns="solver_name", values=col, aggfunc="mean"
    ).dropna(subset=[baseline])
    if pivot.empty:
        return None
    rel = pivot.div(pivot[baseline], axis=0)
    result = {}
    for solver in rel.columns:
        vals = rel[solver].dropna()
        vals = vals[vals > 0]
        if len(vals) > 0:
            result[solver] = float(np.exp(np.log(vals).mean()))
    return result


class Plot(BasePlot):
    """Forecasting leaderboard: Elo + all relative metrics, sorted by Elo."""

    name = "Forecasting leaderboard"
    type = "table"
    options = {}

    def plot(self, df):
        forecast_cols = [c for c, _ in _METRICS if c in df.columns]
        if not forecast_cols:
            return [["(no forecasting data)", *["—"] * (len(_METRICS) + 1)]]

        elo_res = _build_elo(df, _ELO_COL)

        metric_maps = {col: _relative_gmean(df, col) for col, _ in _METRICS}

        # All solvers seen in any metric
        all_solvers: set[str] = set()
        if elo_res is not None:
            all_solvers.update(elo_res["solver"])
        for m in metric_maps.values():
            if m:
                all_solvers.update(m.keys())

        # Sort by Elo descending; fall back to first relative metric ascending
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
            primary = next((c for c, _ in _METRICS if metric_maps.get(c)), None)
            if primary:
                return metric_maps[primary].get(s, float("inf"))
            return float("inf")

        sorted_solvers = sorted(all_solvers, key=sort_key)

        rows = []
        for rank, solver in enumerate(sorted_solvers, start=1):
            elo_entry = elo_lookup.get(solver)
            elo_cell = _elo_cell(*elo_entry) if elo_entry else "—"
            cells = []
            for col, _ in _METRICS:
                m = metric_maps[col]
                val = m.get(solver) if m else None
                cells.append(
                    f'<span style="font-weight:500">{val:.4f}</span>'
                    if val is not None
                    else "—"
                )
            rows.append([str(rank), solver, elo_cell] + cells)

        return rows

    def get_metadata(self, df):
        columns = ["Rank", "Solver", "Elo ↑"] + [f"{label} ↓" for _, label in _METRICS]
        return {
            "title": (
                "Forecasting leaderboard — Elo (BT-MLE on WQL, Seasonal Naive = 1000) "
                "+ geometric mean of (solver / Seasonal Naive) per metric"
            ),
            "columns": columns,
        }
