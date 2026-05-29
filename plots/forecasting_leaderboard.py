"""Forecasting leaderboard — one row per solver, one column per metric.

All metrics are relative to Seasonal Naive: geometric mean of
(solver_metric / seasonal_naive_metric) across datasets.
Values < 1 = better than Seasonal Naive.

Columns: Rank | Solver | rWQL ↓ | rSQL ↓ | rMASE ↓ | rWAPE ↓ | rMAE ↓ | rMSE ↓ | rSMAPE ↓

Sorted by rWQL ascending. Falls back to the first available metric if WQL absent.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from elo import _short, ANCHOR_PREFERENCES  # noqa: E402

import numpy as np
import pandas as pd

from benchopt import BasePlot


_METRICS: list[tuple[str, str]] = [
    ("objective_wql",   "rWQL"),
    ("objective_sql",   "rSQL"),
    ("objective_mase",  "rMASE"),
    ("objective_wape",  "rWAPE"),
    ("objective_mae",   "rMAE"),
    ("objective_mse",   "rMSE"),
    ("objective_smape", "rSMAPE"),
]


def _find_baseline(solvers: list[str]) -> str | None:
    for needle in ANCHOR_PREFERENCES:
        for s in solvers:
            if needle in s.lower():
                return s
    return None


def _relative_gmean(df: pd.DataFrame, col: str) -> dict[str, float] | None:
    """Geometric mean of (solver / baseline) per dataset for one metric."""
    if col not in df.columns:
        return None

    rows = df[["solver_name", "dataset_name", col]].dropna(subset=[col]).copy()
    rows["solver_name"] = rows["solver_name"].map(_short)

    baseline = _find_baseline(rows["solver_name"].unique().tolist())
    if baseline is None:
        # Fall back: mean across datasets (no relativisation)
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
    """Forecasting leaderboard: all metrics relative to Seasonal Naive."""

    name = "Forecasting leaderboard"
    type = "table"
    options = {}

    def plot(self, df):
        forecast_cols = [c for c, _ in _METRICS if c in df.columns]
        if not forecast_cols:
            return [["(no forecasting data)", *["—"] * len(_METRICS)]]

        metric_maps: dict[str, dict[str, float] | None] = {
            col: _relative_gmean(df, col) for col, _ in _METRICS
        }

        all_solvers: set[str] = set()
        for m in metric_maps.values():
            if m:
                all_solvers.update(m.keys())

        # Sort by first available metric ascending (lower relative = better)
        primary = next((c for c, _ in _METRICS if metric_maps.get(c)), None)

        def sort_key(s: str) -> float:
            if primary and metric_maps[primary]:
                return metric_maps[primary].get(s, float("inf"))
            return float("inf")

        sorted_solvers = sorted(all_solvers, key=sort_key)

        rows = []
        for rank, solver in enumerate(sorted_solvers, start=1):
            cells = []
            for col, _ in _METRICS:
                m = metric_maps[col]
                val = m.get(solver) if m else None
                cells.append(
                    f'<span style="font-weight:500">{val:.4f}</span>'
                    if val is not None else "—"
                )
            rows.append([str(rank), solver] + cells)

        return rows

    def get_metadata(self, df):
        columns = ["Rank", "Solver"] + [f"{label} ↓" for _, label in _METRICS]
        return {
            "title": (
                "Forecasting leaderboard — geometric mean of "
                "(solver metric / Seasonal Naive metric) across datasets"
            ),
            "columns": columns,
        }
