"""Forecasting leaderboard — one row per solver, one column per metric.

Columns: Rank | Solver | Elo | rWQL ↓ | rSQL ↓ | rMASE ↓ | rWAPE ↓ | rMAE ↓ |
rMSE ↓ | rSMAPE ↓

Elo is computed via Bradley-Terry MLE on WQL (lower-is-better), anchored at
Seasonal Naive = 1000.  All other metric columns are the geometric mean of
(solver / Seasonal Naive) across datasets (relative, lower = better).

Sorted by Elo descending.
"""

from benchopt import BasePlot

from benchmark_utils.plot_utils import TASKS, leaderboard_rows, table_columns

# (column, display label), the first one ranks by Elo. Direction comes from
# is_higher_better — never hardcoded here.
_METRICS = TASKS["forecasting"][1]


class Plot(BasePlot):
    """Forecasting leaderboard: Elo + all relative metrics, sorted by Elo."""

    name = "Forecasting leaderboard"
    type = "table"
    options = {}

    def plot(self, df):
        return leaderboard_rows(df, "forecasting")

    def get_metadata(self, df):
        columns = table_columns("Elo ↑", _METRICS)
        return {
            "title": (
                "Forecasting leaderboard — Elo (BT-MLE on WQL, Seasonal Naive = 1000) "
                "+ geometric mean of (solver / Seasonal Naive) per metric"
            ),
            "columns": columns,
        }
