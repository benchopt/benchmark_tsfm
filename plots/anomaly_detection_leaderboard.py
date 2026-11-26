"""Anomaly detection leaderboard — one row per solver, one column per metric.

Columns: Rank | Solver | Elo ↑ | AUC-PR ↑ | AUC-ROC ↑ | F1-PA ↑

Elo is computed via Bradley-Terry MLE on AUC-PR (negated internally since
higher-is-better), mean-anchored at 1000. All metric columns are mean across
datasets. Sorted by Elo descending.
"""

from benchopt import BasePlot

from benchmark_utils.plot_utils import TASKS, leaderboard_rows, table_columns

# (column, display label), the first one ranks by Elo. Direction comes from
# is_higher_better — never hardcoded here.
_METRICS = TASKS["anomaly_detection"][1]


class Plot(BasePlot):
    """Anomaly detection leaderboard: Elo + all metrics, sorted by Elo."""

    name = "Anomaly detection leaderboard"
    type = "table"
    options = {}

    def plot(self, df):
        return leaderboard_rows(df, "anomaly_detection")

    def get_metadata(self, df):
        columns = table_columns("Elo ↑", _METRICS)
        return {
            "title": "Anomaly detection leaderboard — Elo (BT-MLE on AUC-PR) + mean "
            "metrics across datasets",
            "columns": columns,
        }
