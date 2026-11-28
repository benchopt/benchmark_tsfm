"""Classification leaderboard — one row per solver, one column per metric.

Columns: Rank | Solver | Elo ↑ | Bal. Acc ↑ | Accuracy ↑ | F1 weighted ↑

Elo is computed via Bradley-Terry MLE on Balanced Accuracy (negated internally
since higher-is-better), mean-anchored at 1000. All metric columns are mean
across datasets. Sorted by Elo descending.
"""

from benchopt import BasePlot

from benchmark_utils.plot_utils import TASKS, leaderboard_rows, table_columns

# (column, display label), the first one ranks by Elo. Direction comes from
# is_higher_better — never hardcoded here.
_METRICS = TASKS["classification"][1]


class Plot(BasePlot):
    """Classification leaderboard: Elo + all metrics, sorted by Elo."""

    name = "Classification leaderboard"
    type = "table"
    options = {}

    def plot(self, df):
        return leaderboard_rows(df, "classification")

    def get_metadata(self, df):
        columns = table_columns("Elo ↑", _METRICS)
        return {
            "title": "Classification leaderboard — Elo (BT-MLE on Bal. Acc) + mean "
            "metrics across datasets",
            "columns": columns,
        }
