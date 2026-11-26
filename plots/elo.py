"""Elo leaderboard custom plot for benchopt — table format with bootstrap 95% CI.
Registered as a benchopt custom plot — appears in the
HTML "Chart type" dropdown. Switch metrics via ``objective_column``.
"""

from __future__ import annotations

from benchmark_utils.elo import (
    ELO_ANCHOR,
    N_BOOTSTRAP,
    _elo_table,
    _find_anchor_idx,
)
from benchmark_utils.metrics import is_higher_better
from benchmark_utils.plot_utils import elo_scores, pivot
from benchopt import BasePlot


class Plot(BasePlot):
    """Elo leaderboard with bootstrap 95% CI — as a benchopt bar chart.

    Appears in the HTML report's "Chart type" dropdown. ``objective_column``
    is auto-populated with every numeric ``objective_*`` column in the
    parquet, switching the chart in place.

    Visual encoding
    ---------------
    Benchopt's bar_chart uses ``median(y)`` as the bar height and renders the
    individual ``y`` values as scatter points on top. We exploit that by
    passing ``y = [ci_low, elo, ci_high]`` per solver:

      • The bar reaches the Elo **point estimate** (the median of those three).
      • Two extra dots appear at the bootstrap 95% **CI bounds**, giving the
        candle-style low/high markers the user asked for.

    Bars are sorted by Elo descending so the leader sits at the left.

    Direction
    ---------
    Per-metric direction comes from ``benchmark_utils.metrics.is_higher_better``
    (the single source of truth, re-exported by the objective). Lower-is-better
    metrics (FEV's SQL/MASE/WAPE/WQL and most forecasting losses) are used as-is;
    higher-is-better metrics (accuracy, AUC, F1, …) are negated before the
    pairwise tally so that the win logic stays lower-is-better throughout.
    """

    name = "Elo"
    type = "bar_chart"
    options = {
        "objective_column": ...,
    }

    def plot(self, df, objective_column):
        # Lower-is-better scores on the datasets usable for Elo, see
        # benchmark_utils.plot_utils.elo_scores.
        scores = elo_scores(df, objective_column)
        if scores is None:
            return [{"y": [0.0], "label": "(insufficient data)", "text": ""}]

        table, _ = _elo_table(scores)
        bars = []
        for row in table.itertuples(index=False):
            bars.append(
                {
                    # y = [ci_low, elo, ci_high] → bar height = median = elo,
                    # benchopt renders the individual values as horizontal
                    # line-ew-open markers (the "candle" low/elo/high ticks).
                    # `text` must be empty for those markers to render — see
                    # benchopt/plotting/html/static/result.js:185-204.
                    "y": [row.ci_low, row.elo, row.ci_high],
                    "label": row.solver,
                    "text": "",
                    **self.get_style(row.solver),
                }
            )
        return bars

    def get_metadata(self, df, objective_column):
        # Re-derive the anchor name from the same data the plot() call saw
        # so the title accurately reflects which solver sits at 1000.
        anchor_name = None
        scores = pivot(df, objective_column)
        scores = scores.loc[:, scores.notna().all(axis=0)]
        if scores.shape[0] >= 2:
            _, anchor_name = _find_anchor_idx(list(scores.index))

        if anchor_name:
            anchor_desc = f"{anchor_name} = {ELO_ANCHOR:.0f}"
        else:
            anchor_desc = f"mean Elo = {ELO_ANCHOR:.0f}"
        direction = (
            "higher better" if is_higher_better(objective_column) else "lower better"
        )
        return {
            "title": (
                f"Elo leaderboard — {objective_column} "
                f"({direction}, bootstrap N={N_BOOTSTRAP}, {anchor_desc}). "
                f"Bar = Elo; dots = 95% CI low / high."
            ),
            "ylabel": "Elo",
        }
