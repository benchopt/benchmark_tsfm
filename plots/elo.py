"""Elo leaderboard custom plot for benchopt — table format with bootstrap 95% CI.

Methodology follows TabArena (Erickson et al. 2025, arXiv:2506.16791) which
itself follows Chatbot Arena (Chiang et al. 2024):

- For each dataset, every pair of solvers plays one "game". Outcome is
  determined by the chosen ``objective_column`` (lower-is-better convention):
  i beats j on dataset d iff metric[i,d] < metric[j,d]; ties count as half-wins
  for each side.
- Ratings are fit by maximum-likelihood Bradley-Terry. The BT model is
  mathematically equivalent to Elo logistic regression:
  P(i beats j) = sigmoid(r_i - r_j). Log-ratings r are converted to Elo points
  via ``elo = r * 400 / ln(10) + 1000`` (the chess convention used by Chatbot
  Arena and TabArena: a 400-Elo gap → 91% win rate, mean rating anchored at
  1000).
- 95% confidence intervals come from 200 bootstrap rounds resampling datasets
  with replacement. CI = [2.5th, 97.5th] percentiles of bootstrapped Elo.

Registered as a benchopt custom plot — appears in the HTML "Chart type"
dropdown when this file is in ``plots/``. Switch metrics via ``objective_column``
in the sidebar.
"""

from __future__ import annotations

import math
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit

from benchopt import BasePlot

# Make the repo root importable so ``benchmark_utils`` resolves whether elo.py
# is loaded by benchopt (root already on sys.path) or imported by a sibling
# plot run standalone (only plots/ on sys.path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark_utils.metrics import is_higher_better  # noqa: E402


# Chess / Chatbot Arena / TabArena convention: 400-Elo gap = 91% win rate
ELO_SCALE = 400.0 / math.log(10)
ELO_ANCHOR = 1000.0  # Elo assigned to the anchor solver (see ANCHOR_PREFERENCES)
N_BOOTSTRAP = 200
EPS = 1e-9

# Reference baseline for Elo normalisation. We prefer "Seasonal Naive" — the
# canonical forecasting reference — but fall back to plain "Naive" if no
# seasonal variant is present, and finally to the mean-anchor if no naive
# baseline exists at all. TabArena anchors default RandomForest at 1000 for
# the same reason: a fixed, well-understood reference makes the Elo gap
# interpretable across metrics and across re-runs.
ANCHOR_PREFERENCES = ("seasonal naive", "seasonal_naive", "naive")


def _short(name: str) -> str:
    return re.sub(r"\[.*?\]$", "", str(name)).strip()


def _find_anchor_idx(solvers: list[str]) -> tuple[int | None, str | None]:
    """Pick the solver index used to anchor Elo at ELO_ANCHOR.

    Returns ``(index, short_name)`` of the matched solver, or ``(None, None)``
    if no preferred baseline is present (the caller falls back to mean-anchor).
    """
    shorts = [_short(s).lower() for s in solvers]
    for needle in ANCHOR_PREFERENCES:
        for i, name in enumerate(shorts):
            if needle in name:
                return i, _short(solvers[i])
    return None, None


def _pairwise_wins(mat: np.ndarray) -> np.ndarray:
    """Vectorised pairwise-wins tally.

    Parameters
    ----------
    mat : (k, N) array, lower-is-better metric values.

    Returns
    -------
    W : (k, k) float array. W[i, j] = # datasets where solver i beats j,
        with ties counted as 0.5 for each side. Diagonal is zero.
    """
    diff = mat[:, None, :] - mat[None, :, :]   # (k, k, N): mat[i] - mat[j]
    wins = (diff < 0).sum(axis=-1).astype(np.float64)
    ties = (diff == 0).sum(axis=-1).astype(np.float64) * 0.5
    W = wins + ties
    np.fill_diagonal(W, 0.0)
    return W


def _fit_bt(W: np.ndarray) -> np.ndarray:
    """Fit Bradley-Terry log-ratings by maximum likelihood.

    The BT log-likelihood of observed wins W is
        sum_{i,j} W[i,j] * log sigmoid(r_i - r_j).
    Location is unidentified, so we anchor at sum(r) = 0 after the fit.
    """
    k = W.shape[0]

    def nll(r):
        diff = r[:, None] - r[None, :]
        # log_expit is numerically stable for all inputs (no overflow).
        return -float((W * log_expit(diff)).sum())

    def grad(r):
        diff = r[:, None] - r[None, :]
        # d/dr_i log sigmoid(r_i - r_j) = sigmoid(r_j - r_i)
        p = expit(-diff)        # (k, k): expected loss probability of i vs j
        # gradient of -nll w.r.t r_i = sum_j W[i,j] * (1 - p_win[i,j])
        #                            - sum_j W[j,i] * p_win[j,i]
        # where p_win[i,j] = sigmoid(r_i - r_j) = 1 - p[i,j]
        g = (W * p).sum(axis=1) - (W.T * (1 - p.T)).sum(axis=1)
        return -g  # because we're minimizing nll

    r0 = np.zeros(k)
    res = minimize(nll, r0, jac=grad, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-9})
    r = res.x - res.x.mean()
    return r


def _elo_table(mat: pd.DataFrame, n_boot: int = N_BOOTSTRAP,
               seed: int = 0) -> tuple[pd.DataFrame, str | None]:
    """Compute per-solver Elo + 95% bootstrap CI.

    Returns ``(table, anchor_name)`` where ``table`` is a DataFrame sorted by
    Elo descending with columns (solver, elo, ci_low, ci_high, games), and
    ``anchor_name`` is the short name of the solver pinned at ``ELO_ANCHOR``
    (or ``None`` if mean-anchoring was used).
    """
    arr = mat.to_numpy(dtype=np.float64)
    k, n = arr.shape
    solvers = list(mat.index)
    anchor_idx, anchor_name = _find_anchor_idx(solvers)

    def _shift(r: np.ndarray) -> np.ndarray:
        # Translate the BT log-ratings so the anchor sits at zero. If no
        # anchor solver is present, the BT fit already centres at the mean.
        if anchor_idx is None:
            return r
        return r - r[anchor_idx]

    # Point estimate on the full data
    r_full = _shift(_fit_bt(_pairwise_wins(arr)))
    elo_full = r_full * ELO_SCALE + ELO_ANCHOR

    # Bootstrap CI by resampling datasets with replacement
    rng = np.random.default_rng(seed)
    boots = np.zeros((n_boot, k))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r_b = _shift(_fit_bt(_pairwise_wins(arr[:, idx])))
        boots[b] = r_b * ELO_SCALE + ELO_ANCHOR

    ci_low = np.percentile(boots, 2.5, axis=0)
    ci_high = np.percentile(boots, 97.5, axis=0)

    # Games per solver: each solver plays (k-1) opponents on each of n datasets
    games = np.full(k, (k - 1) * n, dtype=np.int64)

    out = pd.DataFrame({
        "solver": [_short(s) for s in solvers],
        "elo": elo_full,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "games": games,
    }).sort_values("elo", ascending=False).reset_index(drop=True)
    return out, anchor_name


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
        pivot = df.pivot_table(
            index="solver_name",
            columns="dataset_name",
            values=objective_column,
            aggfunc="mean",
        )
        # For higher-is-better metrics, negate so that _pairwise_wins
        # (which treats lower as better) computes wins correctly.
        if is_higher_better(objective_column):
            pivot = -pivot

        # Strict clean: drop datasets where any solver is NaN (Elo needs
        # complete games), then drop all-tied datasets (variance == 0).
        pivot = pivot.loc[:, pivot.notna().all(axis=0)]
        if pivot.shape[1] > 0:
            var = pivot.var(axis=0, skipna=False)
            pivot = pivot.loc[:, var > 0]

        k, n = pivot.shape
        if k < 2 or n < 2:
            return [{"y": [0.0], "label": "(insufficient data)", "text": ""}]

        table, _ = _elo_table(pivot)
        bars = []
        for row in table.itertuples(index=False):
            bars.append({
                # y = [ci_low, elo, ci_high] → bar height = median = elo,
                # benchopt renders the individual values as horizontal
                # line-ew-open markers (the "candle" low/elo/high ticks).
                # `text` must be empty for those markers to render — see
                # benchopt/plotting/html/static/result.js:185-204.
                "y": [row.ci_low, row.elo, row.ci_high],
                "label": row.solver,
                "text": "",
                **self.get_style(row.solver),
            })
        return bars

    def get_metadata(self, df, objective_column):
        # Re-derive the anchor name from the same data the plot() call saw
        # so the title accurately reflects which solver sits at 1000.
        anchor_name = None
        pivot = df.pivot_table(
            index="solver_name", columns="dataset_name",
            values=objective_column, aggfunc="mean",
        )
        pivot = pivot.loc[:, pivot.notna().all(axis=0)]
        if pivot.shape[0] >= 2:
            _, anchor_name = _find_anchor_idx(list(pivot.index))

        if anchor_name:
            anchor_desc = f"{anchor_name} = {ELO_ANCHOR:.0f}"
        else:
            anchor_desc = f"mean Elo = {ELO_ANCHOR:.0f}"
        direction = "higher better" if is_higher_better(objective_column) else "lower better"
        return {
            "title": (
                f"Elo leaderboard — {objective_column} "
                f"({direction}, bootstrap N={N_BOOTSTRAP}, {anchor_desc}). "
                f"Bar = Elo; dots = 95% CI low / high."
            ),
            "ylabel": "Elo",
        }
