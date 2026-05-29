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
import re

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit

from benchopt import BasePlot


# Chess / Chatbot Arena / TabArena convention: 400-Elo gap = 91% win rate
ELO_SCALE = 400.0 / math.log(10)
ELO_ANCHOR = 1000.0  # mean Elo across solvers
N_BOOTSTRAP = 200
EPS = 1e-9


def _short(name: str) -> str:
    return re.sub(r"\[.*?\]$", "", str(name)).strip()


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
               seed: int = 0) -> pd.DataFrame:
    """Compute per-solver Elo + 95% bootstrap CI.

    Returns a DataFrame sorted by Elo descending with columns:
        solver, elo, ci_low, ci_high, games
    """
    arr = mat.to_numpy(dtype=np.float64)
    k, n = arr.shape
    solvers = list(mat.index)

    # Point estimate on the full data
    r_full = _fit_bt(_pairwise_wins(arr))
    elo_full = r_full * ELO_SCALE + ELO_ANCHOR

    # Bootstrap CI by resampling datasets with replacement
    rng = np.random.default_rng(seed)
    boots = np.zeros((n_boot, k))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r_b = _fit_bt(_pairwise_wins(arr[:, idx]))
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
    return out


class Plot(BasePlot):
    """Elo leaderboard with bootstrap 95% CI — as a benchopt table.

    Appears in the HTML report's "Chart type" dropdown. ``objective_column``
    is auto-populated with every numeric ``objective_*`` column in the
    parquet, switching the leaderboard in place.

    Assumes lower-is-better metrics (FEV's SQL/MASE/WAPE/WQL/error fit, as do
    most forecasting losses). For higher-is-better metrics, negate the column
    upstream (e.g., store ``1 - roc_auc`` rather than ``roc_auc``).
    """

    name = "Elo"
    type = "table"
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
        # Strict clean: drop datasets where any solver is NaN (Elo needs
        # complete games), then drop all-tied datasets (variance == 0).
        pivot = pivot.loc[:, pivot.notna().all(axis=0)]
        if pivot.shape[1] > 0:
            var = pivot.var(axis=0, skipna=False)
            pivot = pivot.loc[:, var > 0]

        k, n = pivot.shape
        if k < 2 or n < 2:
            return [["(insufficient data — need ≥2 solvers and ≥2 complete datasets)",
                     "—", "—", "—", "—", "—"]]

        table = _elo_table(pivot)
        rank_col = list(range(1, len(table) + 1))
        rows = []
        for rank, row in zip(rank_col, table.itertuples(index=False)):
            half_width = (row.ci_high - row.ci_low) / 2.0
            rows.append([
                rank,
                row.solver,
                f"{row.elo:.0f}",
                f"{row.ci_low:.0f}",
                f"{row.ci_high:.0f}",
                f"±{half_width:.0f}",
                int(row.games),
            ])
        return rows

    def get_metadata(self, df, objective_column):
        return {
            "title": (
                f"Elo leaderboard — {objective_column} "
                f"(lower better, bootstrap N={N_BOOTSTRAP}, mean anchored at "
                f"{ELO_ANCHOR:.0f})"
            ),
            "columns": ["Rank", "Solver", "Elo",
                        "95% CI low", "95% CI high", "± half-width", "Games"],
        }
