"""Elo ratings (Bradley-Terry MLE) with bootstrap 95% CI, shared by the plots.

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
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit

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
ANCHOR_PREFERENCES = (
    "seasonal naive",
    "seasonal_naive",
    "seasonalnaive",  # the name of solvers/seasonal_naive.py
    "naive",
)


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
    diff = mat[:, None, :] - mat[None, :, :]  # (k, k, N): mat[i] - mat[j]
    wins = (diff < 0).sum(axis=-1).astype(np.float64)
    ties = (diff == 0).sum(axis=-1).astype(np.float64) * 0.5
    W = wins + ties
    np.fill_diagonal(W, 0.0)
    return W


def _fit_bt(W: np.ndarray, prior_games: float = 1.0) -> np.ndarray:
    """Fit Bradley-Terry log-ratings by maximum likelihood.

    The BT log-likelihood of observed wins W is
        sum_{i,j} W[i,j] * log sigmoid(r_i - r_j).
    Location is unidentified, so we anchor at sum(r) = 0 after the fit.

    Parameters
    ----------
    W : (k, k) win matrix from _pairwise_wins.
    prior_games : float
        Total number of pseudo-games added *per pair*, split 50/50.
        The actual pseudocount added to each W[i,j] is
        ``prior_games / (2 * n_datasets)``, so the prior always represents
        a fixed fraction of the evidence regardless of dataset count.
        Default 1.0 → the prior is equivalent to one extra balanced game
        across the whole dataset, keeping ratings bounded without
        meaningfully shrinking real gaps.
    """
    k = W.shape[0]
    # Total games played per pair = W[i,j] + W[j,i] (≈ n_datasets).
    # Scale prior so it stays a constant fraction of evidence.
    n_datasets = W.sum() / (k * (k - 1)) if k > 1 else 1.0
    eps = prior_games / (2.0 * max(n_datasets, 1.0))
    prior_mat = eps * (1.0 - np.eye(k))
    W_reg = W + prior_mat

    def nll(r):
        diff = r[:, None] - r[None, :]
        return -float((W_reg * log_expit(diff)).sum())

    def grad(r):
        diff = r[:, None] - r[None, :]
        p = expit(-diff)
        g = (W_reg * p).sum(axis=1) - (W_reg.T * (1 - p.T)).sum(axis=1)
        return -g

    r0 = np.zeros(k)
    res = minimize(
        nll, r0, jac=grad, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-9}
    )
    r = res.x - res.x.mean()
    return r


def _elo_table(
    mat: pd.DataFrame, n_boot: int = N_BOOTSTRAP, seed: int = 0
) -> tuple[pd.DataFrame, str | None]:
    """Compute per-solver Elo + 95% bootstrap CI.

    Returns ``(table, anchor_name)`` where ``table`` is a DataFrame sorted by
    Elo descending with columns (name, solver, elo, ci_low, ci_high, games), and
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

    out = (
        pd.DataFrame(
            {
                "name": solvers,  # the index of ``mat``, unique
                "solver": [_short(s) for s in solvers],
                "elo": elo_full,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "games": games,
            }
        )
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )
    return out, anchor_name
