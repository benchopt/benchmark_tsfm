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

import re

import numpy as np
import pandas as pd
from bencheval.evaluator import BenchmarkEvaluator

ELO_ANCHOR = 1000.0  # Elo assigned to the anchor solver (see BASELINE_PREFERENCES)
N_BOOTSTRAP = 200

# Reference baseline for Elo normalisation. We prefer "Seasonal Naive" — the
# canonical forecasting reference — but fall back to plain "Naive" if no
# seasonal variant is present, and finally to the mean-anchor if no naive
# baseline exists at all. TabArena anchors default RandomForest at 1000 for
# the same reason: a fixed, well-understood reference makes the Elo gap
# interpretable across metrics and across re-runs.
BASELINE_PREFERENCES = ("seasonal naive", "seasonal_naive", "naive")


def _short(name: str) -> str:
    return re.sub(r"\[.*?\]$", "", str(name)).strip()


def _find_baseline(solvers: list[str]) -> str | None:
    """Pick the reference baseline solver (see ``BASELINE_PREFERENCES``).

    Used to anchor Elo at ``ELO_ANCHOR`` and as the denominator of relative
    metrics. Returns the matching entry of ``solvers`` unchanged, or ``None``
    if no preferred baseline is present.
    """
    for needle in BASELINE_PREFERENCES:
        for s in solvers:
            if needle in _short(s).lower():
                return s
    return None


def _elo_table(
    mat: pd.DataFrame, n_boot: int = N_BOOTSTRAP, seed: int = 0
) -> tuple[pd.DataFrame, str | None]:
    """Compute per-solver Elo + 95% bootstrap CI.

    Returns ``(table, anchor_name)`` where ``table`` is a DataFrame sorted by
    Elo descending with columns (solver, elo, ci_low, ci_high, games), and
    ``anchor_name`` is the short name of the solver pinned at ``ELO_ANCHOR``
    (or ``None`` if mean-anchoring was used).
    """
    k, n = mat.shape
    baseline = _find_baseline(list(mat.index))

    long = (
        mat.rename_axis(index="method", columns="task")
        .stack()
        .rename("metric_error")
        .reset_index()
    )
    elo = BenchmarkEvaluator(columns_to_agg_extra=None).compute_elo(
        long,
        calibration_framework=baseline,
        calibration_elo=ELO_ANCHOR,
        INIT_RATING=ELO_ANCHOR,
        BOOTSTRAP_ROUNDS=n_boot,
        round_decimals=None,
        clip_negative_ci=False,
        post_calibrate=False,
    )

    out = pd.DataFrame(
        {
            "solver": [_short(s) for s in elo.index],
            "elo": elo["elo"].to_numpy(),
            "ci_low": (elo["elo"] - elo["elo-"]).to_numpy(),
            "ci_high": (elo["elo"] + elo["elo+"]).to_numpy(),
            # Each solver plays (k-1) opponents on each of n datasets
            "games": np.full(k, (k - 1) * n, dtype=np.int64),
        }
    )
    return out, None if baseline is None else _short(baseline)
