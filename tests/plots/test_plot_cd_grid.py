"""Regression test for the CD-diagram plot.

benchopt pre-shortens ``solver_name`` (keeping only the parameters that vary),
so the plot receives unique labels and no longer collapses several
parametrizations of one solver into a duplicate label — which used to make
``scikit_posthocs`` raise "truth value of a Series is ambiguous".
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BENCHMARK_DIR = Path(__file__).parents[2]
sys.path.insert(0, str(BENCHMARK_DIR / "plots"))

import plot_cd_grid as cd  # noqa: E402


def _synthetic_results():
    # Distinct solver names, as benchopt delivers them after short-labeling.
    rng = np.random.default_rng(0)
    solvers = [
        "SeasonalNaive[season_length=1]",
        "SeasonalNaive[season_length=7]",
        "Chronos2",
    ]
    datasets = [f"d{i}" for i in range(6)]
    return pd.DataFrame([
        {
            "solver_name": s,
            "dataset_name": d,
            "objective_mae": rng.random(),
            "objective_mse": rng.random(),
        }
        for s in solvers
        for d in datasets
    ])


def test_cd_for_metric_runs():
    df = _synthetic_results()
    fig, ax = plt.subplots()
    status = cd.cd_for_metric(df, "objective_mae", ax)
    plt.close(fig)
    assert "objective_mae" in status


def test_cd_global_runs():
    df = _synthetic_results()
    fig, ax = plt.subplots()
    status = cd.cd_for_metric(df, cd.GLOBAL_KEY, ax)
    plt.close(fig)
    assert isinstance(status, str)
