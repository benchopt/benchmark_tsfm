"""Tests for the CD diagram plot.

The diagram keeps solver names as given, so the parametrizations of one solver
stay distinct rows.
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
    # Two parametrizations of one solver.
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
