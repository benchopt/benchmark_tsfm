"""Tests for the Chronos2 solver wiring (no model load required)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from benchopt.benchmark import Benchmark

from benchmark_utils.forecasting_constants import CANONICAL_QUANTILE_LEVELS
from benchmark_utils.inputs import ForecastInput

BENCHMARK_DIR = Path(__file__).parents[2]

(Solver,) = Benchmark(BENCHMARK_DIR).check_solver_patterns(
    ["chronos2"], class_only=True
)


def test_model_ids():
    assert (
        Solver.get_instance(model_size="small").model_id
        == "autogluon/chronos-2-small"
    )
    assert Solver.get_instance(model_size="base").model_id == "amazon/chronos-2"


def test_quantile_levels():
    assert Solver.get_instance(
        model_size="small"
    ).quantile_levels == CANONICAL_QUANTILE_LEVELS


@pytest.mark.parametrize("cross_learning", [False, True])
def test_cross_learning_batches_one_window_per_series(cross_learning):
    solver = Solver.get_instance(model_size="small", cross_learning=cross_learning)
    batches = []

    def forecast_batch(inputs, covariates, prediction_length):
        batches.append([len(inp) for inp in inputs])
        # The forecast echoes the history length, to check the output order.
        return [torch.full((prediction_length, 1, 1), len(inp)) for inp in inputs]

    solver.forecast_batch = forecast_batch
    x = ForecastInput(
        x=[np.zeros((20, 1)), np.zeros((15, 1))],
        cutoff_indexes=[[10, 15], [5, 10]],
    )
    out = solver.forecast(x, prediction_length=2, quantile_levels=(0.5,))
    if cross_learning:
        # Each batch holds the windows at one distance from the series end.
        assert batches == [[15, 10], [10, 5]]
    else:
        assert batches == [[10, 15, 5, 10]]
    assert [q[:, 0, 0, 0].tolist() for q in out.quantiles] == [[10, 15], [5, 10]]
