"""Tests for the Chronos2 solver wiring (no model load required)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from benchopt.benchmark import Benchmark

from benchmark_utils.covariates import Covariates
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


def test_covariates_follow_fev_bench():
    solver = Solver.get_instance(model_size="small")
    calls = []

    class FakePipeline:
        def predict_quantiles(self, inputs, prediction_length, **kwargs):
            calls.append((inputs, kwargs))
            return [torch.zeros(1, prediction_length, 9) for _ in inputs], None

    solver.model = FakePipeline()
    hist = np.arange(10.0)[:, None]
    future = np.stack([np.arange(10.0) + 100, [0, 1] * 5], axis=-1)
    x = ForecastInput(
        x=[np.zeros((10, 1))],
        cutoff_indexes=[[6]],
        covariates=Covariates(
            hist_covars=[hist],
            future_covars=[future],
            future_categorical=np.array([False, True]),
        ),
    )
    out = solver.forecast(x, 2, quantile_levels=CANONICAL_QUANTILE_LEVELS)
    assert out.quantiles[0].shape == (1, 2, 1, 9)
    ((inp,), kwargs) = calls[0]
    assert kwargs["batch_size"] == 100
    # past-only covariates over the history, known ones over history and horizon
    assert inp["past_covariates"]["hist_0"].tolist() == list(range(6))
    assert inp["past_covariates"]["future_0"].tolist() == list(range(100, 106))
    assert inp["future_covariates"]["future_0"].tolist() == [106, 107]
    # categories are passed as strings, which Chronos-2 encodes itself
    assert inp["future_covariates"]["future_1"].tolist() == ["0.0", "1.0"]
    assert "hist_0" not in inp["future_covariates"]
