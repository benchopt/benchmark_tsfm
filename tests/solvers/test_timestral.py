"""Tests for the Timestral solver wiring (no model load required)."""

import numpy as np
import torch
from pathlib import Path

from benchopt.benchmark import Benchmark

from benchmark_utils.covariates import Covariates
from benchmark_utils.forecasting_constants import CANONICAL_QUANTILE_LEVELS

BENCHMARK_DIR = Path(__file__).parents[2]

(Solver,) = Benchmark(BENCHMARK_DIR).check_solver_patterns(
    ["timestral"], class_only=True
)


def test_model_id():
    assert Solver.get_instance(model_size="209").model_id == (
        "mistralai/Timestral-209M"
    )
    assert Solver.get_instance(model_size="5").model_id == (
        "mistralai/Timestral-5M"
    )


def test_forecasting_only():
    solver = Solver.get_instance()
    assert solver.supported_tasks == {"forecasting"}


NATIVE_FAN = tuple(
    sorted({0.01, 0.05, 0.99}.union({round(0.1 + 0.05 * i, 3) for i in range(17)}))
)


class _StubConfig:
    def __init__(self, num_heads: int):
        self.num_heads = num_heads


class _StubForecaster:
    """Capture the packed context and emit a native fan of 21 quantiles."""

    def __init__(self, batch, variates, horizon, num_heads=1, known=True):
        self.batch, self.variates, self.horizon = batch, variates, horizon
        self.config = _StubConfig(num_heads)
        self.quantile_levels = NATIVE_FAN
        self.supports_known_covariates = known
        self.calls = []
        self.covariates = []

    def forecast(self, context, prediction_length, **covariates):
        b, v, t = np.asarray(context).shape
        assert v == self.variates and prediction_length == self.horizon
        assert t > 0
        for name, cov in covariates.items():
            length = t + prediction_length if "known" in name else t
            assert cov.shape[0] == b and cov.shape[2] == length, name
        self.calls.append(b)
        self.covariates.append(covariates)
        q = torch.arange(b * v * self.horizon * 21, dtype=torch.float32)
        q = q.reshape(b, v, self.horizon, 21)
        return type("Forecast", (), {"quantiles": q})


def _stub_solver(**overrides):
    solver = object.__new__(Solver)
    solver.inference_batch_size = overrides.pop("inference_batch_size", 128)
    solver.SDPA_SCORE_FLOATS = overrides.pop(
        "SDPA_SCORE_FLOATS", 10_000_000_000
    )
    solver.model = _StubForecaster(
        batch=8, variates=3, horizon=4
    )
    return solver


def test_forecast_batch_axis_order_and_fan():
    # H != Q != V: distinct sizes pin the output mapping (H, V, 9) without a
    # shape coincidence (e.g. jena V=Q=21).
    solver = _stub_solver()
    out = solver.forecast_batch(
        [torch.rand(96, 3), torch.rand(64, 3)],
        [Covariates(), Covariates()],
        prediction_length=4,
    )
    assert [tuple(o.shape) for o in out] == [(4, 3, 9), (4, 3, 9)]
    assert solver.model.calls == [2]
    values = torch.arange(2 * 3 * 4 * 21).reshape(2, 3, 4, 21)
    idx = tuple(NATIVE_FAN.index(q) for q in solver.quantile_levels)
    assert torch.equal(out[0], values[0][..., idx].permute(1, 0, 2))


def test_canonical_fan():
    solver = _stub_solver()
    assert solver.quantile_levels == CANONICAL_QUANTILE_LEVELS
    idx = solver._level_indices
    assert tuple(NATIVE_FAN[i] for i in idx) == solver.quantile_levels


def test_forecast_batch_rows_budget():
    # SDPA budget 36 floats: one window per call.
    solver = _stub_solver(SDPA_SCORE_FLOATS=36)
    solver.forecast_batch(
        [torch.rand(96, 3) for _ in range(5)],
        [Covariates()] * 5,
        prediction_length=4,
    )
    assert solver.model.calls == [1, 1, 1, 1, 1]


def test_forecast_batch_with_past_covariates():
    solver = _stub_solver()
    cov = Covariates(hist_covars=[np.random.rand(96, 2).astype(np.float32)])
    out = solver.forecast_batch(
        [torch.rand(96, 3), torch.rand(48, 3)],
        [cov, cov],
        prediction_length=4,
    )
    assert [tuple(o.shape) for o in out] == [(4, 3, 9), (4, 3, 9)]


def test_forecast_batch_with_known_and_categorical_covariates():
    solver = _stub_solver()
    # 2 past-only columns (the second categorical), 2 known (the first
    # categorical), given over the history and the horizon.
    cov = Covariates(
        hist_covars=[np.arange(10.0).reshape(5, 2)],
        future_covars=[np.arange(18.0).reshape(9, 2)],
        hist_categorical=np.array([False, True]),
        future_categorical=np.array([True, False]),
    )
    solver.forecast_batch([torch.rand(5, 3)], [cov], prediction_length=4)
    (kw,) = solver.model.covariates
    assert sorted(kw) == [
        "categorical_known_covariates",
        "categorical_past_covariates",
        "known_covariates",
        "past_covariates",
    ]
    assert kw["past_covariates"][0].tolist() == [[0, 2, 4, 6, 8]]
    assert kw["categorical_past_covariates"][0].tolist() == [[1, 3, 5, 7, 9]]
    assert kw["categorical_known_covariates"][0, 0].tolist() == list(range(0, 18, 2))
    assert kw["known_covariates"][0, 0].tolist() == list(range(1, 18, 2))


def test_forecast_batch_skips_known_covariates_if_unsupported():
    solver = _stub_solver()
    solver.model.supports_known_covariates = False
    cov = Covariates(
        future_covars=[np.zeros((9, 2))], future_categorical=np.array([True, False])
    )
    solver.forecast_batch([torch.rand(5, 3)], [cov], prediction_length=4)
    assert solver.model.covariates == [{}]
