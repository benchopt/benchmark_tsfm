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
    assert solver.skip("classification")[0]
    assert solver.skip("anomaly_detection")[0]
    assert not solver.skip("forecasting")[0]


NATIVE_FAN = tuple(
    sorted({0.01, 0.05, 0.99}.union({round(0.1 + 0.05 * i, 3) for i in range(17)}))
)


class _StubConfig:
    def __init__(self, num_heads: int):
        self.num_heads = num_heads


class _StubForecaster:
    """Capture the packed context and emit a native fan of 21 quantiles."""

    def __init__(self, batch, variates, horizon, num_heads=1):
        self.batch, self.variates, self.horizon = batch, variates, horizon
        self.config = _StubConfig(num_heads)
        self.quantile_levels = NATIVE_FAN
        self.calls = []

    def forecast(self, context, prediction_length, past_covariates=None):
        b, v, t = np.asarray(context).shape
        assert v == self.variates and prediction_length == self.horizon
        assert t > 0
        assert (
            past_covariates is None
            or np.asarray(past_covariates).shape == (b, 2, t)
        )
        self.calls.append(b)
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
