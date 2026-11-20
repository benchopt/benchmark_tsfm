"""Offline tests for the tabular foundation-model solver, with fake models."""

import numpy as np
import pytest

pytest.importorskip("tabicl.forecast")

from sklearn.base import BaseEstimator  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from tabicl.forecast import TabICLForecaster  # noqa: E402

from benchmark_utils.forecasting_constants import (  # noqa: E402
    CANONICAL_QUANTILE_LEVELS,
)
from benchmark_utils.inputs import ForecastInput  # noqa: E402
from benchmark_utils.tabular_fm import (  # noqa: E402
    TabularFMSolver,
    _RegressorEngine,
    forecast_windows,
)

QS = list(CANONICAL_QUANTILE_LEVELS)


class _ConstRegressor(BaseEstimator):
    def fit(self, X, y):
        return self


def _predict_quantiles(model, X, levels):
    """Predict quantile level q as the constant q."""
    return np.tile(levels, (len(X), 1))


class _MeanRegressor(BaseEstimator):
    def fit(self, X, y):
        self.mean_ = np.mean(y)
        return self


def _predict_mean(model, X, levels):
    return np.full((len(X), len(levels)), model.mean_)


def _forecaster(max_context_length=32):
    forecaster = TabICLForecaster(max_context_length=max_context_length)
    forecaster.predictor = _RegressorEngine(_ConstRegressor(), _predict_quantiles)
    return forecaster


def test_forecast_windows_shapes_and_quantile_order():
    rng = np.random.default_rng(0)
    inputs = [rng.normal(size=(40, 2)), rng.normal(size=50)]
    out = forecast_windows(_forecaster(), inputs, 5, QS, "D")
    assert [o.shape for o in out] == [(5, 2, len(QS)), (5, 1, len(QS))]
    np.testing.assert_allclose(out[0][2, 1], QS)


def test_forecast_windows_keeps_series_and_channel_order():
    # Over 10 series, so tabicl's string sort of item ids ("10:0" < "2:0") shows.
    inputs = [np.full((20, 2), [j, j + 0.5]) for j in range(12)]
    forecaster = TabICLForecaster(max_context_length=32)
    forecaster.predictor = _RegressorEngine(_MeanRegressor(), _predict_mean)
    out = forecast_windows(forecaster, inputs, 3, QS, "D")
    for j, o in enumerate(out):
        np.testing.assert_allclose(o[:, :, 0], [[j, j + 0.5]] * 3)


def test_forecast_windows_very_short_history():
    # tabicl needs at least 2 points, below the datasets' MIN_CONTEXT.
    inputs = [np.arange(2, dtype=float), np.arange(30, dtype=float)]
    out = forecast_windows(_forecaster(), inputs, 4, QS, "D")
    assert out[0].shape == (4, 1, len(QS))


def test_forecast_windows_long_yearly_history():
    # 900 yearly steps from the synthetic anchor go past ns-precision's 2262.
    out = forecast_windows(
        _forecaster(max_context_length=1024), [np.ones(900)], 6, QS, "Y"
    )
    assert out[0].shape == (6, 1, len(QS))


class _FakeSolver(TabularFMSolver):
    name = "fake"
    predict_quantiles = staticmethod(_predict_quantiles)

    def make_classifier(self, device):
        return KNeighborsClassifier(n_neighbors=1)

    def make_regressor(self):
        return _ConstRegressor()


def _fit(task, X_train, y_train=None, **meta):
    solver = _FakeSolver(max_context_length=32, inference_batch_size=4096)
    solver.set_objective(X_train, y_train, task, **meta)
    solver.run(None)
    return solver.get_result()["model"]


def test_solver_classification_fits_on_flat_rows():
    rng = np.random.default_rng(0)
    X = [rng.normal(size=(20, 2)) + k for k in (0, 0, 5, 5)]
    model = _fit("classification", X, np.array([0, 0, 1, 1]), n_classes=2)
    np.testing.assert_array_equal(model.predict(X), [0, 0, 1, 1])


def test_solver_forecasting_end_to_end():
    series = [np.arange(40, dtype=np.float32)[:, None]]
    model = _fit("forecasting", series, prediction_length=3, freq="H")
    x = ForecastInput(x=series, cutoff_indexes=[[30, 37]])
    out = model.predict(x)
    assert out.quantiles[0].shape == (2, 3, 1, len(QS))
