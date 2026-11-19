"""Offline tests for the tabular forecasting helper, with a stub regressor."""

import numpy as np
import pytest

pytest.importorskip("tabicl.forecast")

from tabicl.forecast._engine import ForecastEngine  # noqa: E402

from benchmark_utils.tabular_forecast import (  # noqa: E402
    DEFAULT_FORECAST_QUANTILES,
    build_forecaster,
    forecast_windows,
    safe_freq,
)

QS = list(DEFAULT_FORECAST_QUANTILES)


class _StubEngine(ForecastEngine):
    """Predict quantile level q as the constant q."""

    def _inference_routine(self, train_X, train_y, test_X, quantiles=None):
        n = len(test_X)
        result = {"target": np.full(n, 0.5)}
        result.update({q: np.full(n, q) for q in quantiles})
        return result


def _stub_forecaster(max_context_length=32):
    forecaster = build_forecaster("tabicl", max_context_length=max_context_length)
    forecaster.predictor = _StubEngine()
    return forecaster


def test_forecast_windows_shapes_and_quantile_order():
    rng = np.random.default_rng(0)
    inputs = [rng.normal(size=(40, 2)), rng.normal(size=50)]
    out = forecast_windows(_stub_forecaster(), inputs, 5, QS, "D")
    assert [o.shape for o in out] == [(5, 2, len(QS)), (5, 1, len(QS))]
    np.testing.assert_allclose(out[0][2, 1], QS)


@pytest.mark.parametrize("length", [1, 2])
def test_forecast_windows_very_short_history(length):
    inputs = [np.arange(length, dtype=float), np.arange(30, dtype=float)]
    out = forecast_windows(_stub_forecaster(), inputs, 4, QS, "D")
    assert out[0].shape == (4, 1, len(QS))


def test_forecast_windows_long_yearly_history():
    # 900 yearly steps from the synthetic anchor overflow pandas' range
    # unless the history is trimmed to the context length first.
    out = forecast_windows(_stub_forecaster(), [np.ones(900)], 6, QS, "Y")
    assert out[0].shape == (6, 1, len(QS))


@pytest.mark.parametrize(
    "alias, expected",
    [
        ("A-DEC", "YE-DEC"),
        ("Y", "YE"),
        ("Q", "QE"),
        ("M", "ME"),
        ("H", "h"),
        ("6H", "6h"),
        ("5T", "5min"),
        ("4S", "4s"),
        ("W-SUN", "W-SUN"),
        ("D", "D"),
        ("QS-OCT", "QS-OCT"),
        (None, "h"),
    ],
)
def test_safe_freq_normalizes_legacy_aliases(alias, expected):
    assert safe_freq(alias) == expected


def test_safe_freq_warns_on_unusable_alias():
    with pytest.warns(UserWarning, match="freq"):
        assert safe_freq("not-a-freq") == "h"
