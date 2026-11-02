"""Tests for the behavioural leakage probe."""

import numpy as np
import pytest

from benchmark_utils.inputs import ForecastInput
from benchmark_utils.leakage import _perturb_future, detect_forecast_leakage
from benchmark_utils.outputs import ForecastOutput


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_perturb_future_preserves_dtype_and_history(dtype):
    series = np.arange(20, dtype=dtype).reshape(10, 2)
    out = _perturb_future(series, cut=6, rng=np.random.default_rng(0))
    assert out.dtype == series.dtype
    np.testing.assert_array_equal(out[:6], series[:6])
    assert not np.array_equal(out[6:], series[6:])


class _DtypeSensitiveFan:
    """A non-leaky fan whose scale is computed in the input dtype."""

    levels = np.array([0.1, 0.5, 0.9])

    def predict(self, forecast_input):
        quantiles = []
        for series, cutoffs in zip(forecast_input.x, forecast_input.cutoff_indexes):
            series = np.asarray(series)
            fan = np.empty((len(cutoffs), 1, series.shape[1], len(self.levels)))
            for k, cutoff in enumerate(cutoffs):
                hist = series[:cutoff]
                scale = np.mean(np.abs(hist[:-1]), axis=0)
                point = hist[-1]
                fan[k] = point[None, :, None] + (
                    self.levels - 0.5
                )[None, None, :] * scale[None, :, None]
            quantiles.append(fan)
        return ForecastOutput(quantiles, self.levels)


def test_probe_ignores_dtype_rounding_of_history_statistics():
    # A float64 copy of the float32 series changes the scale by ~1e-4;
    # the probe must not mistake that rounding for future leakage.
    rng = np.random.default_rng(0)
    hist = rng.normal(10_000, 1_000, size=(8, 1)).astype(np.float32)
    point = (-0.4 * np.mean(np.abs(hist), axis=0)).astype(np.float32)
    series = np.concatenate(
        [hist, point.reshape(1, 1), np.array([[123.0]], dtype=np.float32)]
    )
    report = detect_forecast_leakage(
        _DtypeSensitiveFan(),
        ForecastInput(x=[series], cutoff_indexes=[[9]]),
    )
    assert not report.leaked
    assert report.max_abs_diff == 0.0
