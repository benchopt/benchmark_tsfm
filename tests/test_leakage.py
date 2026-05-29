"""Tests for the forecasting leakage probe.

The probe must:
  * pass a leakage-free adapter (forecast depends only on history),
  * catch an adapter that reads the future target,
  * not perturb (and thus not penalise) future *covariates*,
  * skip series with no future to perturb.
"""

import numpy as np
import pytest

from benchmark_utils.adapters.base import BaseTSFMAdapter
from benchmark_utils.covariates import Covariates
from benchmark_utils.inputs import ForecastInput
from benchmark_utils.leakage import detect_forecast_leakage
from benchmark_utils.outputs import ForecastOutput

H = 3


class CleanForecaster(BaseTSFMAdapter):
    """Repeats the last history value — uses only ``x[:cutoff]``."""

    def predict(self, x: ForecastInput) -> ForecastOutput:
        quantiles = []
        for series, cutoffs in zip(x.x, x.cutoff_indexes):
            series = np.asarray(series)
            C = series.shape[1] if series.ndim == 2 else 1
            arr = np.empty((len(cutoffs), 1, H, C), dtype=np.float64)
            for k, cut in enumerate(cutoffs):
                last = series[:cut][-1]
                arr[k, 0] = np.broadcast_to(last, (H, C))
            quantiles.append(arr)
        return ForecastOutput(quantiles=quantiles, quantile_levels=(0.5,))


class LeakyForecaster(BaseTSFMAdapter):
    """Cheats: returns the actual future target window ``x[cut:cut+H]``."""

    def predict(self, x: ForecastInput) -> ForecastOutput:
        quantiles = []
        for series, cutoffs in zip(x.x, x.cutoff_indexes):
            series = np.asarray(series)
            C = series.shape[1] if series.ndim == 2 else 1
            arr = np.empty((len(cutoffs), 1, H, C), dtype=np.float64)
            for k, cut in enumerate(cutoffs):
                arr[k, 0] = series[cut:cut + H].reshape(H, C)
            quantiles.append(arr)
        return ForecastOutput(quantiles=quantiles, quantile_levels=(0.5,))


class GlobalScaleLeaker(BaseTSFMAdapter):
    """Subtly leaky: normalises by stats of the FULL series, not history."""

    def predict(self, x: ForecastInput) -> ForecastOutput:
        quantiles = []
        for series, cutoffs in zip(x.x, x.cutoff_indexes):
            series = np.asarray(series)
            C = series.shape[1] if series.ndim == 2 else 1
            global_mean = series.mean(axis=0)  # peeks at the future
            arr = np.empty((len(cutoffs), 1, H, C), dtype=np.float64)
            for k in range(len(cutoffs)):
                arr[k, 0] = np.broadcast_to(global_mean, (H, C))
            quantiles.append(arr)
        return ForecastOutput(quantiles=quantiles, quantile_levels=(0.5,))


@pytest.fixture
def forecast_input():
    rng = np.random.default_rng(0)
    x = [rng.standard_normal((40, 2)), rng.standard_normal((30, 2))]
    cutoff_indexes = [[20], [15]]
    return ForecastInput(x=x, cutoff_indexes=cutoff_indexes)


def test_clean_forecaster_is_not_flagged(forecast_input):
    report = detect_forecast_leakage(CleanForecaster(), forecast_input)
    assert not report.leaked
    assert report.offending_series == []
    assert report.n_series_tested == 2
    assert report.max_abs_diff == pytest.approx(0.0)


def test_leaky_forecaster_is_caught(forecast_input):
    report = detect_forecast_leakage(LeakyForecaster(), forecast_input)
    assert report.leaked
    assert report.offending_series == [0, 1]
    assert report.max_abs_diff > 0


def test_global_scale_leak_is_caught(forecast_input):
    report = detect_forecast_leakage(GlobalScaleLeaker(), forecast_input)
    assert report.leaked


def test_future_covariates_are_not_perturbed():
    """A model using only future covariates (legitimately known) must pass."""

    class CovariateOnlyForecaster(BaseTSFMAdapter):
        def predict(self, x: ForecastInput) -> ForecastOutput:
            quantiles = []
            for i, (series, cutoffs) in enumerate(zip(x.x, x.cutoff_indexes)):
                C = np.asarray(series).shape[1]
                fc = np.asarray(x.covariates.future_covars[i])  # (H, C)
                arr = np.stack(
                    [fc.reshape(H, C) for _ in cutoffs]
                )[:, None, :, :]
                quantiles.append(arr.astype(np.float64))
            return ForecastOutput(quantiles=quantiles, quantile_levels=(0.5,))

    rng = np.random.default_rng(1)
    x = [rng.standard_normal((40, 2)), rng.standard_normal((30, 2))]
    cutoff_indexes = [[20], [15]]
    future_covars = [rng.standard_normal((H, 2)), rng.standard_normal((H, 2))]
    fi = ForecastInput(
        x=x,
        cutoff_indexes=cutoff_indexes,
        covariates=Covariates(future_covars=future_covars),
    )
    report = detect_forecast_leakage(CovariateOnlyForecaster(), fi)
    assert not report.leaked


def test_series_with_no_future_is_skipped():
    x = [np.arange(20, dtype=np.float64).reshape(20, 1)]
    cutoff_indexes = [[20]]  # cutoff == T → nothing after it to perturb
    fi = ForecastInput(x=x, cutoff_indexes=cutoff_indexes)
    report = detect_forecast_leakage(LeakyForecaster(), fi)
    assert report.n_series_tested == 0
    assert report.n_series_skipped == 1
    assert not report.leaked


def test_max_series_limits_probe(forecast_input):
    report = detect_forecast_leakage(
        LeakyForecaster(), forecast_input, max_series=1
    )
    assert report.n_series_tested == 1
    assert report.offending_series == [0]
