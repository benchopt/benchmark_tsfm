"""Tests for the GIFT-Eval / gluonts forecasting metric semantics."""

import numpy as np
import pytest

from benchmark_utils.metrics import (
    _seasonal_naive_scales_per_window,
    mae,
    mase,
    smape,
    windows_valid_mask,
)
from benchmark_utils.outputs import ForecastOutput


def _point_forecast(point):
    """ForecastOutput with a degenerate fan around ``point`` (M, H, C)."""
    point = np.asarray(point, dtype=np.float64)
    quantiles = point[..., None] * np.ones((1,))
    return ForecastOutput(quantiles=[quantiles], quantile_levels=[0.5])


# Two series, two channels each. A cutoff c means the context is
# series[:c]: the point at index c is the first forecast target.
_HISTORIES = [
    np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]]),
    np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 1.0], [6.0, 3.0], [8.0, 1.0]]),
]
_CUTOFFS = [[4], [4]]


def test_scales_per_window_gluonts_semantics():
    # s0 diffs (s=2): |3-1|=2, |4-2|=2 -> 2.0; ch1: 20.0
    # s1 diffs (s=2): ch0: 4.0; ch1: 1,3,1,3 alternates at lag 2 -> 0.0,
    # and a zero scale is *undefined* (dropped), hence NaN
    scales = _seasonal_naive_scales_per_window(_HISTORIES, _CUTOFFS, 2)
    np.testing.assert_allclose(scales, [[2.0, 20.0], [4.0, np.nan]])


def test_scales_per_window_constant_context_is_nan():
    histories = [np.full((5, 1), 3.0)]
    scales = _seasonal_naive_scales_per_window(histories, [[4]], 2)
    assert np.isnan(scales).all()


def test_scales_per_window_short_context_falls_back_to_season_1():
    histories = [np.array([[1.0], [4.0]])]
    scales = _seasonal_naive_scales_per_window(histories, [[2]], 12)
    np.testing.assert_allclose(scales, [[3.0]])


def test_scales_per_window_nan_mask_excludes_imputed_diffs():
    histories = [np.array([[1.0], [np.nan], [3.0], [5.0]])]
    nan_masks = [np.array([False, True, False, False])]
    # the NaN at position 1 kills the diff |3-1|: only |5-3| remains
    scales = _seasonal_naive_scales_per_window(histories, [[4]], 1, nan_masks=nan_masks)
    np.testing.assert_allclose(scales, [[2.0]])


def test_mase_per_window_matches_gluonts_aggregation():
    # scales: s0 = (2, 20); s1 = (4, NaN) -> s1 ch1 points dropped
    y_true = np.array([[[6.0, 60.0], [7.0, 70.0]], [[10.0, 1.0], [12.0, 3.0]]])
    forecast = _point_forecast(np.zeros_like(y_true))
    value = mase(
        y_true,
        forecast,
        histories=_HISTORIES,
        cutoff_indexes=_CUTOFFS,
        seasonality=2,
    )
    expected = np.mean([6.0 / 2, 7.0 / 2, 60.0 / 20, 70.0 / 20, 10.0 / 4, 12.0 / 4])
    assert value == pytest.approx(expected)


def test_mase_legacy_fallback_without_histories():
    y_true = np.array([[[3.0], [5.0]]])
    forecast = _point_forecast(np.zeros_like(y_true))
    y_train = np.array([[1.0], [2.0]])
    # legacy scale: mean |diff season 1| of y_train = 1 -> (3 + 5) / 2
    assert mase(y_true, forecast, y_train=y_train, seasonality=1) == 4.0


def test_mase_drops_undefined_scale_points():
    # s1 has a constant context: its scale is undefined, only the s0
    # points are scored
    histories = [
        np.array([[1.0], [2.0], [3.0], [4.0]]),
        np.array([[1.0], [1.0], [1.0], [1.0]]),
    ]
    cutoffs = [[4], [4]]
    y_true = np.array([[[2.0]], [[5.0]]])
    forecast = _point_forecast(np.zeros_like(y_true))
    value = mase(
        y_true, forecast, histories=histories, cutoff_indexes=cutoffs, seasonality=1
    )
    # s0 scale = 1 (mean |diff|), s1 scale undefined -> dropped
    assert value == pytest.approx(2.0)


def test_windows_valid_mask():
    nan_masks = [np.array([False, False, False, True, False])]
    cutoffs = [[3, 4]]
    # H=1: window 0 covers index 3 (NaN -> invalid), window 1 covers 4
    mask = windows_valid_mask(nan_masks, cutoffs, (2, 1, 1))
    np.testing.assert_array_equal(mask, [[[False]], [[True]]])


def test_mae_valid_mask():
    y_true = np.array([[[1.0], [2.0]]])
    forecast = _point_forecast(np.zeros_like(y_true))
    valid = np.array([[[True], [False]]])
    assert mae(y_true, forecast, valid_mask=valid) == pytest.approx(1.0)


def test_smape_gluonts_formula_drops_zero_zero():
    y_true = np.array([[[0.0], [2.0]]])
    forecast = _point_forecast(np.zeros_like(y_true))
    # 0/0 is undefined and dropped; the remaining point is 2*2/2 = 2
    assert smape(y_true, forecast) == pytest.approx(2.0)


def test_smape_valid_mask():
    y_true = np.array([[[0.0], [2.0]]])
    forecast = _point_forecast(np.zeros_like(y_true))
    valid = np.array([[[True], [False]]])
    # both points are 0/0 or masked: undefined -> NaN
    assert np.isnan(smape(y_true, forecast, valid_mask=valid))
