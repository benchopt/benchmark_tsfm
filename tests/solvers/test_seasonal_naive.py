"""Tests for the SeasonalNaive quantile fan (statsforecast-native intervals)."""

import numpy as np
from scipy.stats import norm

from benchmark_utils.inputs import ForecastInput
from solvers.seasonal_naive import QUANTILE_LEVELS, _SeasonalNaiveForecaster


def _fan(hist, prediction_length, season_length, cutoffs=None):
    hist = np.asarray(hist, dtype=np.float64)
    if hist.ndim == 1:
        hist = hist[:, None]
    cutoff_indexes = [[hist.shape[0]]] if cutoffs is None else [list(cutoffs)]
    adapter = _SeasonalNaiveForecaster(
        prediction_length=prediction_length, season_length=season_length
    )
    out = adapter.predict(ForecastInput(x=[hist], cutoff_indexes=cutoff_indexes))
    assert out.quantile_levels == QUANTILE_LEVELS
    return out.quantiles[0]  # (n_cutoffs, H, C, Q)


Q_LO, Q_05, Q_HI = 0, 4, 8

# hist: [0, 1, 2, 3] with m=2 -> last season [2, 3], repeated over H=3
_HIST = [0.0, 1.0, 2.0, 3.0]
Z10 = norm.ppf(0.9)  # half-width of the 80% interval


def test_median_is_point_forecast():
    fan = _fan(_HIST, prediction_length=3, season_length=2)
    np.testing.assert_allclose(fan[0, :, 0, Q_05], [2.0, 3.0, 2.0])


def test_sigma_is_rms_seasonal_diff():
    # diffs at lag 2: (2-0)^2, (3-1)^2 -> sigma = sqrt(8/2) = 2
    # cycles over H=3: k = floor([0,1,2]/2) = [0,0,1] -> sigma_h = [2, 2, 2*sqrt(2)]
    fan = _fan(_HIST, prediction_length=3, season_length=2)
    np.testing.assert_allclose(
        fan[0, :, 0, Q_LO],
        [2.0 - Z10 * 2, 3.0 - Z10 * 2, 2.0 - Z10 * 2 * np.sqrt(2)],
    )


def test_fan_levels_nested_around_point():
    fan = _fan(_HIST, prediction_length=3, season_length=2)
    # lo(0.1) and hi(0.9) are symmetric around the point
    np.testing.assert_allclose(
        (fan[0, :, 0, Q_LO] + fan[0, :, 0, Q_HI]) / 2, fan[0, :, 0, Q_05]
    )
    widths = fan[0, 0, 0, :] - fan[0, 0, 0, Q_05]
    assert np.all(np.diff(widths[:5]) > 0)  # lo side grows towards 0.5
    assert np.all(np.diff(widths[5:]) > 0)  # hi side grows after 0.5


def test_short_context_degenerates_to_point():
    # history shorter than the season: no seasonal diff at all -> sigma 0
    fan = _fan([1.0, 2.0], prediction_length=3, season_length=12)
    np.testing.assert_allclose(fan[..., Q_LO], fan[..., Q_05])
    np.testing.assert_allclose(fan[..., Q_HI], fan[..., Q_05])
    np.testing.assert_allclose(fan[0, :, 0, Q_05], [1.0, 2.0, 1.0])


def test_constant_channel_has_zero_sigma():
    hist = np.column_stack([np.arange(4.0), np.array([9.0, 9.0, 9.0, 9.0])])
    fan = _fan(hist, prediction_length=2, season_length=2)
    np.testing.assert_allclose(fan[0, :, 1, Q_HI], [9.0, 9.0])
    # channel 0: sigma = 2 -> 0.9 = point + z(0.9) * 2
    np.testing.assert_allclose(fan[0, :, 0, Q_HI], [2.0 + Z10 * 2, 3.0 + Z10 * 2])


def test_season_length_1_spread_grows_with_step():
    # [0, 1, 2, 3], m=1: sigma = sqrt(3/3) = 1, sigma_h = 1*sqrt(1..H)
    fan = _fan([0.0, 1.0, 2.0, 3.0], prediction_length=3, season_length=1)
    np.testing.assert_allclose(fan[0, :, 0, Q_05], [3.0, 3.0, 3.0])
    np.testing.assert_allclose(fan[0, :, 0, Q_HI], 3.0 + Z10 * np.sqrt([1.0, 2.0, 3.0]))


def test_two_cutoffs_scored_separately():
    # cutoff 2: sigma 1, point 1 ; cutoff 4: sigma 1, point 3, spread
    # sqrt([1, 2]) over the two steps
    fan = _fan(_HIST, prediction_length=2, season_length=1, cutoffs=[2, 4])
    np.testing.assert_allclose(fan[0, 0, 0, Q_HI], 1.0 + Z10)
    np.testing.assert_allclose(fan[1, 0, 0, Q_HI], 3.0 + Z10)
    np.testing.assert_allclose(fan[1, 1, 0, Q_HI], 3.0 + Z10 * np.sqrt(2))
