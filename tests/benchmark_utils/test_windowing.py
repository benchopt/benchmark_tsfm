"""Split-consistency tests for the shared forecasting data builder."""

import numpy as np
import pytest

from benchmark_utils.covariates import Covariates
from benchmark_utils.windowing import build_forecasting_data


def _series(T):
    return np.arange(T, dtype=np.float32).reshape(-1, 1)


def test_split_shapes():
    d = build_forecasting_data([_series(100)], prediction_length=10, n_windows=2)
    assert d["X_train"][0].shape == (80, 1)
    assert d["y_train"] is None
    assert d["cutoff_indexes"][0] == [80, 90]
    assert d["y_test"][0].shape == (2, 10, 1)


def test_train_does_not_overlap_test_windows():
    ts = _series(100)
    d = build_forecasting_data([ts], prediction_length=10, n_windows=2)
    # Training history ends where the first evaluation window starts.
    assert d["X_train"][0].shape[0] <= d["cutoff_indexes"][0][0]
    assert np.array_equal(d["X_train"][0], ts[:80])


def test_short_series_dropped():
    d = build_forecasting_data(
        [_series(5), _series(50)], prediction_length=10, n_windows=1
    )
    assert len(d["X_train"]) == len(d["X_test"]) == 1


def test_windows_with_short_history_skipped():
    # T=25 with H=10: the first of 2 windows would start after 5 points.
    d = build_forecasting_data(
        [_series(25), _series(15)], prediction_length=10, n_windows=2
    )
    assert d["cutoff_indexes"] == [[15]]


def test_all_series_too_short_raises():
    with pytest.raises(ValueError, match="shorter than prediction_length"):
        build_forecasting_data([_series(5)], prediction_length=10)


def test_debug_uses_single_window():
    d = build_forecasting_data(
        [_series(100)], prediction_length=10, n_windows=2, debug=True
    )
    # Train cut still reserves n_windows, but only one window is evaluated.
    assert d["X_train"][0].shape == (80, 1)
    assert d["cutoff_indexes"][0] == [90]


def test_covariates_follow_kept_series():
    hist = [np.full((T, 1), T, dtype=np.float32) for T in (5, 40, 50)]
    d = build_forecasting_data(
        [_series(5), _series(40), _series(50)],
        prediction_length=10,
        covariates=Covariates(hist_covars=hist),
    )
    assert [h[0, 0] for h in d["covariates"].hist_covars] == [40, 50]
    assert d["covariates"].future_covars is None
