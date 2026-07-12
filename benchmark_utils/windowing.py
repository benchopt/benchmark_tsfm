"""Rolling-window utilities for forecasting evaluation.

`make_forecasting_splits` returns the full series alongside per-series
cutoff indexes and target horizons. This shape matches the batched
adapter contract: a forecaster gets the whole history per series plus
the list of cutoffs at which it should forecast.

Outputs
-------
series_full     : List[np.ndarray (T_i, C)]
cutoff_indexes  : List[List[int]] — for each series, the timestep
                  indexes at which a forecast starts (i.e. ``x[:cutoff]``
                  is the history available to the model).
targets         : List[np.ndarray (n_cutoffs_i, prediction_length, C)]
                  ground-truth windows aligned with cutoff_indexes.
"""

from typing import List, Optional, Tuple

import numpy as np


def make_forecasting_splits(
    series: List[np.ndarray],
    prediction_length: int,
    n_windows: int = 1,
    stride: Optional[int] = None,
    min_context: int = 1,
) -> Tuple[List[np.ndarray], List[List[int]], List[np.ndarray]]:
    """Create rolling-window evaluation cutoffs from a list of time series.

    Parameters
    ----------
    series : list of (T_i, C) arrays — full time series.
    prediction_length : int
    n_windows : int
        Number of rolling evaluation windows per series.
    stride : int or None
        Step between consecutive prediction points.
        Defaults to ``prediction_length`` (non-overlapping).
    min_context : int
        Minimum context length required before the first prediction point.
    """
    if stride is None:
        stride = prediction_length

    series_full: List[np.ndarray] = []
    cutoff_indexes: List[List[int]] = []
    targets: List[np.ndarray] = []

    for ts in series:
        ts = np.asarray(ts)
        T = ts.shape[0]
        cutoffs: List[int] = []
        ys: List[np.ndarray] = []
        for w in range(n_windows):
            pred_end = T - (n_windows - 1 - w) * stride
            pred_start = pred_end - prediction_length
            if pred_start < min_context or pred_end > T:
                continue
            cutoffs.append(pred_start)
            ys.append(ts[pred_start:pred_end])
        if not cutoffs:
            continue
        series_full.append(ts)
        cutoff_indexes.append(cutoffs)
        targets.append(np.stack(ys, axis=0))  # (n_cutoffs, H, C)

    return series_full, cutoff_indexes, targets


def build_forecasting_data(
    series: List[np.ndarray],
    prediction_length: int,
    n_windows: int = 1,
    debug: bool = False,
) -> dict:
    """Build the shared train/test split fields of a forecasting data dict.

    Keeps everything but the last ``prediction_length * n_windows`` steps
    of each series as training context, then delegates the evaluation
    windows to :func:`make_forecasting_splits` (a single window in debug
    mode). Series shorter than ``prediction_length + 1`` are dropped.

    ``y_train`` is ``None``: forecasting is self-supervised, so solvers
    that fine-tune carve their own (context, target) windows out of
    ``X_train`` — handing out a fixed pair would either leak the test
    windows or duplicate a slice of ``X_train``.

    Returns a dict with ``X_train``, ``y_train``, ``X_test``, ``y_test``
    and ``cutoff_indexes``; datasets add their task-specific fields
    (metrics, freq, seasonality, ...) on top.
    """
    test_len = prediction_length * n_windows
    X_train, full_series = [], []
    for ts in series:
        if ts.shape[0] < prediction_length + 1:
            continue
        X_train.append(ts[:max(1, ts.shape[0] - test_len)])
        full_series.append(ts)

    if not full_series:
        raise ValueError("All series are shorter than prediction_length.")

    X_test, cutoff_indexes, y_test = make_forecasting_splits(
        full_series,
        prediction_length=prediction_length,
        n_windows=1 if debug else n_windows,
    )
    return dict(
        X_train=X_train,
        y_train=None,
        X_test=X_test,
        y_test=y_test,
        cutoff_indexes=cutoff_indexes,
    )
