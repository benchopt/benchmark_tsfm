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

from benchmark_utils.covariates import Covariates

# Shortest history to forecast from, as fev-bench's m5_1M task.
MIN_CONTEXT = 6


def impute_nan(series: np.ndarray) -> np.ndarray:
    """Impute NaN targets as gluonts LastValueImputation, used by the
    official GIFT-Eval notebooks: forward-fill each channel, flat-fill
    leading NaNs with the first valid value, 0.0 for all-NaN channels.

    Parameters
    ----------
    series : np.ndarray (T, C)
        One series, time first.

    Returns
    -------
    np.ndarray (T, C)
        ``series`` itself when it has no NaN, else an imputed copy.
    """
    if not np.isnan(series).any():
        return series
    series = series.copy()
    for c in range(series.shape[1]):
        col = series[:, c]
        nan = np.isnan(col)
        if not nan.any() or nan.all():
            if nan.all():
                col[:] = 0.0
            continue
        idx = np.where(nan, 0, np.arange(len(col)))
        np.maximum.accumulate(idx, out=idx)
        col[nan] = col[idx[nan]]
        first = int(np.argmax(~nan))
        col[:first] = col[first]
    return series


def windows_by_distance_to_end(
    series_list: List[np.ndarray], cutoff_indexes: List[List[int]]
) -> List[np.ndarray]:
    """Group the evaluation windows by their distance to the series end.

    Windows at the same distance form one fev-bench evaluation window: they
    share a cutoff date when the series end together.

    Parameters
    ----------
    series_list : list of np.ndarray (T_i, C)
        The full series.
    cutoff_indexes : list of list of int
        For each series, the timesteps at which a window starts.

    Returns
    -------
    list of np.ndarray of int
        One array per distance, sorted by distance, of indexes into the
        windows flattened series-major then cutoff-minor.
    """
    dist = np.array(
        [
            len(series) - cutoff
            for series, cutoffs in zip(series_list, cutoff_indexes)
            for cutoff in cutoffs
        ]
    )
    return [np.flatnonzero(dist == d) for d in np.unique(dist)]


def make_forecasting_splits(
    series_list: List[np.ndarray],
    prediction_length: int,
    n_windows: int = 1,
    stride: Optional[int] = None,
    min_context: int = MIN_CONTEXT,
) -> Tuple[List[np.ndarray], List[List[int]], List[np.ndarray]]:
    """Create rolling-window evaluation cutoffs from a list of time series.

    Parameters
    ----------
    series_list : list of np.ndarray (T_i, C)
        The full series, time first.
    prediction_length : int
    n_windows : int
        Number of rolling evaluation windows per series.
    stride : int or None
        Step between consecutive prediction points.
        Defaults to ``prediction_length`` (non-overlapping).
    min_context : int
        Minimum context length required before the first prediction point.
        Windows with a shorter history are skipped.
    """
    if stride is None:
        stride = prediction_length

    series_full: List[np.ndarray] = []
    cutoff_indexes: List[List[int]] = []
    targets: List[np.ndarray] = []

    for ts in series_list:
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
    series_list: List[np.ndarray],
    prediction_length: int,
    n_windows: int = 1,
    debug: bool = False,
    min_context: int = MIN_CONTEXT,
    covariates: Optional[Covariates] = None,
) -> dict:
    """Build the shared train/test split fields of a forecasting data dict.

    Keeps everything but the last ``prediction_length * n_windows`` steps
    of each series as training context, then delegates the evaluation
    windows to :func:`make_forecasting_splits` (a single window in debug
    mode). Windows with fewer than ``min_context`` history steps are skipped,
    and series with no window left are dropped.

    NaN targets would propagate through solver histories and metrics, so
    they are imputed with :func:`impute_nan`. Their locations are then
    returned as ``nan_mask``, one ``(T_i, C)`` boolean array per series,
    for the metrics to score the observed points only.

    ``y_train`` is ``None``: forecasting is self-supervised, so solvers
    that fine-tune carve their own (context, target) windows out of
    ``X_train`` — handing out a fixed pair would either leak the test
    windows or duplicate a slice of ``X_train``.

    Parameters
    ----------
    series_list : list of np.ndarray (T_i, C)
        The full series, time first, possibly with NaN.
    prediction_length : int
    n_windows : int
        Number of rolling evaluation windows per series.
    debug : bool
        If True, keep a single window per series.
    min_context : int
        Minimum history before a window.
    covariates : Covariates or None
        Covariates of every series of ``series_list``.

    Returns
    -------
    dict
        ``X_train``, ``y_train``, ``X_test``, ``y_test`` and
        ``cutoff_indexes``, plus ``covariates`` of the kept series when given
        and ``nan_mask`` when a target has NaN. Datasets add their
        task-specific fields (metrics, freq, seasonality, ...) on top.
    """
    test_len = prediction_length * n_windows
    kept = [
        i
        for i, ts in enumerate(series_list)
        if ts.shape[0] >= prediction_length + min_context
    ]
    nan_mask = [np.isnan(series_list[i]) for i in kept]
    full_series = [impute_nan(series_list[i]) for i in kept]
    X_train = [ts[: max(1, ts.shape[0] - test_len)] for ts in full_series]

    if not full_series:
        raise ValueError("All series are shorter than prediction_length.")

    X_test, cutoff_indexes, y_test = make_forecasting_splits(
        full_series,
        prediction_length=prediction_length,
        n_windows=1 if debug else n_windows,
        min_context=min_context,
    )
    data = dict(
        X_train=X_train,
        y_train=None,
        X_test=X_test,
        y_test=y_test,
        cutoff_indexes=cutoff_indexes,
    )
    if covariates is not None:
        data["covariates"] = covariates.select(kept)
    if any(m.any() for m in nan_mask):
        data["nan_mask"] = nan_mask
    return data
