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
