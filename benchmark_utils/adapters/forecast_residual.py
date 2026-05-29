"""Forecast-residual anomaly detection adapter.

Uses a forecasting model to predict ahead at strided positions in the test
series.  The anomaly score at each timestep is the absolute prediction error
(or norm across channels).

To keep the run fast we do not forecast one point at a time.  Instead, at a
cutoff ``t`` we forecast an adaptive horizon ``h = min(t // 2, max_horizon)``
steps and score all ``h`` predicted positions at once, then advance by ``h``.
The horizon therefore grows with the available context until it saturates at
``max_horizon`` (96).  Cutoffs sharing the same horizon are forecast in a
single batched call.

This is a zero-shot AD strategy: no labels are required.

Usage
-----
    adapter = ForecastResidualAdapter(forecaster)
    # no fit() needed for zero-shot
    scores = adapter.predict(x_test)   # (T,) anomaly scores
"""

import numpy as np

from .base import BaseTSFMAdapter


class ForecastResidualAdapter(BaseTSFMAdapter):
    """Anomaly scoring via adaptive-horizon forecast residuals.

    Parameters
    ----------
    forecaster : object exposing the batched forecasting predict API
        Its ``predict`` must accept an optional ``prediction_length`` override
        (see :class:`BaseTSFMAdapter`).
    max_horizon : int
        Upper bound on the forecast horizon at any cutoff (default 96).
    min_context : int
        Minimum number of past timesteps required before the first prediction.
    """

    def __init__(self, forecaster, max_horizon=96, min_context=10):
        self.forecaster = forecaster
        self.max_horizon = max_horizon
        self.min_context = min_context

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Score every timestep of a test series.

        Parameters
        ----------
        x : (T, C)

        Returns
        -------
        scores : (T,) float — higher means more anomalous.
            Timesteps before ``min_context`` receive score 0.
        """
        T = x.shape[0]
        scores = np.zeros(T, dtype=np.float32)

        # Adaptive-horizon stride plan: at cutoff t forecast h steps and
        # advance by h, so every position in [min_context, T) is scored once.
        groups = {}  # horizon h -> list of cutoff indexes
        t = self.min_context
        while t < T:
            h = min(t // 2, self.max_horizon)
            h = max(h, 1)
            groups.setdefault(h, []).append(t)
            t += h
        if not groups:
            return scores

        from benchmark_utils.inputs import ForecastInput

        for h, cutoffs in groups.items():
            try:
                output = self.forecaster.predict(
                    ForecastInput(x=[x], cutoff_indexes=[cutoffs]),
                    prediction_length=h,
                )
                preds = output.point[0]  # (n_cutoffs, h, C)
            except Exception:
                continue
            for k, c in enumerate(cutoffs):
                actual = x[c : c + h]
                n = actual.shape[0]  # < h only at the series tail
                err = np.abs(preds[k][:n] - actual)  # (n, C)
                scores[c : c + n] = np.mean(err, axis=1)
        return scores
