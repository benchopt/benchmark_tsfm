"""Forecast-residual anomaly detection adapter.

Uses a forecasting model to predict the next step at every position in the
test series.  The anomaly score at each timestep is the absolute prediction
error (or norm across channels).

This is a zero-shot AD strategy: no labels are required.

Usage
-----
    adapter = ForecastResidualAdapter(forecaster, prediction_length=1)
    # no fit() needed for zero-shot
    scores = adapter.predict(x_test)   # (T,) anomaly scores
"""

from collections import defaultdict

import numpy as np
from .base import BaseTSFMAdapter


class ForecastResidualAdapter(BaseTSFMAdapter):
    """Anomaly scoring via one-step-ahead forecast residuals.

    Parameters
    ----------
    forecaster : object with
        ``predict(context: np.ndarray (T, C)) -> np.ndarray (H, C)``
    prediction_length : int
        Maximum number of steps predicted at each position (default 1).
        Early on, the horizon is capped at half the available context so we
        don't forecast far beyond what the context can support; it grows with
        the context until it reaches ``prediction_length``.
    min_context : int
        Minimum number of past timesteps required before the first prediction.
    """

    def __init__(self, forecaster, prediction_length=1, min_context=10):
        self.forecaster = forecaster
        self.prediction_length = prediction_length
        self.min_context = min_context

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Score every timestep of a test series.

        The series is scored in non-overlapping blocks.  At position ``t`` we
        forecast ``h = min(t // 2, prediction_length)`` steps from the context
        ``x[:t]``, assign each forecast step its own residual, then advance by
        ``h``.  This makes ~``T / h`` model calls instead of one per timestep.

        The block schedule is deterministic, so blocks that share the same
        horizon ``h`` are forecast together in a single batched call when the
        forecaster exposes ``predict_batch``.  Because ``h`` reaches its cap
        early, almost every block lands in one group — collapsing the bulk of
        the work into a single batch.

        Parameters
        ----------
        x : (T, C)

        Returns
        -------
        scores : (T,) float — higher means more anomalous.
            Timesteps before ``min_context`` receive score 0.
        """
        T, C = x.shape
        scores = np.zeros(T, dtype=np.float32)

        # Build the (start, horizon) block schedule.  The horizon grows with
        # the context (half of it) until it reaches the configured maximum, and
        # never forecasts past the end of the series.
        blocks_by_h = defaultdict(list)
        t = self.min_context
        while t < T:
            h = max(min(t // 2, self.prediction_length, T - t), 1)
            blocks_by_h[h].append(t)
            t += h

        for h, starts in blocks_by_h.items():
            contexts = [x[:s] for s in starts]  # each (s, C)
            try:
                preds = self._forecast_batch(contexts, h)  # list of (h, C)
            except Exception:
                continue  # leave this group's blocks at score 0
            for s, pred in zip(starts, preds):
                pred = np.asarray(pred).reshape(h, -1)
                actual = x[s : s + h]  # (h, C)
                # Per-step residual: mean absolute error across channels.
                scores[s : s + h] = np.mean(np.abs(pred - actual), axis=1)

        return scores

    def _forecast_batch(self, contexts, h):
        """Forecast ``h`` steps for each context, batching when possible."""
        if hasattr(self.forecaster, "predict_batch"):
            return self.forecaster.predict_batch(contexts, prediction_length=h)
        return [self.forecaster.predict(c, prediction_length=h) for c in contexts]
