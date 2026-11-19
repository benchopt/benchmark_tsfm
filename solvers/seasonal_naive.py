"""Seasonal-naive forecasting baseline.

The forecast at horizon ``h`` is the value observed ``season_length`` steps
ago: for forecast index ``i`` (0-based within the horizon), the prediction
is ``hist[-season_length + (i mod season_length)]``. When the available
history is shorter than ``season_length``, the pattern falls back to
whatever history exists.

A common, calibrated baseline for any dataset with a known seasonal
period. With ``season_length=1`` it collapses to last-value persistence.

Besides the 0.5 (median = point forecast), the adapter emits a 9-level
quantile fan (0.1 to 0.9) replicating the native prediction intervals of
``statsforecast.SeasonalNaive`` (the GIFT-Eval official baseline): per
channel, ``sigma`` is the RMS of the in-sample one-step seasonal
differences ``y[t] - y[t - season_length]``, the spread at step ``i``
(0-based forecast index) is ``sigma_h = sigma * sqrt(k + 1)`` with
``k = floor(i / season_length)`` repeated cycles, and level ``q`` sits
at ``mean +/- norm.ppf(max(q, 1 - q)) * sigma_h``. The level 0.5 matches
the native ``lo-0``/``hi-0`` columns, i.e. the point itself.
"""

import numpy as np
from benchopt import BaseSolver
from scipy.stats import norm

from benchmark_utils.adapters.base import BaseTSFMAdapter
from benchmark_utils.inputs import ForecastInput
from benchmark_utils.outputs import ForecastOutput

SUPPORTED_TASKS = {"forecasting"}

# GIFT-Eval leaderboard distributional metrics (gluonts
# MeanWeightedSumQuantileLoss) use a 0.1..0.9 fan.
QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class _SeasonalNaiveForecaster(BaseTSFMAdapter):
    """Repeat the last ``season_length`` observations to fill the horizon
    with a normal-quantile fan around each point."""

    def __init__(self, prediction_length: int, season_length: int):
        if season_length < 1:
            raise ValueError(f"season_length must be >= 1, got {season_length}")
        self.prediction_length = prediction_length
        self.season_length = season_length
        levels = np.asarray(QUANTILE_LEVELS)
        # statsforecast native CI half-width: the interval of width
        # round(200 * (max(q, 1-q) - 0.5)) percent around the mean is the
        # normal quantile at max(q, 1-q); quantiles below 0.5 take the lo
        # side, above 0.5 the hi side.
        self._z = norm.ppf(np.maximum(levels, 1 - levels))  # (Q,)
        self._sign = np.where(levels < 0.5, -1.0, 1.0)[None, None, :]  # (1,1,Q)

    def predict(self, x: ForecastInput) -> ForecastOutput:
        quantiles = []
        for series, cutoffs in zip(x.x, x.cutoff_indexes):
            series = np.asarray(series)
            C = series.shape[1] if series.ndim == 2 else 1
            fan = np.empty(
                (len(cutoffs), self.prediction_length, C, len(QUANTILE_LEVELS))
            )
            for k, cutoff in enumerate(cutoffs):
                hist = series[:cutoff]
                n = hist.shape[0]
                season = min(self.season_length, n)
                pattern = hist[-season:]
                reps = int(np.ceil(self.prediction_length / season))
                point = np.tile(pattern, (reps, 1))[: self.prediction_length]
                if n > self.season_length:
                    resid = hist[self.season_length :] - hist[: n - self.season_length]
                    sigma = np.sqrt(
                        np.nansum(resid * resid, axis=0) / (n - self.season_length)
                    )
                else:
                    sigma = np.zeros(C)
                k_cycle = np.floor(
                    np.arange(self.prediction_length) / self.season_length
                )
                sigma_h = sigma[None, :] * np.sqrt(k_cycle + 1)[:, None]  # (H, C)
                fan[k] = (
                    point[:, :, None]
                    + self._sign * self._z[None, None, :] * sigma_h[:, :, None]
                )
            quantiles.append(fan)
        return ForecastOutput(quantiles=quantiles, quantile_levels=QUANTILE_LEVELS)


class Solver(BaseSolver):
    """Seasonal-naive baseline.

    Parameters
    ----------
    season_length : int or "auto"
        Number of past steps to repeat. ``1`` recovers last-value
        persistence. ``"auto"`` (the default) uses the dataset's own
        seasonality, ``meta["seasonality"]``.
    """

    name = "SeasonalNaive"

    requirements = []

    parameters = {
        "season_length": ["auto"],
    }

    def skip(self, task, **kwargs):
        if task not in SUPPORTED_TASKS:
            return True, f"SeasonalNaive does not support task={task!r}"
        return False, None

    def set_objective(self, X_train, y_train, task, **meta):
        self.task = task
        self.X_train = X_train
        self.y_train = y_train
        self.meta = meta

    def run(self, _):
        season_length = self.season_length
        if season_length == "auto":
            season_length = self.meta.get("seasonality", 1)
        self._adapter = _SeasonalNaiveForecaster(
            prediction_length=self.meta.get("prediction_length", 1),
            season_length=season_length,
        )

    def get_result(self):
        return {"model": self._adapter}
