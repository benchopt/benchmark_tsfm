"""Seasonal-naive forecasting baseline.

The forecast at horizon ``h`` is the value observed ``season_length`` steps
ago — i.e. for forecast index ``i`` (0-based within the horizon), the
prediction is ``hist[-season_length + (i mod season_length)]``. When the
available history is shorter than ``season_length``, the pattern falls
back to whatever history exists.

A common, calibrated baseline for any dataset with a known seasonal
period. With ``season_length=1`` it collapses to last-value persistence.
"""

import numpy as np
from benchopt import BaseSolver

from benchmark_utils.adapters.base import BaseTSFMAdapter
from benchmark_utils.inputs import ForecastInput
from benchmark_utils.outputs import ForecastOutput


SUPPORTED_TASKS = {"forecasting"}


class _SeasonalNaiveForecaster(BaseTSFMAdapter):
    """Repeat the last ``season_length`` observations to fill the horizon."""

    def __init__(self, prediction_length: int, season_length: int):
        if season_length < 1:
            raise ValueError(f"season_length must be >= 1, got {season_length}")
        self.prediction_length = prediction_length
        self.season_length = season_length

    def predict(self, x: ForecastInput) -> ForecastOutput:
        quantiles = []
        for series, cutoffs in zip(x.x, x.cutoff_indexes):
            series = np.asarray(series)
            C = series.shape[1] if series.ndim == 2 else 1
            preds = np.empty((len(cutoffs), self.prediction_length, C), dtype=np.float32)
            for k, cutoff in enumerate(cutoffs):
                hist = series[:cutoff]
                season = min(self.season_length, hist.shape[0])
                pattern = hist[-season:]
                reps = int(np.ceil(self.prediction_length / season))
                preds[k] = np.tile(pattern, (reps, 1))[:self.prediction_length]
            quantiles.append(preds[:, None, :, :])
        return ForecastOutput(quantiles=quantiles, quantile_levels=(0.5,))


class Solver(BaseSolver):
    """Seasonal-naive baseline.

    Parameters
    ----------
    season_length : int
        Number of past steps to repeat. ``1`` recovers last-value
        persistence; common picks are ``7`` (daily → weekly), ``12``
        (monthly → yearly), ``24`` (hourly → daily), ``52`` (weekly →
        yearly).
    """

    name = "SeasonalNaive"

    requirements = []

    sampling_strategy = "run_once"

    parameters = {
        "season_length": [1, 7, 12, 24],
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
        self._adapter = _SeasonalNaiveForecaster(
            prediction_length=self.meta.get("prediction_length", 1),
            season_length=self.season_length,
        )

    def get_result(self):
        return {"model": self._adapter}
