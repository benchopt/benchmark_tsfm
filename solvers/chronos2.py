"""Chronos-2 solver for the TSFM benchmark.

Supports:
  - forecasting        : zero-shot via Chronos2Pipeline
  - anomaly_detection  : forecast-residual (zero-shot)

Classification is not yet implemented; the solver skips that task.

Model loading is done in ``set_objective`` (untimed).
Adaptation fitting is done in ``run`` (timed).

Adding a new task
-----------------
1. Add the task name to ``SUPPORTED_TASKS``.
2. In ``run``, instantiate the appropriate adapter from
   ``benchmark_utils.adapters`` (or implement a new one there).
"""

import numpy as np
import torch
from benchopt import BaseSolver
from chronos import Chronos2Pipeline

from benchmark_utils.adapters.forecast_residual import ForecastResidualAdapter

SUPPORTED_TASKS = {"forecasting", "anomaly_detection"}

MODEL_ID = "amazon/chronos-2"


# ---------------------------------------------------------------------------
# Thin wrapper exposing the predict() interface expected by the objective
# ---------------------------------------------------------------------------


class _ChronosForecaster:
    """Wraps Chronos2Pipeline to expose predict(x (T, C)) -> (H, C)."""

    def __init__(self, pipeline, prediction_length):
        self.pipeline = pipeline
        self.prediction_length = prediction_length
        # Chronos-2 returns quantile forecasts; locate the median.
        self._median_idx = list(pipeline.quantiles).index(0.5)

    def predict(self, x: np.ndarray) -> np.ndarray:

        # Chronos expects (batch, C, time) tensors.
        x = np.asarray(x, dtype=np.float32).T[None]  # (1, C, T)

        context = torch.from_numpy(x)
        forecast = self.pipeline.predict(
            context,
            prediction_length=self.prediction_length,
        )
        # forecast is a list of length batch; each entry has shape
        # (n_variates, n_quantiles, H). Take the median quantile and cast
        # to float32 (bfloat16 tensors can't be converted to numpy).
        out = forecast[0].float().cpu().numpy()  # (C, Q, H)
        return out[:, self._median_idx, :].T  # (H, C)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class Solver(BaseSolver):
    """Chronos-2 zero-shot solver (fixed model: amazon/chronos-2)."""

    name = "Chronos2"

    requirements = ["pip::chronos-forecasting>=2.0", "pip::torch"]

    sampling_strategy = "run_once"

    parameters = {}

    def skip(self, task, **kwargs):
        if task not in SUPPORTED_TASKS:
            return True, f"Chronos2 solver does not support task={task!r}"
        return False, None

    # ------------------------------------------------------------------

    def set_objective(self, X_train, y_train, task, **meta):

        self.task = task
        self.X_train = X_train
        self.meta = meta

        if not hasattr(self, "_pipeline"):
            self._pipeline = Chronos2Pipeline.from_pretrained(
                MODEL_ID,
                device_map="auto",
                dtype=torch.bfloat16,
            )

    def run(self, _):
        pred_len = self.meta.get("prediction_length", 1)
        forecaster = _ChronosForecaster(self._pipeline, pred_len)

        if self.task == "forecasting":
            self._adapter = forecaster

        elif self.task == "anomaly_detection":
            self._adapter = ForecastResidualAdapter(
                forecaster, prediction_length=1
            )

    def get_result(self):
        return {"model": self._adapter}
