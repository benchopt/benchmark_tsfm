"""Chronos solver for the TSFM benchmark.

Uses the Uses the official ``mantis-tsfm`` API to load a pretrained Mantis checkpoint,
extract embeddings with ``MantisTrainer.transform``, and train a Random Forest
classifier on top.

Supports:
  - forecasting        : zero-shot via ChronosPipeline
  - anomaly_detection  : forecast-residual (zero-shot)

Classification is not yet implemented; the solver skips that task.

Model loading is done in ``set_objective`` (untimed).
Adaptation fitting is done in ``run`` (timed).

Adding a new task
-----------------
1. Add the task name to ``SUPPORTED_TASKS``.
2. In ``run``, instantiate the appropriate adapter from
   ``benchmark_utils.adapters`` (or implement a new one there).

References:
    https://github.com/amazon-science/chronos-forecasting
"""

import numpy as np
from benchopt import BaseSolver

from benchmark_utils.adapters.forecast_residual import ForecastResidualAdapter


SUPPORTED_TASKS = {"forecasting", "anomaly_detection"}


# ---------------------------------------------------------------------------
# Thin wrapper exposing the predict() interface expected by the objective
# ---------------------------------------------------------------------------


class _ChronosForecaster:
    """Wraps ChronosPipeline to expose predict(x (T, C)) -> (H, C)."""

    def __init__(self, pipeline, prediction_length):
        self.pipeline = pipeline
        self.prediction_length = prediction_length

    def predict(self, x: np.ndarray) -> np.ndarray:
        import torch

        x = np.asarray(x, dtype=np.float32)  # (T, C)
        C = x.shape[1]

        # Chronos expects (batch, time) tensors — one channel at a time,
        # then stack.
        preds = []
        for c in range(C):
            context = torch.from_numpy(x[:, c]).unsqueeze(0)  # (1, T)
            forecast = self.pipeline.predict(
                context,
                prediction_length=self.prediction_length,
            )
            # forecast: (1, n_samples, H) for sample-based pipelines,
            # or (1, H) for point pipelines — take median.
            f = forecast[0]
            if f.ndim == 2:  # (n_samples, H) → median
                f = f.median(dim=0).values
            preds.append(f.numpy())  # (H,)

        return np.stack(preds, axis=-1).astype(np.float32)  # (H, C)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class Solver(BaseSolver):
    """Chronos zero-shot solver.

    Parameters
    ----------
    model_size : str
        Chronos model variant: "tiny", "mini", "small", "base", "large".
    task_adaptation : str
        How to use Chronos for each task:
          "zeroshot"          — direct forecasting API (forecasting only)
          "forecast_residual" — anomaly score = forecast error (AD only)
    """

    name = "Chronos"

    requirements = ["pip::chronos-forecasting>=1.4", "pip::torch"]

    sampling_strategy = "run_once"

    parameters = {
        "model_size": ["small"],
        "task_adaptation": ["zeroshot"],
    }

    def skip(self, task, **kwargs):
        if task not in SUPPORTED_TASKS:
            return True, f"Chronos solver does not support task={task!r}"
        return False, None

    # ------------------------------------------------------------------

    def set_objective(self, X_train, y_train, task, **meta):
        import torch
        from chronos import ChronosPipeline

        self.task = task
        self.X_train = X_train
        self.meta = meta

        # Load model once; reuse across consecutive dataset configs.
        model_id = f"amazon/chronos-t5-{self.model_size}"
        if not hasattr(self, "_pipeline") or self._loaded_model != model_id:
            self._pipeline = ChronosPipeline.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            self._loaded_model = model_id

    def run(self, _):
        pred_len = self.meta.get("prediction_length", 1)
        forecaster = _ChronosForecaster(self._pipeline, pred_len)

        if self.task == "forecasting":
            self._adapter = forecaster

        elif self.task == "anomaly_detection":
            self._adapter = ForecastResidualAdapter(forecaster, prediction_length=1)

    def get_result(self):
        return {"model": self._adapter}
