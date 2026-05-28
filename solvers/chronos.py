"""Chronos solver for the TSFM benchmark.

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
"""

import numpy as np
from benchopt import BaseSolver

from benchmark_utils.adapters.forecast_residual import ForecastResidualAdapter


SUPPORTED_TASKS = {"forecasting", "anomaly_detection"}


# ---------------------------------------------------------------------------
# Thin wrapper exposing the predict() interface expected by the objective
# ---------------------------------------------------------------------------

class _ChronosForecaster:
    """Wraps ChronosPipeline with the batched series+cutoffs predict API."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, x, cutoff_indexes, covariates, horizon):
        del covariates
        import torch

        results = []
        for series, cutoffs in zip(x, cutoff_indexes):
            series = np.asarray(series, dtype=np.float32)
            C = series.shape[1] if series.ndim == 2 else 1
            out = np.empty((len(cutoffs), horizon, C), dtype=np.float32)
            for k, cutoff in enumerate(cutoffs):
                hist = series[:cutoff]
                if hist.ndim == 1:
                    hist = hist[:, None]
                # Chronos expects (batch, time) — one channel at a time.
                for c in range(C):
                    context = torch.from_numpy(hist[:, c]).unsqueeze(0)
                    forecast = self.pipeline.predict(
                        context,
                        prediction_length=horizon,
                    )
                    f = forecast[0]
                    if f.ndim == 2:
                        f = f.median(dim=0).values
                    out[k, :, c] = f.numpy()
            results.append(out)
        return results


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
        forecaster = _ChronosForecaster(self._pipeline)

        if self.task == "forecasting":
            self._adapter = forecaster

        elif self.task == "anomaly_detection":
            self._adapter = ForecastResidualAdapter(
                forecaster, prediction_length=1
            )

    def get_result(self):
        return {"model": self._adapter}
