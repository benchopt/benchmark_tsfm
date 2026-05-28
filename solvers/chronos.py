"""Chronos-2 solver for the TSFM benchmark (local inference).

Supports:
  - forecasting        : zero-shot via ``Chronos2Pipeline``
  - anomaly_detection  : forecast-residual on top of the same forecaster

Classification is not yet implemented; the solver skips that task.

Model loading is done in ``set_objective`` (untimed). Inference batches
every (series, cutoff) pair into a single ``Chronos2Pipeline.predict``
call — the pipeline accepts a list of variable-length tensors and
applies left-padding internally, so all the per-cutoff work happens in
one forward pass.
"""

import numpy as np
import torch
from benchopt import BaseSolver

from benchmark_utils.adapters.base import BaseTSFMAdapter
from benchmark_utils.adapters.forecast_residual import ForecastResidualAdapter
from benchmark_utils.inputs import ForecastInput
from benchmark_utils.outputs import ForecastOutput


SUPPORTED_TASKS = {"forecasting", "anomaly_detection"}


class _ChronosForecaster(BaseTSFMAdapter):
    """Batched Chronos-2 adapter returning a full quantile fan."""

    def __init__(self, pipeline, prediction_length):
        self.pipeline = pipeline
        self.prediction_length = prediction_length
        self.quantile_levels = tuple(float(q) for q in pipeline.quantiles)

    def predict(self, x: ForecastInput) -> ForecastOutput:
        inputs = []
        layout = []              # (series_idx, cutoff_idx) per input element
        per_series_shape = []    # (C, n_cutoffs) per series
        for series_idx, (series, cutoffs) in enumerate(zip(x.x, x.cutoff_indexes)):
            series = np.asarray(series, dtype=np.float32)
            if series.ndim == 1:
                series = series[:, None]
            _, C = series.shape
            per_series_shape.append((C, len(cutoffs)))
            for cutoff_idx, cutoff in enumerate(cutoffs):
                hist = series[:cutoff]                   # (T_cutoff, C)
                inputs.append(torch.from_numpy(hist.T))  # (C, T_cutoff)
                layout.append((series_idx, cutoff_idx))

        if not inputs:
            return ForecastOutput(quantiles=[], quantile_levels=self.quantile_levels)

        with torch.no_grad():
            forecast = self.pipeline.predict(
                inputs,
                prediction_length=self.prediction_length,
            )
        # forecast: list[(n_variates, Q, prediction_length)] aligned with `inputs`.

        Q = len(self.quantile_levels)
        per_series = [
            np.empty((n_cutoffs, Q, self.prediction_length, C), dtype=np.float32)
            for C, n_cutoffs in per_series_shape
        ]
        for (series_idx, cutoff_idx), pred in zip(layout, forecast):
            arr = pred.float().cpu().numpy()                # (C, Q, H)
            per_series[series_idx][cutoff_idx] = arr.transpose(1, 2, 0)
        return ForecastOutput(quantiles=per_series, quantile_levels=self.quantile_levels)


class Solver(BaseSolver):
    """Chronos-2 zero-shot solver.

    Parameters
    ----------
    model_size : str
        Chronos-2 variant suffix used in ``autogluon/chronos-2-{model_size}``.
    task_adaptation : str
        Per-task usage of the forecaster:
          ``"zeroshot"``          — direct forecasting (forecasting only)
          ``"forecast_residual"`` — anomaly score = forecast error (AD only)
    """

    name = "Chronos"

    requirements = ["pip::chronos-forecasting>=2.2,<3"]

    sampling_strategy = "run_once"

    parameters = {
        "model_size": ["small"],
        "task_adaptation": ["zeroshot"],
    }

    def skip(self, task, **kwargs):
        if task not in SUPPORTED_TASKS:
            return True, f"Chronos solver does not support task={task!r}"
        return False, None

    def set_objective(self, X_train, y_train, task, **meta):
        from chronos import Chronos2Pipeline

        self.task = task
        self.X_train = X_train
        self.meta = meta

        # bfloat16 is fine on CUDA but poorly supported on CPU / MPS;
        # fall back to float32 there so inference doesn't crash or stall.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model_id = f"autogluon/chronos-2-{self.model_size}"
        if not hasattr(self, "_pipeline") or self._loaded_model != model_id:
            self._pipeline = Chronos2Pipeline.from_pretrained(
                model_id,
                device_map=device,
                dtype=dtype,
            )
            self._loaded_model = model_id

    def run(self, _):
        pred_len = self.meta.get("prediction_length", 1)
        if self.task == "forecasting":
            self._adapter = _ChronosForecaster(self._pipeline, pred_len)
        elif self.task == "anomaly_detection":
            # AD uses one-step-ahead forecasts.
            self._adapter = ForecastResidualAdapter(
                _ChronosForecaster(self._pipeline, prediction_length=1),
                prediction_length=1,
            )

    def get_result(self):
        return {"model": self._adapter}
