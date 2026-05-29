"""Toto-2.0 solver for zero-shot forecasting and anomaly detection.

The benchmark objective expects solvers to return an adapter exposing
``predict(x: np.ndarray (T, C)) -> np.ndarray (H, C)``.  Toto-2.0 forecasts
quantiles for tensors shaped ``(batch, n_variates, time_steps)``; this solver
uses the median quantile as the point forecast.
"""

from benchopt import BaseSolver
import numpy as np

from benchmark_utils.adapters.base import BaseTSFMAdapter
from benchmark_utils.adapters.forecast_residual import ForecastResidualAdapter


SUPPORTED_TASKS = {"forecasting", "anomaly_detection"}


# ---------------------------------------------------------------------------
# Thin wrapper exposing the predict() interface expected by the objective
# ---------------------------------------------------------------------------

class _Toto2Forecaster(BaseTSFMAdapter):
    """Wraps Toto2Model to expose predict(x (T, C)) -> (H, C)."""

    def __init__(
        self,
        model,
        device,
        prediction_length,
        context_length=None,
        decode_block_size=None,
        patch_size=32,
    ):
        self.model = model
        self.device = device
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.decode_block_size = decode_block_size
        self.patch_size = patch_size

    def predict(self, x: np.ndarray) -> np.ndarray:
        import torch

        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]
        if self.context_length is not None:
            x = x[-self.context_length:]

        # Toto expects (batch, n_variates, time_steps).
        target_np = np.swapaxes(x, 0, 1)[None, :, :]
        finite_mask_np = np.isfinite(target_np)
        target_np = np.nan_to_num(target_np, nan=0.0, posinf=0.0, neginf=0.0)

        pad_len = (-target_np.shape[-1]) % self.patch_size
        if pad_len:
            target_np = np.pad(
                target_np,
                ((0, 0), (0, 0), (pad_len, 0)),
                mode="constant",
                constant_values=0.0,
            )
            finite_mask_np = np.pad(
                finite_mask_np,
                ((0, 0), (0, 0), (pad_len, 0)),
                mode="constant",
                constant_values=False,
            )

        has_missing_values = not bool(finite_mask_np.all())

        target = torch.from_numpy(target_np).to(self.device)
        target_mask = torch.from_numpy(finite_mask_np).to(
            self.device, dtype=torch.bool
        )
        series_ids = torch.zeros(
            target.shape[0],
            target.shape[1],
            dtype=torch.long,
            device=self.device,
        )

        with torch.inference_mode():
            quantiles = self.model.forecast(
                {
                    "target": target,
                    "target_mask": target_mask,
                    "series_ids": series_ids,
                },
                horizon=self.prediction_length,
                decode_block_size=self.decode_block_size,
                has_missing_values=has_missing_values,
            )

        # Quantiles are documented as (9, batch, n_variates, horizon)
        # Median forecast at quantile level 0.5
        median = quantiles[4, 0].detach().float().cpu().numpy()
        return np.swapaxes(median, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class Solver(BaseSolver):
    """Datadog Toto-2.0 zero-shot solver."""

    name = "Toto-2.0"

    requirements = [
        "pip::torch>=2.5",
        "pip::toto-2 @ git+https://github.com/DataDog/toto.git#subdirectory=toto2",
    ]

    sampling_strategy = "run_once"

    parameters = {
        "checkpoint": ["Datadog/Toto-2.0-2.5B"],
        "context_length": [512],
        "decode_block_size": [None],
        "patch_size": [32],
    }

    def skip(self, task, **kwargs):
        if task not in SUPPORTED_TASKS:
            return True, f"Toto-2.0 solver does not support task={task!r}"
        return False, None

    def set_objective(self, X_train, y_train, task, **meta):
        import torch
        from toto2 import Toto2Model

        self.task = task
        self.meta = meta

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if not hasattr(self, "_model") or self._loaded_checkpoint != self.checkpoint:
            self._model = Toto2Model.from_pretrained(self.checkpoint)
            self._model = self._model.to(self._device).eval()
            self._loaded_checkpoint = self.checkpoint

    def run(self, _):
        pred_len = self.meta.get("prediction_length", 1)
        forecaster = _Toto2Forecaster(
            self._model,
            self._device,
            prediction_length=pred_len,
            context_length=self.context_length,
            decode_block_size=self.decode_block_size,
            patch_size=self.patch_size,
        )

        if self.task == "forecasting":
            self._adapter = forecaster

        elif self.task == "anomaly_detection":
            self._adapter = ForecastResidualAdapter(
                forecaster, prediction_length=1
            )

    def get_result(self):
        return {"model": self._adapter}
