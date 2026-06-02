"""Toto-2.0 solver for the TSFM benchmark (local inference).

Supports:
  - forecasting     : zero-shot via Toto2Model
  - classification  : linear probe on pooled transformer patch embeddings
  - anomaly_detection  : forecast-residual on top of the same forecaster

References:
    https://github.com/datadog/toto
"""

from benchopt import BaseSolver

from toto2 import Toto2Model
import numpy as np
import torch


from benchmark_utils.adapters import (
    Encoder,
    LastPooler,
    LinearProbeAdapter,
    MaxPooler,
    MeanPooler,
    UnpooledEncoder,
)
from benchmark_utils.adapters.base import BaseTSFMAdapter
from benchmark_utils.adapters.forecast_residual import ForecastResidualAdapter
from benchmark_utils.inputs import ForecastInput
from benchmark_utils.outputs import ForecastOutput

SUPPORTED_TASKS = {"forecasting", "classification", "anomaly_detection"}

POOLERS = {
    "mean": MeanPooler,
    "max": MaxPooler,
    "last": LastPooler,
}


# ---------------------------------------------------------------------------
# Thin wrapper exposing the predict() interface expected by the objective
# ---------------------------------------------------------------------------


class _Toto2Forecaster(BaseTSFMAdapter):
    """Toto2Model adapter for the forecasting contract."""

    quantile_levels = tuple(float(q) / 10 for q in range(1, 10))

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

    def _forecast_context(self, context: np.ndarray) -> np.ndarray:
        import torch

        x = np.asarray(context, dtype=np.float32)
        if self.context_length is not None:
            x = x[-self.context_length :]

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
        target_mask = torch.from_numpy(finite_mask_np).to(self.device, dtype=torch.bool)
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

        return quantiles[:, 0].detach().float().cpu().numpy().transpose(0, 2, 1)

    def predict(self, x: ForecastInput) -> ForecastOutput:
        per_series = []

        for series, cutoffs in zip(x.x, x.cutoff_indexes):
            series = np.asarray(series, dtype=np.float32)
            if series.ndim == 1:
                series = series[:, None]

            C = series.shape[1]
            forecasts = np.empty(
                (
                    len(cutoffs),
                    len(self.quantile_levels),
                    self.prediction_length,
                    C,
                ),
                dtype=np.float32,
            )
            for cutoff_idx, cutoff in enumerate(cutoffs):
                forecasts[cutoff_idx] = self._forecast_context(series[:cutoff])

            per_series.append(forecasts)

        return ForecastOutput(
            quantiles=per_series,
            quantile_levels=self.quantile_levels,
        )


class _Toto2EmbedEncoder(UnpooledEncoder):
    """Use Toto-2 transformer patch states as sequence embeddings.

    ``layer=None`` captures the final output of ``model.transformer``
    after its output norm. An integer ``layer`` registers a forward hook
    on ``model.transformer.layers[layer]`` and returns the output of that
    selected Toto transformer layer. Negative indexing is supported
    (``-1`` = last layer).
    """

    def __init__(self, model, device, context_length=None, layer=None):
        self.model = model
        self.device = device
        self.context_length = context_length
        self.layer = layer
        self.patch_size = model.config.patch_size

    def _prepare_batch(self, X):
        x = np.asarray(X, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :, None]
        elif x.ndim == 2:
            x = x[None]
        elif x.ndim != 3:
            raise ValueError(
                "Toto-2 classification expects input with shape "
                f"(time, variates) or (batch, time, variates); got {x.shape}."
            )

        if self.context_length is not None:
            x = x[:, -self.context_length :, :]

        # Toto expects (batch, variates, time_steps).
        batch = x.transpose(0, 2, 1)
        mask = np.isfinite(batch)
        batch = np.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0)

        pad_len = (-batch.shape[-1]) % self.patch_size
        if pad_len:
            batch = np.pad(
                batch,
                ((0, 0), (0, 0), (pad_len, 0)),
                mode="constant",
                constant_values=0.0,
            )
            mask = np.pad(
                mask,
                ((0, 0), (0, 0), (pad_len, 0)),
                mode="constant",
                constant_values=False,
            )

        return batch, mask

    def encode(self, X) -> np.ndarray:
        import torch

        target_np, target_mask_np = self._prepare_batch(X)

        target = torch.from_numpy(target_np).to(self.device)
        target_mask = torch.from_numpy(target_mask_np).to(self.device, dtype=torch.bool)
        cpm_mask = torch.ones_like(target_mask)
        series_ids = torch.zeros(
            target.shape[0],
            target.shape[1],
            dtype=torch.long,
            device=self.device,
        )
        captured_output = {}

        if self.layer is None:
            hook_module = self.model.transformer
        else:
            n_layers = len(self.model.transformer.layers)
            if not -n_layers <= self.layer < n_layers:
                raise IndexError(
                    f"layer {self.layer} out of range for {n_layers} "
                    "Toto-2 transformer layers"
                )
            hook_module = self.model.transformer.layers[self.layer % n_layers]

        def _capture_output(_module, _inputs, output):
            captured_output["hidden_state"] = output.detach()

        handle = hook_module.register_forward_hook(_capture_output)
        with torch.inference_mode():
            try:
                self.model(
                    target=target,
                    target_mask=target_mask,
                    cpm_mask=cpm_mask,
                    series_ids=series_ids,
                )
            finally:
                handle.remove()

        embeddings = captured_output["hidden_state"]
        if embeddings.ndim != 4:
            raise ValueError(
                "Toto-2 transformer output should have shape "
                f"(batch, variates, patches, dim); got {tuple(embeddings.shape)}."
            )

        # Toto: (B, C, T_patch, D) -> benchmark: (B, T_patch, C, D).
        return embeddings.transpose(1, 2).float().cpu().numpy()


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
        "layer": [None],
        "pooler": ["mean"],
        "classifier": ["log_reg"],
        "penalty": ["l2"],
        "C": [1.0],
        "alpha": [1.0],
        "n_estimators": [100],
    }

    def skip(self, task, **kwargs):
        if task not in SUPPORTED_TASKS:
            return True, f"Toto-2.0 solver does not support task={task!r}"
        return False, None

    def set_objective(self, X_train, y_train, task, **meta):
        self.task = task
        self.X_train = X_train
        self.y_train = y_train
        self.meta = meta

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        should_reload = (
            not hasattr(self, "_model") or self._loaded_checkpoint != self.checkpoint
        )
        if should_reload:
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

        elif self.task == "classification":
            base_encoder = _Toto2EmbedEncoder(
                self._model,
                self._device,
                context_length=self.context_length,
                layer=self.layer,
            )
            encoder = Encoder(base_encoder, POOLERS[self.pooler]())
            adapter = LinearProbeAdapter(
                encoder,
                task="classification",
                n_classes=self.meta.get("n_classes"),
            )
            adapter.fit(self.X_train, self.y_train)
            self._adapter = adapter

        elif self.task == "anomaly_detection":
            self._adapter = ForecastResidualAdapter(forecaster, prediction_length=1)

    def get_result(self):
        return {"model": self._adapter}
