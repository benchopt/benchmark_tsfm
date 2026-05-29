"""Chronos-2 solver for the TSFM benchmark (local inference).

Supports:
  - forecasting     : zero-shot via Chronos2Pipeline
  - classification  : linear probe on pooled encoder embeddings
  - anomaly_detection  : forecast-residual on top of the same forecaster

Model loading is done in ``set_objective`` (untimed). Inference batches
every (series, cutoff) pair into a single ``Chronos2Pipeline.predict``
call — the pipeline accepts a list of variable-length tensors and
applies left-padding internally, so all the per-cutoff work happens in
one forward pass.
"""

import numpy as np
import torch
from benchopt import BaseSolver
from chronos import Chronos2Pipeline

from benchmark_utils.adapters import (
    Encoder,
    LinearProbeAdapter,
    UnpooledEncoder,
)
from benchmark_utils.adapters.forecast_residual import ForecastResidualAdapter
from benchmark_utils.outputs import ForecastOutput
from .chronos import (
    _ChronosForecaster,
    POOLERS,
    SUPPORTED_TASKS,
)


# ---------------------------------------------------------------------------
# Chronos-2 encoders — embed() has a different signature than Chronos v1:
# Chronos2Pipeline.embed takes (B, V, T) and returns List[(V, T_tok, D)].
# ---------------------------------------------------------------------------


def _to_context(x):
    """Reshape ``(T, V)`` or ``(B, T, V)`` to Chronos-2 input ``(B, V, T)``."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[None]
    return x.transpose(0, 2, 1)


class _Chronos2Forecaster(_ChronosForecaster):
    """Chronos-2 variant — uses native quantile output from the pipeline."""

    def __init__(self, pipeline, prediction_length):
        self.pipeline = pipeline
        self.prediction_length = prediction_length
        self.quantile_levels = tuple(float(q) for q in pipeline.quantiles)

    def _build_inputs(self, x):
        """Build (C, T) tensors (all channels together); layout omits channel idx."""
        inputs = []
        layout = []  # (series_idx, cutoff_idx)
        per_series_shape = []  # (C, n_cutoffs)
        for series_idx, (series, cutoffs) in enumerate(zip(x.x, x.cutoff_indexes)):
            series = np.asarray(series, dtype=np.float32)
            if series.ndim == 1:
                series = series[:, None]
            _, C = series.shape
            per_series_shape.append((C, len(cutoffs)))
            for cutoff_idx, cutoff in enumerate(cutoffs):
                hist = series[:cutoff]
                inputs.append(torch.from_numpy(hist.T))  # (C, T_cutoff)
                layout.append((series_idx, cutoff_idx))
        return inputs, layout, per_series_shape

    def _assemble_output(self, forecast, layout, per_series_shape):
        """Use quantile tensors directly from Chronos-2 pipeline."""
        # forecast: list[(n_variates, Q, prediction_length)]
        Q = len(self.quantile_levels)
        per_series = [
            np.empty((n_cutoffs, Q, self.prediction_length, C), dtype=np.float32)
            for C, n_cutoffs in per_series_shape
        ]
        for (series_idx, cutoff_idx), pred in zip(layout, forecast):
            arr = pred.float().cpu().numpy()  # (C, Q, H)
            per_series[series_idx][cutoff_idx] = arr.transpose(1, 2, 0)
        return ForecastOutput(quantiles=per_series, quantile_levels=self.quantile_levels)


class _Chronos2EmbedEncoder(UnpooledEncoder):
    """Uses ``Chronos2Pipeline.embed`` which returns a list of tensors."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def encode(self, X) -> np.ndarray:
        context = _to_context(X)  # (B, V, T)
        with torch.no_grad():
            embeddings, _ = self.pipeline.embed(context)  # list[(V, T_tok, D)]
        stacked = torch.stack(list(embeddings))  # (B, V, T_tok, D)
        return stacked.transpose(1, 2).float().cpu().numpy()  # (B, T_tok, V, D)


class _Chronos2HookEncoder(UnpooledEncoder):
    """Forward hook on ``encoder.block[layer]``."""

    def __init__(self, pipeline, layer: int):
        self.pipeline = pipeline
        n_blocks = len(pipeline.model.model.encoder.block)
        if not -n_blocks <= layer < n_blocks:
            raise IndexError(
                f"layer {layer} out of range for {n_blocks} encoder blocks"
            )
        self._block_idx = layer % n_blocks

    def encode(self, X) -> np.ndarray:
        context = _to_context(X)  # (B, V, T)
        token_ids, attn_mask, _ = self.pipeline.tokenizer.context_input_transform(
            torch.from_numpy(context)
        )
        device = self.pipeline.model.device
        token_ids = token_ids.to(device)
        attn_mask = attn_mask.to(device)

        encoder = self.pipeline.model.model.encoder
        captured = {}

        def _hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["h"] = hidden.detach()

        handle = encoder.block[self._block_idx].register_forward_hook(_hook)
        try:
            with torch.no_grad():
                encoder(input_ids=token_ids, attention_mask=attn_mask)
        finally:
            handle.remove()

        return captured["h"].float().cpu().numpy()


def _Chronos2Encoder(pipeline, layer=None):
    """Build a Chronos-2 feature extractor."""
    if layer is None:
        return _Chronos2EmbedEncoder(pipeline)
    return _Chronos2HookEncoder(pipeline, layer)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class Solver(BaseSolver):
    """Chronos-2 zero-shot solver.

    Parameters
    ----------
    model_size : str
        Chronos-2 model variant: "tiny", "mini", "small", "base", "large".
    layer : int or None
        Encoder block index for classification embeddings. ``None`` uses
        ``Chronos2Pipeline.embed`` (post-final-norm).
    pooler : {"mean", "max", "last"}
        Pooling strategy over the time-token axis for classification.
    """

    name = "Chronos2"

    requirements = ["pip::chronos-forecasting>=2.2,<3"]

    sampling_strategy = "run_once"

    parameters = {
        "model_size": ["small"],
        "layer": [None],
        "pooler": ["mean"],
    }

    def skip(self, task, **kwargs):
        if task not in SUPPORTED_TASKS:
            return True, f"Chronos2 solver does not support task={task!r}"
        return False, None

    def set_objective(self, X_train, y_train, task, **meta):
        self.task = task
        self.X_train = X_train
        self.y_train = y_train
        self.meta = meta

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
            self._adapter = _Chronos2Forecaster(self._pipeline, pred_len)

        elif self.task == "classification":
            base_encoder = _Chronos2Encoder(self._pipeline, layer=self.layer)
            encoder = Encoder(base_encoder, POOLERS[self.pooler]())
            adapter = LinearProbeAdapter(
                encoder,
                task="classification",
                n_classes=self.meta.get("n_classes"),
            )
            adapter.fit(self.X_train, self.y_train)
            self._adapter = adapter

        elif self.task == "anomaly_detection":
            self._adapter = ForecastResidualAdapter(
                _Chronos2Forecaster(self._pipeline, prediction_length=1),
                prediction_length=1,
            )

    def get_result(self):
        return {"model": self._adapter}
