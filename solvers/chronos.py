"""Chronos solver for the TSFM benchmark (local inference).

Supports:
  - forecasting     : zero-shot via Chronos2Pipeline
  - classification  : linear probe on pooled encoder embeddings
  - anomaly_detection  : forecast-residual on top of the same forecaster

Model loading is done in ``set_objective`` (untimed). Inference batches
every (series, cutoff) pair into a single call — the pipeline accepts a
list of variable-length tensors and applies left-padding internally, so
all the per-cutoff work happens in one forward pass.
"""

from typing import Sequence

import numpy as np
import torch
from chronos.chronos2 import Chronos2Pipeline
from einops import rearrange

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
from benchmark_utils.base_solver import BaseTSFMSolver
from benchmark_utils.covariates import Covariates
from benchmark_utils.inputs import ForecastInput
from benchmark_utils.outputs import ForecastOutput

SUPPORTED_TASKS = {"forecasting", "classification", "anomaly_detection"}

POOLERS = {
    "mean": MeanPooler,
    "max": MaxPooler,
    "last": LastPooler,
}


class _ChronosForecaster(BaseTSFMAdapter):
    """Batched Chronos v1 adapter; quantiles are derived from sample draws."""

    DEFAULT_QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    def __init__(self, pipeline, prediction_length, quantile_levels=None):
        self.pipeline = pipeline
        self.prediction_length = prediction_length
        self.quantile_levels = quantile_levels or self.DEFAULT_QUANTILE_LEVELS

    # ------------------------------------------------------------------
    # Template method — subclasses override _build_inputs / _assemble
    # ------------------------------------------------------------------

    def predict(self, x: ForecastInput) -> ForecastOutput:
        inputs, layout, per_series_shape = self._build_inputs(x)
        if not inputs:
            return ForecastOutput(quantiles=[], quantile_levels=self.quantile_levels)

        with torch.no_grad():
            output = self.pipeline.predict(
                inputs,
                prediction_length=self.prediction_length,
            )
        return self._assemble_output(output, layout, per_series_shape)

    def _build_inputs(self, x):
        """Build list of 1-D tensors (one per channel) and track layout."""
        inputs = []
        layout = []  # (series_idx, cutoff_idx, channel_idx)
        per_series_shape = []  # (C, n_cutoffs)
        for series_idx, (series, cutoffs) in enumerate(zip(x.x, x.cutoff_indexes)):
            series = np.asarray(series, dtype=np.float32)
            if series.ndim == 1:
                series = series[:, None]
            _, C = series.shape
            per_series_shape.append((C, len(cutoffs)))
            for cutoff_idx, cutoff in enumerate(cutoffs):
                hist = series[:cutoff]
                for c in range(C):
                    inputs.append(torch.from_numpy(hist[:, c]))
                    layout.append((series_idx, cutoff_idx, c))
        return inputs, layout, per_series_shape

    def _assemble_output(self, samples, layout, per_series_shape):
        """Derive quantile fan from Monte-Carlo sample draws."""
        # samples: (n_inputs, num_samples, H)
        q_arr = np.quantile(
            samples.float().cpu().numpy(),
            q=list(self.quantile_levels),
            axis=1,
        ).transpose(1, 0, 2)  # (n_inputs, Q, H)

        Q = len(self.quantile_levels)
        per_series = [
            np.empty((n_cutoffs, Q, self.prediction_length, C), dtype=np.float32)
            for C, n_cutoffs in per_series_shape
        ]
        for i, (series_idx, cutoff_idx, c) in enumerate(layout):
            per_series[series_idx][cutoff_idx, :, :, c] = q_arr[i]

        return ForecastOutput(quantiles=per_series, quantile_levels=self.quantile_levels)



class _ChronosEmbedEncoder(UnpooledEncoder):
    """Default path — uses ``Chronos2Pipeline.embed``.

    Returns hidden states *after* ``encoder.final_layer_norm``.
    """

    def __init__(self, pipeline: Chronos2Pipeline):
        self.pipeline = pipeline

    def encode(self, X) -> np.ndarray:
        # X: (B, T, V) or (T, V).
        X = np.asarray(X, dtype=np.float32)
        batched = X.ndim == 3
        if not batched:
            X = X[None]  # (1, T, V)
        B, T, V = X.shape

        # Chronos is univariate — flatten B & V into the batch axis.
        flat = X.reshape(B * V, T)  # (B*V, T)
        with torch.no_grad():
            emb, _ = self.pipeline.embed(torch.from_numpy(flat))  # (B*V, T_tok, D)

        # (B*V, T_tok, D) -> (B, T_tok, V, D)
        return emb.float().cpu().numpy().reshape(B, -1, V, emb.shape[-1])


class _ChronosHookEncoder(UnpooledEncoder):
    """Layer-specific path — forward hook on ``encoder.block[layer]``.

    Returns the *pre-norm* hidden state at the chosen block. Negative
    indices are allowed (``-1`` = last block).
    """

    def __init__(self, pipeline: Chronos2Pipeline, layer: int):
        self.pipeline = pipeline
        n_blocks = len(pipeline.model.model.encoder.block)
        if not -n_blocks <= layer < n_blocks:
            raise IndexError(
                f"layer {layer} out of range for {n_blocks} encoder blocks"
            )
        self._block_idx = layer % n_blocks

    def encode(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        batched = X.ndim == 3
        if not batched:
            X = X[None]  # (1, T, V)
        B, T, V = X.shape

        flat = X.reshape(B * V, T)  # (B*V, T)
        context = torch.from_numpy(flat)
        token_ids, attn_mask, _ = self.pipeline.tokenizer.context_input_transform(
            context
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

        # (B*V, T_tok, D) -> (B, T_tok, V, D)
        return captured["h"].float().cpu().numpy().reshape(B, -1, V, captured["h"].shape[-1])


def ChronosEncoder(
    pipeline: Chronos2Pipeline, layer: int | None = None
) -> UnpooledEncoder:
    """Build a Chronos feature extractor.

    Parameters
    ----------
    pipeline : Chronos2Pipeline
        A loaded Chronos pipeline.
    layer : int, optional
        Encoder block index to read hidden states from. ``None`` (default)
        uses :meth:`Chronos2Pipeline.embed`, which returns post-final-norm
        states from the full encoder. An integer ``layer`` registers a
        forward hook on ``encoder.block[layer]`` and returns the pre-norm
        hidden state there. Negative indexing supported.

    Returns
    -------
    UnpooledEncoder
        Object exposing ``encode(x: np.ndarray (T, C)) -> np.ndarray
        (T_tok, C, D)``. Embeddings are *not* pooled.

    Notes
    -----
    ``ChronosEncoder(pipeline)`` and ``ChronosEncoder(pipeline, layer=-1)``
    differ only by ``encoder.final_layer_norm`` — they will be close but
    not identical.
    """
    if layer is None:
        return _ChronosEmbedEncoder(pipeline)
    return _ChronosHookEncoder(pipeline, layer)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class Solver(BaseTSFMSolver):
    """Chronos zero-shot solver.

    Parameters
    ----------
    model_size : str
        Chronos model variant: "tiny", "mini", "small", "base", "large".
    layer : int or None
        Encoder block index for classification embeddings. ``None`` uses
        ``Chronos2Pipeline.embed`` (post-final-norm).
    pooler : {"mean", "max", "last"}
        Pooling strategy over the time-token axis for classification.
    task_adaptation : str
        Per-task usage of the forecaster:
          ``"zeroshot"``          — direct forecasting (forecasting only)
          ``"forecast_residual"`` — anomaly score = forecast error (AD only)
    """

    name = "Chronos"

    requirements = ["pip::chronos-forecasting>=2.2", "pip::torch"]

    parameters = {
        "model_size": ["small"],
        "layer": [None],
        "pooler": ["mean"],
    }

    @property
    def supported_tasks(self):
        return SUPPORTED_TASKS

    def load_model(self, device, dtype):
        """Load Chronos-2 pipeline (cached if already loaded)."""
        model_id = f"autogluon/chronos-2-{self.model_size}"
        if self._loaded_model != model_id:
            self._pipeline = Chronos2Pipeline.from_pretrained(
                model_id,
                device_map=device,
                dtype=dtype,
            )
            self._loaded_model = model_id
        return self._pipeline

    def forecast_batch(self, inputs: list[torch.Tensor], covariates: Sequence[Covariates]) -> list[torch.Tensor]:
        with torch.no_grad():
            inputs_t = [x.T for x in inputs]
            preds = self.model.predict(inputs_t, prediction_length=self.meta["prediction_length"])
            return [
                rearrange(pred, "n_variates n_quantiles prediction_length -> prediction_length n_variates n_quantiles")
                for pred in preds
            ]

    def build_adapter(self, task, model):
        # TODO later: put that code in base_solver.py
        # and make it rely on .forecast(), .embed() and .time_embed() only, once those are all properly coded
        """Create task-specific adapter for Chronos."""
        pred_len = self.meta.get("prediction_length", 1)
        if task == "forecasting":
            return _ChronosForecaster(model, pred_len)

        elif task == "classification":
            base_encoder = ChronosEncoder(model, layer=self.layer)
            encoder = Encoder(base_encoder, POOLERS[self.pooler]())
            adapter = LinearProbeAdapter(
                encoder,
                task="classification",
                n_classes=self.meta.get("n_classes"),
            )
            adapter.fit(self.X_train, self.y_train)
            return adapter

        elif task == "anomaly_detection":
            return ForecastResidualAdapter(
                _ChronosForecaster(model, prediction_length=1),
                prediction_length=1,
            )

        else:
            raise ValueError(f"Unsupported task: {task}")
