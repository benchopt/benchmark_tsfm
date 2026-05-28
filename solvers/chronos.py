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
from chronos import Chronos2Pipeline, ChronosPipeline
from benchmark_utils.adapters import UnpooledEncoder
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

def _to_context_tensor(x: np.ndarray):
    """Convert ``(T,)`` or ``(T, C)`` array to ``(C, T)`` float32 tensor.

    Channels become the batch dim so a single Chronos forward pass covers
    all of them — Chronos is univariate, so channels are independent.
    """

    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    return torch.from_numpy(x.T)


class _ChronosEmbedEncoder(UnpooledEncoder):
    """Default path — uses ``ChronosPipeline.embed``.

    Returns hidden states *after* ``encoder.final_layer_norm``.
    """

    def __init__(self, pipeline: ChronosPipeline):
        self.pipeline = pipeline

    def encode(self, x: np.ndarray) -> np.ndarray:
        context = _to_context_tensor(x)  # (C, T)
        with torch.no_grad():
            embeddings, _ = self.pipeline.embed(context)  # (C, T_tok, D)
        # (C, T_tok, D) -> (T_tok, C, D)
        return embeddings.transpose(0, 1).float().cpu().numpy()


class _ChronosHookEncoder(UnpooledEncoder):
    """Layer-specific path — forward hook on ``encoder.block[layer]``.

    Returns the *pre-norm* hidden state at the chosen block. Negative
    indices are allowed (``-1`` = last block).
    """

    def __init__(self, pipeline: ChronosPipeline, layer: int):
        self.pipeline = pipeline
        n_blocks = len(pipeline.model.model.encoder.block)
        if not -n_blocks <= layer < n_blocks:
            raise IndexError(
                f"layer {layer} out of range for {n_blocks} encoder blocks"
            )
        self._block_idx = layer % n_blocks

    def encode(self, x: np.ndarray) -> np.ndarray:
        context = _to_context_tensor(x)  # (C, T)
        token_ids, attn_mask, _ = (
            self.pipeline.tokenizer.context_input_transform(context)
        )
        device = self.pipeline.model.device
        token_ids = token_ids.to(device)
        attn_mask = attn_mask.to(device)

        encoder = self.pipeline.model.model.encoder
        captured = {}

        def _hook(_module, _inputs, output):
            # Hook to capture the embeddings while performing a forward pass
            # T5Block returns a tuple; first element is the hidden state.
            hidden = output[0] if isinstance(output, tuple) else output
            captured["h"] = hidden.detach()

        handle = encoder.block[self._block_idx].register_forward_hook(_hook)
        try:
            with torch.no_grad():
                encoder(input_ids=token_ids, attention_mask=attn_mask)
        finally:
            handle.remove()

        # (C, T_tok, D) -> (T_tok, C, D)
        return captured["h"].transpose(0, 1).float().cpu().numpy()


def ChronosEncoder(
    pipeline: ChronosPipeline, layer: int | None = None
) -> UnpooledEncoder:
    """Build a Chronos feature extractor.

    Parameters
    ----------
    pipeline : ChronosPipeline
        A loaded Chronos pipeline.
    layer : int, optional
        Encoder block index to read hidden states from. ``None`` (default)
        uses :meth:`ChronosPipeline.embed`, which returns post-final-norm
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
