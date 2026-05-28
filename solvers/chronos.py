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
import torch
from benchopt import BaseSolver
from chronos import Chronos2Pipeline, ChronosPipeline

from benchmark_utils.adapters import UnpooledEncoder
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
        # Chronos-2 returns quantile forecasts; locate the median.
        self._median_idx = list(pipeline.quantiles).index(0.5)

    def predict(self, x: np.ndarray) -> np.ndarray:

        # Chronos expects (batch, C, time) tensors.
        x = np.asarray(x, dtype=np.float32).T[None]  # (N, C, T)

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

        self.task = task
        self.X_train = X_train
        self.meta = meta

        # Load model once; reuse across consecutive dataset configs.
        # bfloat16 is fine on GPU but poorly supported on CPU — fall back
        # to float32 there so inference doesn't crash or stall.
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
        forecaster = _ChronosForecaster(self._pipeline, pred_len)

        if self.task == "forecasting":
            self._adapter = forecaster

        elif self.task == "anomaly_detection":
            self._adapter = ForecastResidualAdapter(
                forecaster, prediction_length=1
            )

    def get_result(self):
        return {"model": self._adapter}
