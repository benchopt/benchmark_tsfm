"""Chronos solver for the TSFM benchmark (local inference).

Supports:
  - forecasting     : zero-shot via ChronosPipeline
  - classification  : linear probe on pooled encoder embeddings
  - anomaly_detection  : forecast-residual on top of the same forecaster
  - event_detection : EventHead trained on frozen encoder embeddings

Model loading is done in ``set_objective`` (untimed). Inference batches
every (series, cutoff) pair into a single ``Chronos2Pipeline.predict``
call — the pipeline accepts a list of variable-length tensors and
applies left-padding internally, so all the per-cutoff work happens in
one forward pass.
"""

import numpy as np
import torch
from chronos import ChronosPipeline

from benchmark_utils.adapters import (
    Encoder,
    LastPooler,
    LinearProbeAdapter,
    MaxPooler,
    MeanPooler,
    UnpooledEncoder,
)
from benchmark_utils.adapters.base import BaseTSFMAdapter
from benchmark_utils.adapters.event_detection import (
    ChronosEventAdapter,
    fit_event_head,
    precompute_embeddings,
)
from benchmark_utils.adapters.forecast_residual import ForecastResidualAdapter
from benchmark_utils.base_solver import BaseTSFMSolver
from benchmark_utils.inputs import ForecastInput
from benchmark_utils.outputs import ForecastOutput

SUPPORTED_TASKS = {"forecasting", "classification", "anomaly_detection", "event_detection"}

# Chronos encoder output dimension by model size
_CHRONOS_D = {"tiny": 64, "mini": 128, "small": 512, "base": 768, "large": 1024}

POOLERS = {
    "mean": MeanPooler,
    "max": MaxPooler,
    "last": LastPooler,
}


def _to_context(x):
    """Reshape ``(T, V)`` or ``(B, T, V)`` to Chronos input ``(B, V, T)``."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[None]
    return x.transpose(0, 2, 1)


class _ChronosEmbedEncoder(UnpooledEncoder):
    """Default path — uses ``Chronos2Pipeline.embed``.

    Returns hidden states *after* ``encoder.final_layer_norm`` for each
    series in the batch.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def encode(self, X) -> np.ndarray:
        context = _to_context(X)  # (B, V, T)
        with torch.no_grad():
            # embed returns a list of B tensors, each of shape (V, T, D).
            embeddings, _ = self.pipeline.embed(context)
        stacked = torch.stack(list(embeddings))  # (B, V, T, D)
        return stacked.transpose(1, 2).float().cpu().numpy()  # (B, T, V, D)


class _ChronosHookEncoder(UnpooledEncoder):
    """Layer-specific path — forward hook on ``encoder.block[layer]``.

    Returns the *pre-norm* hidden state at the chosen block. Negative
    indices are allowed (``-1`` = last block).
    """

    def __init__(self, pipeline, layer: int):
        self.pipeline = pipeline
        n_blocks = len(pipeline.model.model.encoder.block)
        if not -n_blocks <= layer < n_blocks:
            raise IndexError(
                f"layer {layer} out of range for {n_blocks} encoder blocks"
            )
        self._block_idx = layer % n_blocks

    def encode(self, x: np.ndarray) -> np.ndarray:
        context = _to_context(x)  # (B, V, T)
        token_ids, attn_mask, _ = self.pipeline.tokenizer.context_input_transform(
            context
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


def ChronosEncoder(pipeline, layer: int | None = None) -> UnpooledEncoder:
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


class Solver(BaseTSFMSolver):
    """Chronos-2 zero-shot solver.

    Parameters
    ----------
    model_size : str
        Chronos model variant: "tiny", "mini", "small", "base", "large".
    layer : int or None
        Encoder block index for classification embeddings. ``None`` uses
        ``ChronosPipeline.embed`` (post-final-norm).
    pooler : {"mean", "max", "last"}
        Pooling strategy over the time-token axis for classification.
    model_path : str
        Local directory path to load the Chronos model from. When empty
        (default), the model is loaded from HuggingFace Hub.
    """

    name = "Chronos"

    requirements = ["pip::chronos-forecasting>=2.2,<3"]

    parameters = {
        "model_size": ["small"],
        "layer": [None],
        "pooler": ["mean"],
        # event_detection — single values so no cross-product for other tasks
        "model_path": [""],
        "batch_size": [32],
        "num_epochs": [100],
        "lr": [3e-4],
        "weight_decay": [1e-4],
        "warmup_epochs": [5],
        "num_dec_layers": [2],
        "lambda_cls": [1.0],
        "num_queries": [10],
    }

    def __init__(
        self,
        model_size="small",
        layer=None,
        pooler="mean",
        model_path="",
        batch_size=32,
        num_epochs=100,
        lr=3e-4,
        weight_decay=1e-4,
        warmup_epochs=5,
        num_dec_layers=2,
        lambda_cls=1.0,
        num_queries=10,
    ):
        """Initialize Chronos-specific state.

        Parameters
        ----------
        model_size : str, default="small"
            Chronos model variant to load.
        layer : int or None, default=None
            Encoder block index for classification embeddings.
        pooler : {"mean", "max", "last"}, default="mean"
            Pooling strategy over the time-token axis for classification.
        model_path : str, default=""
            Local model directory; empty = load from HuggingFace Hub.
        """
        super().__init__(
            model_size=model_size,
            layer=layer,
            pooler=pooler,
            model_path=model_path,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            num_dec_layers=num_dec_layers,
            lambda_cls=lambda_cls,
            num_queries=num_queries,
        )
        self._pipeline = None
        self._loaded_model = None

    @property
    def supported_tasks(self):
        return SUPPORTED_TASKS

    def load_model(self, device, dtype):
        """Load Chronos-2 pipeline (cached if already loaded)."""
        from chronos import Chronos2Pipeline

        model_id = f"autogluon/chronos-2-{self.model_size}"
        model_id = self.model_path if self.model_path else model_id
        if not hasattr(self, "_pipeline") or self._loaded_model != model_id:
            self._pipeline = Chronos2Pipeline.from_pretrained(
                model_id,
                device_map=device,
                dtype=dtype,
            )
            self._loaded_model = model_id
        return self._pipeline

    def set_objective(self, X_train, y_train, task, **meta):
        """Load pipeline then pre-compute embeddings for event_detection."""
        super().set_objective(X_train, y_train, task, **meta)

        if task == "event_detection":
            self._n_classes = int(meta["n_classes"])
            self._d_model = _CHRONOS_D.get(self.model_size, 512)
            self._Z_train = precompute_embeddings(self.model, X_train)

    def forecast_batch(self, inputs):
        """Chronos-specific batch prediction.

        Parameters
        ----------
        inputs : list of torch.Tensor
            Each tensor shape (C, T_cutoff)

        Returns
        -------
        list of torch.Tensor
            Each tensor shape (C, Q, H)
        """
        with torch.no_grad():
            return self.model.predict(inputs, prediction_length=self.prediction_length)

    def build_adapter(self, task, model):
        # TODO later: put that code in base_solver.py
        # and make it rely on .forecast(), .embed() and .time_embed() only, once those are all properly coded
        """Create task-specific adapter for Chronos."""
        pred_len = self.meta.get("prediction_length", 1)

        if task == "forecasting":
            self.prediction_length = pred_len
            quantile_levels = tuple(float(q) for q in model.quantiles)

            # Create a simple adapter that calls self.forecast()
            class _ForecastAdapter(BaseTSFMAdapter):
                def __init__(self, solver, quantile_levels):
                    self.solver = solver
                    self.quantile_levels = quantile_levels

                def predict(self, x: ForecastInput) -> ForecastOutput:
                    return self.solver.forecast(x, self.solver.prediction_length, self.quantile_levels)

            return _ForecastAdapter(self, quantile_levels)

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
            # AD uses one-step-ahead forecasts.
            self.prediction_length = 1
            quantile_levels = (0.5,)

            # Create a forecaster adapter for residual-based anomaly detection
            class _ForecasterForAD(BaseTSFMAdapter):
                def __init__(self, solver, quantile_levels):
                    self.solver = solver
                    self.quantile_levels = quantile_levels

                def predict(self, x: ForecastInput) -> ForecastOutput:
                    return self.solver.forecast(x, 1, self.quantile_levels)

            forecaster = _ForecasterForAD(self, quantile_levels)
            return ForecastResidualAdapter(forecaster, prediction_length=1)

        elif task == "event_detection":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            head = fit_event_head(
                self._Z_train, self.y_train, self._n_classes, self._d_model,
                device, self.batch_size, self.num_epochs, self.lr,
                self.weight_decay, self.warmup_epochs, self.num_dec_layers,
                self.lambda_cls, self.num_queries,
            )
            return ChronosEventAdapter(model, head, device, self._n_classes)

        raise ValueError(f"Unknown task: {task}")
