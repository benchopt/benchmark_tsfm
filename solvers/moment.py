"""Moment solver for the TSFM benchmark.

Moment is a time series foundation model from Alibaba. This solver supports:
  - forecasting     : zero-shot via Moment pipeline
  - classification  : linear probe on pooled encoder embeddings

Model loading is done in ``set_objective`` (untimed). For forecasting,
inference batches every (series, cutoff) pair into a single forward pass.
For classification, training embeddings are extracted and a linear probe
or classifier is trained on top.

References:
    https://huggingface.co/AutonLab/MOMENT-1-large
"""

import numpy as np
import torch
from benchopt import BaseSolver

from benchmark_utils.adapters import (
    Encoder,
    LastPooler,
    LinearProbeAdapter,
    MaxPooler,
    MeanPooler,
    UnpooledEncoder,
)
from benchmark_utils.adapters.base import BaseTSFMAdapter
from benchmark_utils.inputs import ForecastInput
from benchmark_utils.outputs import ForecastOutput

try:
    from momentfm import MOMENTPipeline

    HAS_MOMENT = True
except ImportError:
    HAS_MOMENT = False

SUPPORTED_TASKS = {"forecasting", "classification"}

POOLERS = {
    "mean": MeanPooler,
    "max": MaxPooler,
    "last": LastPooler,
}


class _MomentForecaster(BaseTSFMAdapter):
    """Moment forecasting adapter."""

    def __init__(self, pipeline, prediction_length):
        self.pipeline = pipeline
        self.prediction_length = prediction_length

    def predict(self, x: ForecastInput) -> ForecastOutput:
        quantiles = []
        for series, cutoffs in zip(x.x, x.cutoff_indexes):
            series = np.asarray(series, dtype=np.float32)
            if series.ndim == 1:
                series = series[:, None]
            T, C = series.shape

            preds_per_series = []
            for cutoff in cutoffs:
                hist = series[:cutoff]  # (T_cutoff, C)

                if hist.ndim == 1:
                    hist = hist[None, :]

                # Moment expects (B, channels, seq_len)
                hist_tensor = (
                    torch.from_numpy(hist.transpose(1, 0)).unsqueeze(0).float()
                )
                device = next(self.pipeline.parameters()).device
                hist_tensor = hist_tensor.to(device)
                input_mask = torch.ones(
                    (hist_tensor.shape[0], hist_tensor.shape[2]),
                    dtype=torch.float32,
                    device=device,
                )

                with torch.no_grad():
                    outputs = self.pipeline.forecast(
                        x_enc=hist_tensor,
                        input_mask=input_mask,
                        prediction_length=self.prediction_length,
                    )

                forecast = outputs.forecast if hasattr(outputs, "forecast") else outputs
                if isinstance(forecast, tuple):
                    forecast = forecast[0]

                arr = forecast.squeeze(0).cpu().numpy()

                if arr.ndim == 1:
                    arr = arr[:, None]

                # Moment returns (channels, horizon) by default.
                if arr.ndim == 2 and arr.shape[0] != self.prediction_length:
                    arr = arr.T

                if arr.shape[0] > self.prediction_length:
                    arr = arr[: self.prediction_length]

                if arr.shape[0] != self.prediction_length:
                    raise ValueError(
                        f"Unexpected forecast shape after transpose/slice: {arr.shape}"
                    )

                preds_per_series.append(arr)

            # Stack predictions: (n_cutoffs, prediction_length, C)
            stacked = np.stack(preds_per_series, axis=0)
            # Add quantile dimension: (n_cutoffs, 1, prediction_length, C)
            quantiles.append(stacked[:, None, :, :])

        return ForecastOutput(quantiles=quantiles, quantile_levels=(0.5,))


class _MomentEncoder(UnpooledEncoder):
    """Moment encoder for extracting embeddings."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def encode(self, X) -> np.ndarray:
        """Extract embeddings from time series data.

        Args:
            X: np.ndarray of shape (T, C) or (B, T, C)

        Returns:
            np.ndarray of shape (B, T, V, D)
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[None]
        elif X.ndim != 3:
            raise ValueError(f"Unexpected input shape for Moment encoder: {X.shape}")

        # Moment expects (B, channels, seq_len)
        X = X.transpose(0, 2, 1)

        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float()
            device = next(self.pipeline.parameters()).device
            X_tensor = X_tensor.to(device)
            outputs = self.pipeline.embed(x_enc=X_tensor, reduction="none")
            emb = outputs.embeddings

        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()

        if emb.ndim != 4:
            raise ValueError(f"Unexpected Moment embedding shape: {emb.shape}")

        # Moment returns (B, channels, n_patches, D); transform to
        # (B, n_patches, channels, D) for the benchmark encoder API.
        return emb.transpose(0, 2, 1, 3)


class Solver(BaseSolver):
    """Moment foundation model solver.

    Supports forecasting (zero-shot) and classification (with linear probe).
    The model is loaded once in ``set_objective`` (not timed).
    """

    name = "Moment"

    # moment-fm package required for the model
    requirements = [
        "pip::moment @ git+https://github.com/moment-timeseries-foundation-model/moment.git",
    ]

    sampling_strategy = "run_once"

    parameters = {
        "checkpoint": ["AutonLab/MOMENT-1-large"],
        "task_config": ["forecasting"],  # forecasting or classification
        "pooler": ["mean"],  # pooler for classification embeddings
        "batch_size": [32],
        "classifier": ["log_reg"],
        "penalty": ["l2"],
        "C": [1.0],
        "alpha": [1.0],
        "n_iterators": [100],
    }

    def skip(self, task, **kwargs):
        if task not in SUPPORTED_TASKS:
            return True, f"Moment solver does not support task={task!r}"
        if not HAS_MOMENT:
            return True, "momentfm package not installed"
        return False, None

    def set_objective(self, X_train, y_train, task, **meta):
        """Prepare the solver for a given dataset configuration.

        Model loading is done here (not inside ``run``) so that the
        checkpoint download/loading time is excluded from the benchmark
        timing.
        """
        self.task = task
        self.X_train = X_train
        self.y_train = y_train
        self.meta = meta
        self.prediction_length = meta.get("prediction_length")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the model only on the first call for this checkpoint
        should_reload = (
            not hasattr(self, "_pipeline")
            or not hasattr(self, "_loaded_checkpoint")
            or self._loaded_checkpoint != self.checkpoint
        )
        if should_reload:
            try:
                self._pipeline = MOMENTPipeline.from_pretrained(
                    self.checkpoint,
                    torch_dtype=torch.float32,
                )
                self._pipeline = self._pipeline.to(device)
                self._loaded_checkpoint = self.checkpoint
                print(
                    f"✓ Moment checkpoint loaded: {self.checkpoint} on device: {device}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Moment checkpoint '{self.checkpoint}' "
                    f"from Hugging Face: {e}. Make sure you have internet "
                    "access and the model is available."
                )

        self._device = device

    def run(self, _):
        """Fit the model or adapter on the training data."""
        if self.task == "forecasting":
            self._adapter = _MomentForecaster(
                pipeline=self._pipeline,
                prediction_length=self.prediction_length,
            )
        elif self.task == "classification":
            base_encoder = _MomentEncoder(pipeline=self._pipeline)
            encoder = Encoder(base_encoder, POOLERS[self.pooler]())

            self._adapter = LinearProbeAdapter(
                encoder=encoder,
                task="classification",
                n_classes=self.meta.get("n_classes"),
                classifier=self.classifier,
                max_iter=self.max_iter,
                n_estimators=self.n_estimators,
            )
            self._adapter.fit(self.X_train, self.y_train)
        else:
            raise ValueError(f"Unsupported task: {self.task}")

    def get_result(self):
        return {"model": self._adapter}
