"""T0 solver for the TSFM benchmark (local inference).

Runs The Forecasting Company's open-weights T0 model
(https://huggingface.co/theforecastingcompany/t0-alpha) for zero-shot
forecasting via the ``tfc-t0`` package.

Model loading is done in ``set_objective`` (untimed). Inference batches
every (series, cutoff) pair into ``T0Forecaster.predict`` calls: the
model treats NaN as missing, so variable-length histories are
left-padded with NaN into a single ``(B, C, T_max)`` tensor. Pairs are
grouped by channel count so each group forms a rectangular batch, and
each group is chunked by ``batch_size`` to bound memory.

References
----------
    https://huggingface.co/theforecastingcompany/t0-alpha
"""

import numpy as np
import torch
from benchopt import BaseSolver
from t0 import T0Forecaster

from benchmark_utils.adapters.base import BaseTSFMAdapter
from benchmark_utils.inputs import ForecastInput
from benchmark_utils.outputs import ForecastOutput

SUPPORTED_TASKS = {"forecasting"}

QUANTILE_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)


class _T0Forecaster(BaseTSFMAdapter):
    """T0 forecaster — native quantile output, NaN-padded batching."""

    def __init__(self, model, prediction_length, batch_size):
        self.model = model
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.quantile_levels = QUANTILE_LEVELS

    def predict(self, x: ForecastInput, prediction_length=None) -> ForecastOutput:
        horizon = prediction_length or self.prediction_length
        groups, per_series_shape = self._build_groups(x)
        if not groups:
            return ForecastOutput(quantiles=[], quantile_levels=self.quantile_levels)

        Q = len(self.quantile_levels)
        per_series = [
            np.empty((n_cutoffs, horizon, C, Q), dtype=np.float32)
            for C, n_cutoffs in per_series_shape
        ]
        for histories, layout in groups.values():
            preds = self._predict_group(histories, horizon)  # (B, C, H, Q)
            for (series_idx, cutoff_idx), pred in zip(layout, preds):
                # (C, H, Q) -> (H, C, Q)
                per_series[series_idx][cutoff_idx] = pred.transpose(1, 0, 2)
        return ForecastOutput(
            quantiles=per_series, quantile_levels=self.quantile_levels
        )

    def _build_groups(self, x):
        """Group (series, cutoff) histories by channel count.

        Returns ``{C: (histories, layout)}`` where ``histories`` is a list
        of ``(C, T_cutoff)`` arrays and ``layout`` the matching
        ``(series_idx, cutoff_idx)`` pairs, plus per-series ``(C, n_cutoffs)``.
        """
        groups = {}
        per_series_shape = []
        for series_idx, (series, cutoffs) in enumerate(zip(x.x, x.cutoff_indexes)):
            series = np.asarray(series, dtype=np.float32)
            if series.ndim == 1:
                series = series[:, None]
            _, C = series.shape
            per_series_shape.append((C, len(cutoffs)))
            histories, layout = groups.setdefault(C, ([], []))
            for cutoff_idx, cutoff in enumerate(cutoffs):
                histories.append(series[:cutoff].T)  # (C, T_cutoff)
                layout.append((series_idx, cutoff_idx))
        return groups, per_series_shape

    def _predict_group(self, histories, horizon):
        """Left-pad same-C histories with NaN and predict in chunks."""
        outputs = []
        for start in range(0, len(histories), self.batch_size):
            chunk = histories[start : start + self.batch_size]
            T_max = max(h.shape[1] for h in chunk)
            C = chunk[0].shape[0]
            context = np.full((len(chunk), C, T_max), np.nan, dtype=np.float32)
            for i, hist in enumerate(chunk):
                context[i, :, T_max - hist.shape[1] :] = hist
            with torch.no_grad():
                out = self.model.predict(
                    context,
                    horizon=horizon,
                    quantiles=self.quantile_levels,
                )
            outputs.append(out.quantiles.float().cpu().numpy())  # (b, C, H, Q)
        return np.concatenate(outputs, axis=0)


class Solver(BaseSolver):
    """T0 zero-shot forecasting solver.

    Parameters
    ----------
    model_id : str
        Hugging Face model id of the T0 checkpoint.
    batch_size : int
        Maximum number of (series, cutoff) windows per forward pass.
    """

    name = "T0"

    requirements = ["pip::tfc-t0"]

    parameters = {
        "model_id": ["theforecastingcompany/t0-alpha"],
        "batch_size": [256],
    }

    def skip(self, task, **kwargs):
        if task not in SUPPORTED_TASKS:
            return True, f"T0 solver does not support task={task!r}"
        return False, None

    def set_objective(self, X_train, y_train, task, **meta):
        self.task = task
        self.X_train = X_train
        self.y_train = y_train
        self.meta = meta

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not hasattr(self, "_model") or self._loaded_model != self.model_id:
            self._model = (
                T0Forecaster.from_pretrained(self.model_id).to(device).eval()
            )
            self._loaded_model = self.model_id

    def run(self, _):
        self._adapter = _T0Forecaster(
            self._model,
            prediction_length=self.meta.get("prediction_length", 1),
            batch_size=self.batch_size,
        )

    def get_result(self):
        return {"model": self._adapter}
