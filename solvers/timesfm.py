"""TimesFM 2.5 zero-shot forecasting with independent target channels.

Weights are loaded in set_objective (untimed). Histories are truncated and
left-padded by TimesFM, which also interpolates missing values within the
available history. Empty, all-missing, and infinite-valued channels are rejected.
Covariates are not used. Classification and detection tasks are unsupported.

Reference: https://github.com/google-research/timesfm
"""

import numpy as np
import torch
from timesfm import ForecastConfig, TimesFM_2p5_200M_torch

from benchmark_utils.base_solver import BaseTSFMSolver


class Solver(BaseTSFMSolver):
    """TimesFM 2.5 with float32 inference and optional continuous quantiles.

    context_length must be a positive multiple of 32. TimesFM rounds the
    allocated horizon up to a multiple of 128; context plus prepared horizon
    cannot exceed 16384. The continuous quantile head supports at most 1024
    steps. Disable it explicitly to evaluate longer horizons with autoregressive
    quantiles instead. Normalization and quantile-crossing correction are enabled;
    automatic positivity handling and flip averaging are enabled. Context adapts
    to each batch up to context_length. Continuous quantiles use a 1024-step head
    as in the official GIFT-Eval notebook. This does not change dataset splits or
    metric formulas. Batch size can affect forecasts through adaptive context.
    """

    name = "TimesFM-2.5"
    requirements = ["pip::timesfm[torch]==2.0.2"]
    sampling_strategy = "run_once"

    parameters = {
        "checkpoint": ["google/timesfm-2.5-200m-pytorch"],
        "revision": [None],
        "context_length": [15360],
        "batch_size": [16],
        "use_continuous_quantile_head": [True],
    }

    test_config = {"context_length": 64, "batch_size": 1}

    @property
    def supported_tasks(self):
        return {"forecasting"}

    @property
    def model_id(self):
        return f"{self.checkpoint}@{self.revision}"

    def _validate_config(self, prediction_length):
        if self.context_length <= 0 or self.context_length % 32:
            raise ValueError("context_length must be a positive multiple of 32")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if prediction_length < 1:
            raise ValueError("prediction_length must be positive")
        allocated_horizon = (
            1024 if self.use_continuous_quantile_head
            else ((prediction_length + 127) // 128) * 128
        )
        if self.context_length + allocated_horizon > 16384:
            raise ValueError("context_length + rounded horizon must not exceed 16384")
        if self.use_continuous_quantile_head and prediction_length > 1024:
            raise ValueError("continuous quantile head requires a horizon <= 1024")

    def skip(self, task, **meta):
        skipped, reason = super().skip(task, **meta)
        if skipped:
            return skipped, reason
        try:
            self._validate_config(meta.get("prediction_length", 1))
        except ValueError as exc:
            return True, str(exc)
        return False, None

    def load_model(self, device, dtype):
        # This upstream API selects CUDA/CPU itself and forecasts in float32.
        model = TimesFM_2p5_200M_torch.from_pretrained(
            self.checkpoint, revision=self.revision, torch_compile=False
        )
        model.model.to(device=device, dtype=torch.float32).eval()
        return model

    def set_objective(self, X_train, y_train, task, **meta):
        horizon = meta.get("prediction_length", 1)
        self._validate_config(horizon)
        super().set_objective(X_train, y_train, task, **meta)
        self.dtype = torch.float32
        self._prepare_forecast(self.context_length, horizon)

    def _prepare_forecast(self, context, horizon):
        """Prepare inference; this is distinct from torch graph compilation."""
        self.model.compile(
            ForecastConfig(
                max_context=context,
                max_horizon=1024 if self.use_continuous_quantile_head else horizon,
                per_core_batch_size=self.batch_size,
                normalize_inputs=True,
                use_continuous_quantile_head=self.use_continuous_quantile_head,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )

    @property
    def quantile_levels(self):
        return tuple(q / 10 for q in range(1, 10))

    def forecast_batch(self, inputs, covariates, prediction_length):
        self._validate_config(prediction_length)
        channels, channel_counts = [], []
        for inp in inputs:
            x = inp.detach().float().cpu().numpy()
            if x.ndim == 1:
                x = x[:, None]
            if x.ndim != 2 or not x.shape[0] or not x.shape[1]:
                raise ValueError("TimesFM expects nonempty (time, channel) histories")
            x = x[-self.context_length :]
            if np.isinf(x).any() or np.isnan(x).all(axis=0).any():
                raise ValueError(
                    "TimesFM history contains Inf or an all-missing channel"
                )
            channel_counts.append(x.shape[1])
            channels.extend(x[:, c].copy() for c in range(x.shape[1]))

        if not channels:
            return []

        predictions = []
        for start in range(0, len(channels), self.batch_size):
            batch = channels[start : start + self.batch_size]
            n_channels = len(batch)  # Upstream may append padding inputs to batch.
            # Match the notebook's batch-dependent, patch-aligned context.
            context = ((max(len(x) for x in batch) + 31) // 32) * 32
            self._prepare_forecast(min(self.context_length, context), prediction_length)
            _, quantiles = self.model.forecast(horizon=prediction_length, inputs=batch)
            quantiles = np.asarray(quantiles)
            expected = (n_channels, prediction_length, 10)
            if quantiles.shape != expected:
                raise ValueError(
                    f"TimesFM returned {quantiles.shape}; expected {expected}"
                )
            # Component 0 is the mean; components 1:10 are q0.1 through q0.9.
            quantiles = quantiles[..., 1:]
            if not np.isfinite(quantiles).all():
                raise ValueError("TimesFM produced non-finite quantile forecasts")
            predictions.append(quantiles)

        flat = np.concatenate(predictions, axis=0)  # (total_channels, H, Q)
        results, offset = [], 0
        for count in channel_counts:
            forecast = flat[offset : offset + count].transpose(1, 0, 2)
            results.append(torch.from_numpy(forecast.copy()).float())  # (H, C, Q)
            offset += count
        return results
