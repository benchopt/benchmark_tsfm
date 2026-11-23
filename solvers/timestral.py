"""Timestral solver for the TSFM benchmark (local inference).

Directly supports:
  - forecasting : zero-shot via ``TimestralForecaster.forecast``

Model loading is done in ``set_objective`` (untimed). Windows are packed
into one left-NaN-padded ``(B, V, T)`` context per call; ``hist_covars``
are forwarded as ``past_covariates``.

References
----------
    https://github.com/mistralai/timestral
"""

import numpy as np
from collections.abc import Sequence

import torch
from einops import rearrange
from timestral import TimestralForecaster

from benchmark_utils.base_solver import BaseTSFMSolver
from benchmark_utils.covariates import Covariates
from benchmark_utils.forecasting_constants import CANONICAL_QUANTILE_LEVELS


class Solver(BaseTSFMSolver):
    """Mistral Timestral zero-shot solver.

    Parameters
    ----------
    model_size : str
        Family size in millions of parameters: "5", "28", "123", "209".
    """

    name = "Timestral"

    requirements = [
        "pip::timestral @ git+https://github.com/mistralai/timestral.git",
        "pip::einops",
    ]

    # Attention scores budget (fp32): R * heads * S^2 * 4 bytes per call,
    # R = B * (V + C_past) rows.
    SDPA_SCORE_FLOATS = 750_000_000

    sampling_strategy = "run_once"

    parameters = {
        "model_size": ["209"],
        "inference_batch_size": [128],
    }

    test_config = {
        "model_size": "5",
    }

    @property
    def supported_tasks(self):
        return {"forecasting"}

    @property
    def model_id(self) -> str:
        return f"mistralai/Timestral-{self.model_size}M"

    def load_model(self, device, dtype):
        # Upstream loader validates float32 only; no dtype forwarded.
        return TimestralForecaster.from_pretrained(self.model_id, device=device)

    @property
    def quantile_levels(self) -> tuple[float, ...]:
        return CANONICAL_QUANTILE_LEVELS

    @property
    def _level_indices(self) -> tuple[int, ...]:
        """Indices of the canonical fan inside the model's native fan."""
        levels = self.model.quantile_levels
        return tuple(levels.index(q) for q in self.quantile_levels)

    def forecast_batch(
        self,
        inputs: list[torch.Tensor],
        covariates: Sequence[Covariates],
        prediction_length: int,
    ) -> list[torch.Tensor]:
        past = [
            None if cov.hist_covars is None else cov.hist_covars[0]
            for cov in covariates
        ]
        groups: dict[tuple[int, int], list[int]] = {}
        for idx, (inp, past_row) in enumerate(zip(inputs, past)):
            n_cov = 0 if past_row is None else past_row.shape[1]
            groups.setdefault((inp.shape[1], n_cov), []).append(idx)

        results: dict[int, torch.Tensor] = {}
        for idxs in groups.values():
            for chunk in self._split_rows(inputs, past, idxs):
                batch = self._forecast_group(inputs, past, prediction_length, chunk)
                for idx, pred in zip(chunk, batch, strict=True):
                    results[idx] = pred
        return [results[idx] for idx in range(len(inputs))]

    def _split_rows(
        self,
        inputs: list[torch.Tensor],
        past: Sequence[np.ndarray | None],
        idxs: list[int],
    ) -> list[list[int]]:
        """Split ``idxs`` so the packed SDPA scores stay within budget."""
        t_max = max(inputs[idx].shape[0] for idx in idxs)
        n_patched = (t_max + 31) // 32
        heads = self.model.config.num_heads
        rows_per_call = max(
            1, self.SDPA_SCORE_FLOATS // (heads * n_patched**2)
        )
        v = inputs[idxs[0]].shape[1]
        n_cov = 0 if past[idxs[0]] is None else past[idxs[0]].shape[1]
        batch_cap = max(
            1, min(self.inference_batch_size, rows_per_call // (v + n_cov))
        )
        return [
            idxs[start : start + batch_cap]
            for start in range(0, len(idxs), batch_cap)
        ]

    def _forecast_group(
        self,
        inputs: list[torch.Tensor],
        past: Sequence[np.ndarray | None],
        prediction_length: int,
        idxs: list[int],
    ) -> list[torch.Tensor]:
        """Forecast the subset ``idxs`` of a batch in a single call."""
        t_max = max(inputs[idx].shape[0] for idx in idxs)
        v = inputs[idxs[0]].shape[1]
        n_past = 0 if past[idxs[0]] is None else past[idxs[0]].shape[1]
        # NaN = an unobserved step for the model; left padding hence
        # encodes a series start moved later, never altered data.
        context = np.full((len(idxs), v, t_max), np.nan, dtype=np.float32)
        cov = (
            None
            if n_past == 0
            else np.full((len(idxs), n_past, t_max), np.nan, dtype=np.float32)
        )
        for row, idx in enumerate(idxs):
            seq = inputs[idx].numpy().T  # (T, V) -> (V, T)
            context[row, :, t_max - seq.shape[-1] :] = seq
            if n_past:
                row_cov = np.asarray(past[idx], dtype=np.float32).T
                cov[row, :, t_max - row_cov.shape[-1] :] = row_cov
        forecast = self.model.forecast(
            context,
            prediction_length=prediction_length,
            past_covariates=cov,
        )
        idx = self._level_indices
        return [
            rearrange(forecast.quantiles[row][..., idx], "v h q -> h v q")
            for row in range(len(idxs))
        ]
