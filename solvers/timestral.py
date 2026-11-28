"""Timestral solver for the TSFM benchmark (local inference).

Directly supports:
  - forecasting : zero-shot via ``TimestralForecaster.forecast``

Model loading is done in ``set_objective`` (untimed). Windows are packed
into one left-NaN-padded ``(B, V, T)`` context per call. As fev-bench's
Timestral, ``hist_covars`` are forwarded as past covariates and
``future_covars`` as known covariates, with the categorical columns apart.

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
    # R = B * (V + C) rows, C counting every past and known covariate row.
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
        # Per window, the (rows, time) covariates of each ``forecast`` keyword,
        # as fev-bench's Timestral: past-only over the history, known over the
        # history and the horizon, category codes apart (target-encoded by the
        # model). Checkpoints that cannot route known covariates skip them.
        covs = []
        for cov in covariates:
            kinds = {}
            for name, arrays, categorical in (
                ("past", cov.hist_covars, cov.hist_categorical),
                ("known", cov.future_covars, cov.future_categorical),
            ):
                if name == "known" and not self.model.supports_known_covariates:
                    arrays = categorical = None
                rows = (
                    np.zeros((0, 0), np.float32)
                    if arrays is None
                    else np.asarray(arrays[0], dtype=np.float32).T
                )
                cat = np.zeros(len(rows), bool) if categorical is None else categorical
                kinds[f"{name}_covariates"] = rows[~cat]
                kinds[f"categorical_{name}_covariates"] = rows[cat]
            covs.append(kinds)

        groups: dict[tuple[int, ...], list[int]] = {}
        for idx, (inp, kinds) in enumerate(zip(inputs, covs)):
            key = (inp.shape[1], *(len(rows) for rows in kinds.values()))
            groups.setdefault(key, []).append(idx)

        results: dict[int, torch.Tensor] = {}
        for key, idxs in groups.items():
            for chunk in self._split_rows(inputs, sum(key), idxs):
                batch = self._forecast_group(inputs, covs, prediction_length, chunk)
                for idx, pred in zip(chunk, batch, strict=True):
                    results[idx] = pred
        return [results[idx] for idx in range(len(inputs))]

    def _split_rows(
        self,
        inputs: list[torch.Tensor],
        n_rows: int,
        idxs: list[int],
    ) -> list[list[int]]:
        """Split ``idxs`` so the packed SDPA scores stay within budget."""
        t_max = max(inputs[idx].shape[0] for idx in idxs)
        n_patched = (t_max + 31) // 32
        heads = self.model.config.num_heads
        rows_per_call = max(1, self.SDPA_SCORE_FLOATS // (heads * n_patched**2))
        batch_cap = max(1, min(self.inference_batch_size, rows_per_call // n_rows))
        return [
            idxs[start : start + batch_cap] for start in range(0, len(idxs), batch_cap)
        ]

    def _forecast_group(
        self,
        inputs: list[torch.Tensor],
        covs: Sequence[dict[str, np.ndarray]],
        prediction_length: int,
        idxs: list[int],
    ) -> list[torch.Tensor]:
        """Forecast the subset ``idxs`` of a batch in a single call."""
        t_max = max(inputs[idx].shape[0] for idx in idxs)
        # NaN = an unobserved step for the model; left padding hence
        # encodes a series start moved later, never altered data.
        context = np.full(
            (len(idxs), inputs[idxs[0]].shape[1], t_max), np.nan, dtype=np.float32
        )
        for row, idx in enumerate(idxs):
            seq = inputs[idx].numpy().T  # (T, V) -> (V, T)
            context[row, :, t_max - seq.shape[-1] :] = seq
        kwargs = {}
        for name, first in covs[idxs[0]].items():
            if not len(first):
                continue
            length = t_max + (prediction_length if "known" in name else 0)
            padded = np.full((len(idxs), len(first), length), np.nan, np.float32)
            for row, idx in enumerate(idxs):
                rows = covs[idx][name]
                padded[row, :, length - rows.shape[-1] :] = rows
            kwargs[name] = padded
        forecast = self.model.forecast(
            context, prediction_length=prediction_length, **kwargs
        )
        idx = self._level_indices
        return [
            rearrange(forecast.quantiles[row][..., idx], "v h q -> h v q")
            for row in range(len(idxs))
        ]
