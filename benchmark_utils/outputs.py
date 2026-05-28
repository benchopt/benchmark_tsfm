"""Typed outputs returned by forecasting adapters.

Forecasting predict() returns ``Sequence[ForecastOutput]`` — one entry
per input series. Each ``ForecastOutput`` carries a quantile-resolved
forecast with shape ``(n_cutoffs, Q, prediction_length, C)`` plus the
quantile levels themselves. Point forecasters set ``quantile_levels =
(0.5,)`` and Q=1; probabilistic forecasters can return as many quantiles
as their model produces.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ForecastOutput:
    """Per-series forecast.

    Attributes
    ----------
    quantiles : np.ndarray
        Shape ``(n_cutoffs, Q, prediction_length, C)``. ``quantiles[k, q]``
        is the forecast for the k-th cutoff at quantile level
        ``quantile_levels[q]``.
    quantile_levels : sequence of float
        Length ``Q``. Each entry is a quantile level in (0, 1).
    """

    quantiles: np.ndarray
    quantile_levels: Sequence[float]

    def __post_init__(self):
        if self.quantiles.ndim != 4:
            raise ValueError(
                f"quantiles must have ndim=4 (n_cutoffs, Q, prediction_length, C); "
                f"got shape {self.quantiles.shape}"
            )
        if self.quantiles.shape[1] != len(self.quantile_levels):
            raise ValueError(
                f"quantiles.shape[1] ({self.quantiles.shape[1]}) must equal "
                f"len(quantile_levels) ({len(self.quantile_levels)})"
            )

    @property
    def point(self) -> np.ndarray:
        """Best point estimate — median when available, else mean over quantiles.

        Shape: ``(n_cutoffs, prediction_length, C)``.
        """
        levels = list(self.quantile_levels)
        if 0.5 in levels:
            return self.quantiles[:, levels.index(0.5), :, :]
        return self.quantiles.mean(axis=1)
