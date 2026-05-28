"""Typed output returned by forecasting adapters.

Forecasting predict() returns a single :class:`ForecastOutput` covering
every input series in the matching :class:`ForecastInput`. The output is
shape-aware: ``quantiles[i]`` is the per-series ndarray
``(n_cutoffs_i, Q, prediction_length, C)``, aligned with the same index
order as the input ``x``.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ForecastOutput:
    """Quantile-resolved forecast for a batch of series.

    Attributes
    ----------
    quantiles : sequence of np.ndarray
        One ndarray per series, each shape
        ``(n_cutoffs_i, Q, prediction_length, C)``. ``quantiles[i][k, q]``
        is the forecast for series ``i``, cutoff ``k``, at quantile level
        ``quantile_levels[q]``.
    quantile_levels : sequence of float
        Length ``Q``. Each entry is a quantile level in (0, 1). The same
        ``Q`` applies to every series in the batch.
    """

    quantiles: Sequence[np.ndarray]
    quantile_levels: Sequence[float]

    def __post_init__(self):
        Q = len(self.quantile_levels)
        for i, arr in enumerate(self.quantiles):
            if arr.ndim != 4:
                raise ValueError(
                    f"quantiles[{i}] must have ndim=4 "
                    f"(n_cutoffs, Q, prediction_length, C); got shape {arr.shape}"
                )
            if arr.shape[1] != Q:
                raise ValueError(
                    f"quantiles[{i}].shape[1] ({arr.shape[1]}) must equal "
                    f"len(quantile_levels) ({Q})"
                )

    @property
    def point(self) -> Sequence[np.ndarray]:
        """Best point estimate per series — median when available, else mean across quantiles.

        Each entry has shape ``(n_cutoffs_i, prediction_length, C)``.
        """
        levels = list(self.quantile_levels)
        if 0.5 in levels:
            idx = levels.index(0.5)
            return [arr[:, idx, :, :] for arr in self.quantiles]
        return [arr.mean(axis=1) for arr in self.quantiles]
