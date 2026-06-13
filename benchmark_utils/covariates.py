"""Covariates payload passed to forecasting adapters.

Each field is either ``None`` (that covariate kind is absent for every
series) or a sequence with one entry per series. Datasets without
covariates just pass ``Covariates()`` (all fields ``None``).
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Covariates:
    """Per-series covariates aligned with the ``x`` sequence in ``predict``.

    Each field is either ``None`` (absent for every series) or a sequence
    whose length equals ``len(x)`` — one entry per series. The per-series
    element shapes are listed below; see the forecasting predict() contract
    in :mod:`benchmark_utils.adapters.base`.

    Parameters
    ----------
    static_covars
        ``None``, or a length-``len(x)`` sequence of arrays of shape (channels,)
    hist_covars
        ``None``, or a length-``len(x)`` sequence of arrays of shape (time, channels)
    future_covars
        ``None``, or a length-``len(x)`` sequence of arrays of shape (time, channels)
    """

    static_covars: Sequence[np.ndarray] | None = None
    hist_covars: Sequence[np.ndarray] | None = None
    future_covars: Sequence[np.ndarray] | None = None

    def __post_init__(self):
        # Every provided (non-None) field must cover the same set of series.
        lengths = {
            len(f)
            for f in (self.static_covars, self.hist_covars, self.future_covars)
            if f is not None
        }
        if len(lengths) > 1:
            raise ValueError(
                "All provided covariate sequences must have the same length"
            )

    def __len__(self) -> int:
        """Number of series covered (0 if no covariates are present)."""
        for f in (self.static_covars, self.hist_covars, self.future_covars):
            if f is not None:
                return len(f)
        return 0

    def slice(self, series_idx: int, cutoff: int, horizon: int) -> 'Covariates':
        """Covariates for a single ``(series, cutoff)`` window.

        Selects series ``series_idx`` and slices its time axis: history up to
        ``cutoff`` and the future window ``[cutoff, cutoff + horizon)``.
        """
        return Covariates(
            static_covars=None
            if self.static_covars is None
            else [self.static_covars[series_idx]],
            hist_covars=None
            if self.hist_covars is None
            else [self.hist_covars[series_idx][:cutoff]],
            future_covars=None
            if self.future_covars is None
            else [self.future_covars[series_idx][cutoff:cutoff + horizon]],
        )
