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
        ``None``, or a length-``len(x)`` sequence of arrays of shape
        (time, channels). Past-only covariates, unknown over the horizon.
    future_covars
        ``None``, or a length-``len(x)`` sequence of arrays of shape
        (time, channels). Known covariates, given over the history and the
        horizon.
    hist_categorical, future_categorical
        ``None``, or a boolean array of shape (channels,), shared by all
        series: which channels of ``hist_covars`` / ``future_covars`` hold
        integer category codes.
    """

    static_covars: Sequence[np.ndarray] | None = None
    hist_covars: Sequence[np.ndarray] | None = None
    future_covars: Sequence[np.ndarray] | None = None
    hist_categorical: np.ndarray | None = None
    future_categorical: np.ndarray | None = None

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

    def select(self, series_idx: Sequence[int]) -> "Covariates":
        """Covariates of the series at ``series_idx``, in that order."""
        return Covariates(
            *(
                None if f is None else [f[i] for i in series_idx]
                for f in (self.static_covars, self.hist_covars, self.future_covars)
            ),
            self.hist_categorical,
            self.future_categorical,
        )

    def slice(self, series_idx: int, cutoff: int, horizon: int) -> "Covariates":
        """Covariates for a single ``(series, cutoff)`` window.

        Selects series ``series_idx`` and slices its time axis: the past-only
        covariates up to ``cutoff``, the known ones up to ``cutoff + horizon``.
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
            else [self.future_covars[series_idx][: cutoff + horizon]],
            hist_categorical=self.hist_categorical,
            future_categorical=self.future_categorical,
        )
