"""Covariates payload passed to forecasting adapters.

A small dataclass so the contract is typed and IDE-discoverable. All
three fields default to empty sequences, so datasets without covariates
can just pass ``Covariates()``.
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Covariates:
    """Per-series covariates aligned with the ``x`` sequence in ``predict``.

    Each field is a sequence whose length equals ``len(x)``. Within a
    series, the inner structure depends on the covariate kind — see the
    forecasting predict() contract in :mod:`benchmark_utils.adapters.base`.

    Parameters
    ----------
    static_covars
        Shape is (channels,)
    hist_covars
        Shape is (time, channels)
    future_covars
        Shape is (time, channels)
    """

    static_covars: Sequence[np.ndarray] = field(default_factory=list)
    hist_covars: Sequence[np.ndarray] = field(default_factory=list)
    future_covars: Sequence[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        if len(self.static_covars) != len(self.hist_covars) != len(self.future_covars):
            raise ValueError(
                "All covariate sequences must have the same length as x"
            )
    
    def __len__(self) -> int:
        # or hist_covars or future_covars, they all have the same length
        return len(self.static_covars)

    def slice(self, cutoff: int, horizon: int) -> 'Covariates':
        """Get covariates for a single series."""
        return Covariates(
            static_covars=self.static_covars,
            hist_covars=self.hist_covars[:cutoff],
            future_covars=self.future_covars[cutoff:cutoff + horizon],
        )
