"""Typed inputs for adapter ``predict()`` methods.

Forecasting adapters receive a :class:`ForecastInput` (one struct per
call), while classification and anomaly-detection adapters receive a
plain :class:`numpy.ndarray`. The base ``predict`` signature is a union
of the two — see :mod:`benchmark_utils.adapters.base`.
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from benchmark_utils.covariates import Covariates


@dataclass(frozen=True)
class ForecastInput:
    """Bundle of arguments passed to a forecasting adapter's predict().

    Attributes
    ----------
    x : sequence of np.ndarray
        One ``(T_i, C)`` array per series. The adapter must use only
        ``x[i][:cutoff]`` as history for the cutoff at index k.
    cutoff_indexes : sequence of sequence of int
        Jagged — per-series timestep indexes at which a forecast starts.
    covariates : Covariates
        Static / historical / future covariates aligned with ``x``. Each
        covariate field is either ``None`` (absent for every series) or has
        one entry per series (length ``len(x)``). Defaults to all-``None``.
    """

    x: Sequence[np.ndarray]
    cutoff_indexes: Sequence[Sequence[int]]
    covariates: Covariates = field(default_factory=Covariates)

    def __post_init__(self):
        if len(self.x) != len(self.cutoff_indexes):
            raise ValueError("x and cutoff_indexes must have the same length")
        # Covariates either cover all series (one entry each) or are absent.
        # len(covariates) is the shared per-field length (0 when absent), so
        # covariate-free inputs need no boilerplate.
        n_cov = len(self.covariates)
        if n_cov not in (0, len(self.x)):
            raise ValueError(
                f"covariates must cover every series: got length {n_cov}, "
                f"expected 0 (absent) or len(x)={len(self.x)}"
            )
