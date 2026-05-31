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
        Static / historical / future covariates aligned with ``x``.
        Defaults to empty.
    """

    x: Sequence[np.ndarray]
    cutoff_indexes: Sequence[Sequence[int]]
    covariates: Covariates = field(default_factory=Covariates)
