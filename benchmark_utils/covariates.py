"""Covariates payload passed to forecasting adapters.

A small dataclass so the contract is typed and IDE-discoverable. All
three fields default to empty sequences, so datasets without covariates
can just pass ``Covariates()``.
"""

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class Covariates:
    """Per-series covariates aligned with the ``x`` sequence in ``predict``.

    Each field is a sequence whose length equals ``len(x)``. Within a
    series, the inner structure depends on the covariate kind — see the
    forecasting predict() contract in
    :mod:`benchmark_utils.adapters.base`.
    """

    static_covars: Sequence = field(default_factory=list)
    hist_covars: Sequence = field(default_factory=list)
    future_covars: Sequence = field(default_factory=list)
