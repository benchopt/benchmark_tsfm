from .base import BaseTSFMAdapter
from .encoder import (
    BasePooler,
    Encoder,
    LastPooler,
    MaxPooler,
    MeanPooler,
    UnpooledEncoder,
)
from .forecast_residual import ForecastResidualAdapter
from .linear_probe import LinearProbeAdapter

POOLERS = {
    "mean": MeanPooler,
    "max": MaxPooler,
    "last": LastPooler,
}

__all__ = [
    "BaseTSFMAdapter",
    "UnpooledEncoder",
    "BasePooler",
    "MeanPooler",
    "MaxPooler",
    "LastPooler",
    "Encoder",
    "LinearProbeAdapter",
    "ForecastResidualAdapter",
]
