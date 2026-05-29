"""Base interface that all task adapters must implement.

A *fitted* adapter is what solvers return via ``get_result()``.
The objective calls ``adapter.predict(x)`` with task-appropriate inputs.

Predict signature by task
--------------------------
forecasting:

    predict(x: ForecastInput) -> ForecastOutput

  :class:`~benchmark_utils.inputs.ForecastInput` bundles the per-series
  history list, the jagged per-series cutoff indexes, and a
  :class:`~benchmark_utils.covariates.Covariates` dataclass.

  :class:`~benchmark_utils.outputs.ForecastOutput` is a single object
  covering every input series — its ``quantiles`` field is a Sequence
  of ``(n_cutoffs_i, Q, prediction_length, C)`` arrays, one per series,
  with a shared ``quantile_levels`` tuple. Point forecasters set
  ``quantile_levels=(0.5,)`` and Q=1.

  ``prediction_length`` is dataset-level — the solver reads it from the
  objective and wires it into the adapter at construction time.

classification:

    predict(x: np.ndarray (N, T, C)) -> np.ndarray (N,) int labels

anomaly detection:

    predict(x: np.ndarray (T, C)) -> np.ndarray (T,) float anomaly scores
"""

from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np

from benchmark_utils.inputs import ForecastInput


PredictInput = Union[ForecastInput, np.ndarray]


class BaseTSFMAdapter(ABC):
    """Abstract base for fitted model + adaptation strategy.

    Subclasses must implement ``predict``.  ``fit`` is optional (used by
    supervised adaptations such as linear probe or fine-tuning).
    """

    def fit(self, X_train, y_train, **kwargs):
        """Optional supervised fitting step (called inside Solver.run())."""
        return self

    @abstractmethod
    def predict(self, x: PredictInput) -> Any:
        """Task-specific inference. See module docstring for per-task signatures."""
