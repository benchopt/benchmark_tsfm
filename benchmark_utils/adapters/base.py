"""Base interface that all task adapters must implement.

A *fitted* adapter is what solvers return via ``get_result()``.
The objective calls ``adapter.predict(...)`` with task-appropriate inputs.

Predict signature by task
--------------------------
forecasting:

    predict(
        x: list[np.ndarray (T_i, C)],
        cutoff_indexes: list[list[int]],
        covariates: dict,
        horizon: int,
    ) -> list[np.ndarray (n_cutoffs_i, horizon, C)]

  ``cutoff_indexes[i][k]`` is the timestep index in ``x[i]`` at which
  the k-th forecast for series ``i`` starts. The model must use only
  ``x[i][:cutoff]`` as history. The ``covariates`` dict has shape
  ``{"static_covars": list, "hist_covars": list, "future_covars": list}``;
  the keys are always present (empty lists when unused).

classification:

    predict(x: np.ndarray (N, T, C)) -> np.ndarray (N,) int labels

anomaly detection:

    predict(x: np.ndarray (T, C)) -> np.ndarray (T,) float anomaly scores
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseTSFMAdapter(ABC):
    """Abstract base for fitted model + adaptation strategy.

    Subclasses must implement ``predict``.  ``fit`` is optional (used by
    supervised adaptations such as linear probe or fine-tuning).
    """

    def fit(self, X_train, y_train, **kwargs):
        """Optional supervised fitting step (called inside Solver.run())."""
        return self

    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        """Task-specific inference. See module docstring for per-task signatures."""
