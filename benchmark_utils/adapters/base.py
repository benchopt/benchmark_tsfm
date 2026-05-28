"""Base interface that all task adapters must implement.

A *fitted* adapter is what solvers return via ``get_result()``.
The objective calls ``adapter.predict(x)`` for each test sample.

Predict signature by task
--------------------------
forecasting      : x (T, C)  →  y_pred (H, C)
classification   : x (T, C)  →  int label
anomaly detection: x (T, C)  →  scores (T,) float — one score per timestep
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseTSFMAdapter(ABC):
    """Abstract base for fitted model + adaptation strategy.

    Subclasses must implement ``predict``.  ``fit`` is optional (used by
    supervised adaptations such as linear probe or fine-tuning).
    """

    def fit(self, X_train, y_train, **kwargs):
        """Optional supervised fitting step (called inside Solver.run())."""
        return self

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run inference on a single sample.

        Parameters
        ----------
        x : np.ndarray, shape (T, C)
            One time series (variable length allowed).

        Returns
        -------
        np.ndarray
            Task-specific output — see module docstring.
        """
