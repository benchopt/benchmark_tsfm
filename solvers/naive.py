"""Naive baseline solver — works for all four tasks.

Forecasting      : seasonal naive (repeat last season)
Classification   : most-frequent-class in training set
Anomaly detection: constant zero scores (everything is normal)
Event detection  : predict no events (empty box array)

This solver has no model dependencies and should always pass ``benchopt test``.
It also serves as a reference for the expected solver structure.
"""

import numpy as np
from benchopt import BaseSolver

from benchmark_utils.adapters.base import BaseTSFMAdapter


# ---------------------------------------------------------------------------
# Adapter implementations
# ---------------------------------------------------------------------------

class _NaiveForecaster(BaseTSFMAdapter):
    """Repeat the last ``seasonality`` values to fill the horizon."""

    def __init__(self, prediction_length, seasonality=1):
        self.prediction_length = prediction_length
        self.seasonality = seasonality

    def predict(self, x: np.ndarray) -> np.ndarray:
        # x: (T, C)
        T, C = x.shape
        season = min(self.seasonality, T)
        pattern = x[-season:]                # (season, C)
        reps = int(np.ceil(self.prediction_length / season))
        forecast = np.tile(pattern, (reps, 1))[:self.prediction_length]
        return forecast.astype(np.float32)   # (H, C)


class _MajorityClassifier(BaseTSFMAdapter):
    """Always predict the most frequent training class."""

    def __init__(self):
        self._label = 0

    def fit(self, X_train, y_train, **kwargs):
        labels, counts = np.unique(y_train, return_counts=True)
        self._label = int(labels[np.argmax(counts)])
        return self

    def predict(self, x: np.ndarray) -> int:
        return [self._label] * len(x)


class _ConstantScorer(BaseTSFMAdapter):
    """Return zero anomaly score for every timestep."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(x.shape[0], dtype=np.float32)


class _NoEventPredictor(BaseTSFMAdapter):
    """Predict no events — returns an empty (0, 2+K) box array."""

    def __init__(self, n_classes):
        self._n_classes = n_classes

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.zeros((0, 2 + self._n_classes), dtype=np.float32)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class Solver(BaseSolver):
    """Naive baseline — no model required.

    Supports all three tasks; uses ``skip()`` only to guard against unexpected
    task names.
    """

    name = "Naive"

    # No extra requirements beyond the objective's scikit-learn + aeon.
    requirements = []

    sampling_strategy = "run_once"

    parameters = {
        "seasonality": [1],
    }

    SUPPORTED_TASKS = {"forecasting", "classification", "anomaly_detection",
                       "event_detection"}

    def skip(self, task, **kwargs):
        if task not in self.SUPPORTED_TASKS:
            return True, f"Unknown task {task!r}"
        return False, None

    def set_objective(self, X_train, y_train, task, **meta):
        self.task = task
        self.X_train = X_train
        self.y_train = y_train
        self.meta = meta

    def run(self, _):
        if self.task == "forecasting":
            pred_len = self.meta.get("prediction_length", 1)
            self._adapter = _NaiveForecaster(pred_len, self.seasonality)

        elif self.task == "classification":
            self._adapter = _MajorityClassifier()
            self._adapter.fit(self.X_train, self.y_train)

        elif self.task == "anomaly_detection":
            self._adapter = _ConstantScorer()

        elif self.task == "event_detection":
            self._adapter = _NoEventPredictor(self.meta.get("n_classes", 1))

    def get_result(self):
        return {"model": self._adapter}
