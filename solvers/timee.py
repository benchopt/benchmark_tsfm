"""TimEE solver for time series classification.

TimEE is a time-series foundation model for in-context classification that
processes training and test series in a single forward pass without gradient
updates.

References:
    https://arxiv.org/abs/2607.07500
    https://github.com/automl/timee
    https://huggingface.co/liamsbhoo/timee
"""

import numpy as np
from timee import TimeeClassifier, TimeeMultivariateClassifier

from benchmark_utils.adapters.base import BaseTSFMAdapter
from benchmark_utils.base_solver import BaseTSFMSolver


def _to_timee_array(X):
    """Convert ``[(T_i, C)]`` (or ``[(T_i,)]``) series to ``(N, C, T)`` float32."""
    series = [
        np.asarray(s, dtype=np.float32)[:, None]
        if np.ndim(s) == 1
        else np.asarray(s, dtype=np.float32)
        for s in X
    ]
    max_len = max(s.shape[0] for s in series)
    out = np.full((len(series), series[0].shape[1], max_len), np.nan, dtype=np.float32)
    for i, s in enumerate(series):
        out[i, :, max_len - s.shape[0] :] = s.T
    return out


class _TimeeAdapter(BaseTSFMAdapter):
    """In-context classification adapter wrapping a TimEE classifier."""

    def __init__(self, model, X_train, y_train):
        self.model = model
        self.X_train = _to_timee_array(X_train)
        self.y_train = np.asarray(y_train)

    def predict(self, x):
        X_train, X_test = self.X_train, _to_timee_array(x)
        # Without ensemble transforms, train and test are concatenated along N.
        if not getattr(self.model, "transforms", True):
            target = max(X_train.shape[-1], X_test.shape[-1])
            if X_train.shape[-1] < target:
                pad = target - X_train.shape[-1]
                X_train = np.pad(
                    X_train, ((0, 0), (0, 0), (pad, 0)), constant_values=np.nan
                )
            if X_test.shape[-1] < target:
                pad = target - X_test.shape[-1]
                X_test = np.pad(
                    X_test, ((0, 0), (0, 0), (pad, 0)), constant_values=np.nan
                )
        preds, _ = self.model.predict(X_train, self.y_train, X_test)
        return np.asarray(preds)


class Solver(BaseTSFMSolver):
    """TimEE in-context learning solver."""

    name = "TimEE"
    requirements = ["pip::timee-ts>=0.1.1"]
    supported_tasks = {"classification"}
    parameters = {"checkpoint": ["liamsbhoo/timee"], "use_ensemble": [True]}

    @property
    def model_id(self):
        return f"{self.checkpoint}:ensemble={self.use_ensemble}"

    def load_model(self, device, dtype):
        cls = (
            TimeeMultivariateClassifier
            if "multivariate" in self.checkpoint
            else TimeeClassifier
        )
        return cls.from_pretrained(
            self.checkpoint, device=device, use_ensemble=self.use_ensemble
        )

    def build_adapter(self, task, model):
        if task == "classification":
            return _TimeeAdapter(model, self.X_train, self.y_train)
        return super().build_adapter(task, model)
