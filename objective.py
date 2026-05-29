"""
Unified objective for the TSFM benchmark.

Supports three tasks — forecasting, classification, anomaly detection —
dispatched via the ``task`` field provided by each dataset.

Data contract
-------------
All datasets must return (via ``get_data``):

    X_train : List[np.ndarray (T_i, C)]   training time series
    y_train : array-like or None           task-specific (see below)
    X_test  : List[np.ndarray (T_j, C)]   test contexts / series
    y_test  : array-like                   task-specific (see below)
    task    : str  one of {"forecasting", "classification",
                            "anomaly_detection", "event_detection"}
    metrics : List[str]  names from benchmark_utils.metrics.ALL_METRICS

Task-specific shapes
--------------------
forecasting        y_train  List[(H, C)] or None
                   y_test   List[(H, C)]
                   extra    prediction_length (int), freq (str)
classification     y_train  (N,) int
                   y_test   (M,) int
                   extra    n_classes (int)
anomaly_detection  y_train  None
                   y_test   List[(T_j,)] int  point-level binary labels
event_detection    y_train  List[(N_i, 2+K)] float  object-detection boxes
                   y_test   List[(N_j, 2+K)] float  object-detection boxes
                   extra    n_classes (int)

Solver contract
---------------
``Solver.get_result()`` must return ``{"model": adapter}`` where ``adapter``
is a fitted :class:`~benchmark_utils.adapters.base.BaseTSFMAdapter` with a
``predict(x: np.ndarray (T, C)) -> np.ndarray`` method.
"""

import numpy as np
from benchopt import BaseObjective

from benchmark_utils.metrics import ALL_METRICS


class Objective(BaseObjective):
    name = "TSFM Benchmark"
    url = "https://github.com/benchopt/benchmark_tsfm"
    min_benchopt_version = "1.9"

    # Shared requirements across ALL solvers — solvers declare model-specific
    # extras in their own ``requirements`` list.
    requirements = ["scikit-learn", "aeon"]

    sampling_strategy = "run_once"

    # Minimal config for ``benchopt test``
    test_dataset_name = "monash"
    test_config = {"dataset": {"debug": True}}

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def set_data(self, X_train, y_train, X_test, y_test,
                 task, metrics, **meta):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.task = task
        self.metrics = metrics
        self.meta = meta  # freq, prediction_length, n_classes, …

    # ------------------------------------------------------------------
    # Passed to the solver
    # ------------------------------------------------------------------

    def get_objective(self):
        return dict(
            X_train=self.X_train,
            y_train=self.y_train,
            task=self.task,
            **self.meta,
        )

    # ------------------------------------------------------------------
    # Evaluation — objective calls adapter.predict(), not the solver
    # ------------------------------------------------------------------

    def evaluate_result(self, model):
        if self.task == "forecasting":
            return self._eval_forecasting(model)
        elif self.task == "classification":
            return self._eval_classification(model)
        elif self.task == "anomaly_detection":
            return self._eval_anomaly_detection(model)
        elif self.task == "event_detection":
            return self._eval_event_detection(model)
        else:
            raise ValueError(f"Unknown task: {self.task!r}")

    # --- forecasting ---------------------------------------------------

    def _eval_forecasting(self, model):
        preds, targets = [], []
        for x, y in zip(self.X_test, self.y_test):
            pred = np.asarray(model.predict(x))  # (H, C)
            preds.append(pred)
            targets.append(np.asarray(y))

        preds = np.array(preds)    # (M, H, C)
        targets = np.array(targets)

        result = {}
        for name in self.metrics:
            fn = ALL_METRICS[name]
            if name == "mase":
                result[name] = fn(targets, preds, y_train=self.X_train,
                                  seasonality=self.meta.get("seasonality", 1))
            else:
                result[name] = fn(targets, preds)
        return result

    # --- classification ------------------------------------------------

    def _eval_classification(self, model):
        y_pred = np.asarray(model.predict(self.X_test))
        y_true = np.asarray(self.y_test)

        result = {}
        for name in self.metrics:
            result[name] = ALL_METRICS[name](y_true, y_pred)
        return result

    # --- event detection -----------------------------------------------

    def _eval_event_detection(self, model):
        # model.predict returns (N, 2+K) float array per series
        preds = [np.asarray(model.predict(x)) for x in self.X_test]

        result = {}
        for name in self.metrics:
            result[name] = ALL_METRICS[name](self.y_test, preds)
        return result

    # --- anomaly detection ---------------------------------------------

    def _eval_anomaly_detection(self, model):
        # model.predict returns (T_j,) float scores per series
        scores = [np.asarray(model.predict(x)) for x in self.X_test]

        result = {}
        for name in self.metrics:
            result[name] = ALL_METRICS[name](self.y_test, scores)
        return result

    # ------------------------------------------------------------------
    # benchopt helpers
    # ------------------------------------------------------------------

    def get_one_result(self):
        """Return a minimal valid result for benchopt's internal checks."""
        from benchmark_utils.adapters.base import BaseTSFMAdapter

        class _ConstantAdapter(BaseTSFMAdapter):
            def __init__(self, task, meta, X_test):
                self._task = task
                self._meta = meta
                self._X_test = X_test

            def predict(self, x):
                if self._task == "forecasting":
                    H = self._meta.get("prediction_length", 1)
                    C = x.shape[1] if x.ndim == 2 else 1
                    return np.zeros((H, C))
                elif self._task == "classification":
                    return 0
                elif self._task == "anomaly_detection":
                    return np.zeros(x.shape[0])
                elif self._task == "event_detection":
                    return np.zeros((0, 2 + self._meta.get("n_classes", 1)))

        return {"model": _ConstantAdapter(self.task, self.meta, self.X_test)}
