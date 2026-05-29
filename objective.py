"""
Unified objective for the TSFM benchmark.

Supports three tasks — forecasting, classification, anomaly detection —
dispatched via the ``task`` field provided by each dataset.

Data contract
-------------
All datasets must return (via ``get_data``):

    X_train : List[np.ndarray (T_i, C)]   training time series
    y_train : array-like or None          task-specific (see below)
    X_test  : List[np.ndarray]            test data (shape depends on task)
    y_test  : array-like                  task-specific (see below)
    task    : str  one of {"forecasting", "classification",
                            "anomaly_detection", "event_detection"}
    metrics : List[str]  names from benchmark_utils.metrics.ALL_METRICS

Task-specific shapes
--------------------
forecasting        X_test         List[(T_i, C)]  full series — adapter uses
                                                  ``x[:cutoff]`` as history
                   cutoff_indexes List[List[int]] jagged per-series cutoffs
                   y_test         List[(n_cutoffs, H, C)]
                   covariates     Covariates      dataclass with
                                                  static / hist / future
                                                  covariate lists
                   extra          prediction_length (int), freq (str) —
                                                  the solver reads these
                                                  from the objective once
                                                  and wires them into the
                                                  adapter
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
is a fitted :class:`~benchmark_utils.adapters.base.BaseTSFMAdapter`.
See that module for per-task predict signatures.
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
                 task, metrics, cutoff_indexes=None, covariates=None,
                 **meta):
        from benchmark_utils.covariates import Covariates

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.cutoff_indexes = cutoff_indexes
        self.covariates = covariates if covariates is not None else Covariates()
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
        from benchmark_utils.inputs import ForecastInput

        output = model.predict(
            ForecastInput(
                x=self.X_test,
                cutoff_indexes=self.cutoff_indexes,
                covariates=self.covariates,
            )
        )

        preds, targets = [], []
        for series_point, series_targets in zip(output.point, self.y_test):
            sp = np.asarray(series_point)  # (n_cutoffs, H, C)
            st = np.asarray(series_targets)
            for k in range(sp.shape[0]):
                preds.append(sp[k])
                targets.append(st[k])

        preds = np.array(preds)
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
        from benchmark_utils.outputs import ForecastOutput

        class _ConstantAdapter(BaseTSFMAdapter):
            def __init__(self, task, prediction_length):
                self._task = task
                self._prediction_length = prediction_length

            def predict(self, x):
                if self._task == "forecasting":
                    H = self._prediction_length
                    qs = []
                    for series, cutoffs in zip(x.x, x.cutoff_indexes):
                        C = series.shape[1] if series.ndim == 2 else 1
                        qs.append(np.zeros((len(cutoffs), 1, H, C), dtype=np.float32))
                    return ForecastOutput(quantiles=qs, quantile_levels=(0.5,))
                elif self._task == "classification":
                    return np.zeros(len(x), dtype=np.int64)
                elif self._task == "anomaly_detection":
                    return np.zeros(x.shape[0], dtype=np.float32)
                elif self._task == "event_detection":
                    return np.zeros((0, 2 + self._meta.get("n_classes", 1)))

        return {"model": _ConstantAdapter(
            self.task, self.meta.get("prediction_length", 1)
        )}
