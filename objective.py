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

from benchmark_utils.metrics import (  # noqa: F401  (re-exported)
    ALL_METRICS,
    HIGHER_IS_BETTER,
    is_higher_better,
)


class Objective(BaseObjective):
    name = "TSFM Benchmark"
    url = "https://github.com/benchopt/benchmark_tsfm"
    min_benchopt_version = "1.9.2"

    # Shared requirements across ALL solvers — solvers declare model-specific
    # extras in their own ``requirements`` list.
    requirements = ["scikit-learn", "aeon"]

    sampling_strategy = "run_once"

    # Minimal config for ``benchopt test``
    test_config = {
        "dataset": {
            "name": [
                "monash", "ucr", "yahoo", "mitdb", "dummy-classification",
            ],
            "debug": True,
        }
    }

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def set_data(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        task,
        metrics,
        cutoff_indexes=None,
        covariates=None,
        **meta,
    ):
        from benchmark_utils.covariates import Covariates

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.cutoff_indexes = cutoff_indexes
        self.covariates = covariates if covariates is not None else Covariates()
        self.task = task
        self.metrics = metrics
        # Pull any skip marker out of meta so it doesn't leak into
        # ``get_objective()`` payloads.
        self._skip_reason = meta.pop("_skip_reason", None)
        self.meta = meta  # freq, prediction_length, n_classes, …

    def skip(self, **data):
        """Honor a ``_skip_reason`` field set by the dataset.

        Datasets that want to filter their own parameter grid (e.g.
        :mod:`datasets.gifteval` skipping non-leaderboard (path, term)
        combos) return ``_skip_reason="..."`` from ``get_data()`` and we
        propagate it here so benchopt records a clean skip rather than
        running an empty objective.
        """
        reason = data.get("_skip_reason")
        if reason:
            return True, reason
        return False, None

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
        match self.task:
            case "forecasting":
                return self._eval_forecasting(model)
            case "classification":
                return self._eval_classification(model)
            case "anomaly_detection":
                return self._eval_anomaly_detection(model)
            case "event_detection":
                return self._eval_event_detection(model)
            case _:
                raise ValueError(f"Unknown task: {self.task!r}")

    # --- forecasting ---------------------------------------------------

    def _eval_forecasting(self, model):
        from benchmark_utils.inputs import ForecastInput
        from benchmark_utils.leakage import detect_forecast_leakage

        forecast_input = ForecastInput(
            x=self.X_test,
            cutoff_indexes=self.cutoff_indexes,
            covariates=self.covariates,
        )

        # Disqualify models that peek at the future target. A leakage-free
        # forecaster's output is invariant to changes beyond each cutoff;
        # any sensitivity to the future means the reported metrics would be
        # invalid, so we surface ``leakage=1`` and set every metric to +inf
        # (the worst value, since benchopt minimises).
        report = detect_forecast_leakage(model, forecast_input)
        if report.leaked:
            return {name: float("inf") for name in self.metrics} | {
                "value": float("inf"),
                "leakage": 1.0,
            }

        forecast = model.predict(forecast_input).flatten()  # (M, H, C, Q)

        # Concatenate per-series targets into a single (M, H, C) array, in the
        # same order the flattened forecast iterates (series-major, cutoff-minor).
        y_true = np.concatenate([np.asarray(yt) for yt in self.y_test], axis=0)

        kwargs = dict(
            y_train=self.X_train,
            seasonality=self.meta.get("seasonality", 1),
            alpha=self.meta.get("mcis_alpha", 0.05),
        )
        result = {
            name: ALL_METRICS[name](y_true, forecast, **kwargs) for name in self.metrics
        }
        result["leakage"] = 0.0
        # benchopt's stopping criterion monitors a single 'value' key; expose
        # the primary requested metric under that name (mirrors the leakage
        # path above, which sets value=inf as the worst possible score).
        result["value"] = result[self.metrics[0]]
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
            def __init__(self, task, meta):
                self._task = task
                self._meta = meta

            def predict(self, x):
                if self._task == "forecasting":
                    H = self._meta.get("prediction_length", 1)
                    qs = []
                    for series, cutoffs in zip(x.x, x.cutoff_indexes):
                        C = series.shape[1] if series.ndim == 2 else 1
                        qs.append(np.zeros((len(cutoffs), H, C, 1), dtype=np.float32))
                    return ForecastOutput(quantiles=qs, quantile_levels=(0.5,))
                elif self._task == "classification":
                    return np.zeros(len(x), dtype=np.int64)
                elif self._task == "anomaly_detection":
                    return np.zeros(x.shape[0], dtype=np.float32)
                elif self._task == "event_detection":
                    return np.zeros((0, 2 + self._meta.get("n_classes", 1)))

        return {"model": _ConstantAdapter(self.task, self.meta)}
