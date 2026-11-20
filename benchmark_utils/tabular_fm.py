"""Shared solver for tabular foundation models (TabICL, TabPFN, ...).

These models predict by in-context learning on a table of rows.

- **Classification** flattens each ``(T, C)`` sample into a ``(T*C,)`` row and
  fits the model's classifier on the training rows.
- **Forecasting** recasts each series as tabular regression on
  timestamp-derived features (index / calendar / auto-seasonal),
  reusing tabicl's forecasting pipeline (``tabicl.forecast``) with the model's
  own regressor plugged in through :class:`_RegressorEngine`.

A new model subclasses :class:`TabularFMSolver`. ``doc/tabular_fm.rst``
explains how its two hooks serve each task.
"""

from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from tabicl.forecast import TabICLForecaster
from tabicl.forecast._engine import ForecastEngine

from benchmark_utils.base_solver import BaseTSFMSolver
from benchmark_utils.forecasting_constants import (
    CANONICAL_QUANTILE_LEVELS,
    safe_freq,
)

# Anchor for synthesized timestamps, in ms so long yearly windows stay in range.
SYNTH_TS_START = "2000-01-01"


def to_tabular(X):
    """Flatten a list of ``(T, C)`` samples into an ``(N, T*C)`` matrix."""
    arr = np.asarray(X, dtype=np.float32)  # (N, T) or (N, T, C)
    if arr.ndim == 3:
        arr = arr.reshape(arr.shape[0], -1)  # flatten C into features
    return arr


class _RegressorEngine(ForecastEngine):
    """tabicl ForecastEngine that fits a given regressor on each series.

    Only the per-series inference routine changes. tabicl's SeriesDispatcher
    runs it on each series in loky processes (up to 8 on CPU, one per GPU
    with several GPUs) or in the main process on a single GPU. The engine is
    pickled to those processes, so the regressor must be unfitted.
    """

    def __init__(self, regressor, predict_quantiles):
        super().__init__()
        self._regressor = regressor
        self._predict_quantiles = predict_quantiles

    def _inference_routine(self, train_X, train_y, test_X, quantiles=None):
        # Fit a clone so the template never carries fitted weights.
        model = clone(self._regressor).fit(train_X, train_y)
        q = self._predict_quantiles(model, test_X, quantiles)  # (n_test, Q)
        # tabicl requires a point column; the benchmark only reads quantiles.
        result = {"target": q.mean(axis=1)}
        result.update({level: q[:, i] for i, level in enumerate(quantiles)})
        return result


def forecast_windows(forecaster, inputs, prediction_length, quantile_levels, freq):
    """Forecast a batch of history windows with a tabicl-pipeline forecaster.

    Parameters
    ----------
    forecaster : TabICLForecaster
        Built by :meth:`TabularFMSolver.load_model`.
    inputs : list of array-like, shape (lookback_i, C)
        History windows already sliced at each cutoff by ``BaseTSFMSolver``.
    prediction_length : int
    quantile_levels : sequence of float
    freq : str
        Frequency alias for the synthesized timestamps.

    Returns
    -------
    list of np.ndarray, shape ``(prediction_length, C, Q)``
        One forecast per input window, aligned with ``inputs``.
    """
    qs = list(quantile_levels)
    freq = safe_freq(freq)

    context, future = [], []
    shapes = []
    for j, window in enumerate(inputs):
        window = np.asarray(window, dtype=np.float32)
        if window.ndim == 1:
            window = window[:, None]
        # tabicl only uses the last max_context_length steps.
        window = window[-forecaster.max_context_length :]
        T, C = window.shape
        shapes.append(C)
        # Explicit future timestamps: tabicl would otherwise infer the freq,
        # which fails on histories shorter than 3 steps.
        ts = pd.date_range(
            SYNTH_TS_START, periods=T + prediction_length, freq=freq, unit="ms"
        )
        for c in range(C):
            item_id = f"{j}:{c}"
            context.append(
                pd.DataFrame(
                    {"item_id": item_id, "timestamp": ts[:T], "target": window[:, c]}
                )
            )
            future.append(pd.DataFrame({"item_id": item_id, "timestamp": ts[T:]}))

    if not context:
        return []

    pred = forecaster.predict_df(
        pd.concat(context, ignore_index=True),
        future_df=pd.concat(future, ignore_index=True),
        quantiles=qs,
    )

    # Rows in input order (tabicl sorts item ids), prediction_length per item.
    ids = [f"{j}:{c}" for j, C in enumerate(shapes) for c in range(C)]
    q = pred.loc[ids, qs].to_numpy(np.float32)
    q = q.reshape(len(ids), prediction_length, len(qs)).transpose(1, 0, 2)
    return np.split(q, np.cumsum(shapes)[:-1], axis=1)


class TabularFMSolver(BaseTSFMSolver):
    """Base solver for tabular foundation models.

    A subclass, in its own ``solvers/`` file, sets:

    - ``name``, ``requirements`` and ``parameters`` as literals: benchopt reads
      them statically before the dependencies are installed, so they are not
      inherited. ``requirements`` must include ``pip::tabicl[forecast]`` and
      ``parameters`` must define ``max_context_length``.
    - ``make_classifier(device)``: an unfitted scikit-learn classifier.
    - ``make_regressor()``: an unfitted scikit-learn regressor.
    - ``predict_quantiles(model, X, levels)``: a staticmethod returning the
      ``(n, Q)`` quantiles predicted by the fitted regressor.

    A model without native quantiles sets ``supported_tasks`` to
    classification only.
    """

    supported_tasks = {"classification", "forecasting"}
    quantile_levels = CANONICAL_QUANTILE_LEVELS

    @property
    def model_id(self):
        # Parameters are fixed per solver instance, only the task changes.
        return self.task

    @abstractmethod
    def make_classifier(self, device):
        """Return an unfitted classifier placed on ``device``."""

    @abstractmethod
    def make_regressor(self):
        """Return an unfitted regressor; tabicl assigns it a device per worker."""

    @staticmethod
    @abstractmethod
    def predict_quantiles(model, X, levels):
        """Return the ``(n, Q)`` quantiles of ``X`` at ``levels``."""

    def load_model(self, device, dtype):
        # Weights load lazily in fit(): fit a copy on a toy table so the
        # checkpoint download and load are not timed in run().
        X_toy, y_toy = np.random.default_rng(0).normal(size=(8, 2)), np.arange(8) % 2
        if self.task == "classification":
            classifier = self.make_classifier(device)
            clone(classifier).fit(X_toy, y_toy)
            # The model is itself the classifier, fitted on flat tabular rows.
            model = make_pipeline(FunctionTransformer(to_tabular), classifier)
        elif self.task == "forecasting":
            regressor = self.make_regressor()
            clone(regressor).fit(X_toy, y_toy.astype(float))
            model = TabICLForecaster(max_context_length=self.max_context_length)
            model.predictor = _RegressorEngine(regressor, self.predict_quantiles)
        else:
            raise ValueError(f"{self.name} does not support task {self.task!r}")
        return model

    def build_adapter(self, task, model):
        if task == "classification":
            return model.fit(self.X_train, np.asarray(self.y_train))
        return super().build_adapter(task, model)

    def forecast_batch(self, inputs, covariates, prediction_length):
        preds = forecast_windows(
            self.model,
            inputs,
            prediction_length,
            self.quantile_levels,
            self.meta.get("freq"),
        )
        return [torch.from_numpy(p) for p in preds]  # each (H, C, Q)
