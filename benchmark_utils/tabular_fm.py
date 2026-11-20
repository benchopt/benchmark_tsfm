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

    if not inputs:
        return []
    # tabicl only uses the last max_context_length steps.
    windows = [
        np.asarray(w, dtype=np.float32).reshape(len(w), -1)[
            -forecaster.max_context_length :
        ]
        for w in inputs
    ]
    shapes = [w.shape[1] for w in windows]
    # One tabicl item per (window, channel), with its history length.
    ids = [f"{j}:{c}" for j, C in enumerate(shapes) for c in range(C)]
    lengths = np.repeat([w.shape[0] for w in windows], shapes)
    # Explicit future timestamps: tabicl would otherwise infer the freq,
    # which fails on histories shorter than 3 steps.
    ts = pd.date_range(
        SYNTH_TS_START,
        periods=lengths.max() + prediction_length,
        freq=freq,
        unit="ms",
    )
    # Step index of each row within its item, for the history then the future.
    ctx_pos = np.arange(lengths.sum()) - np.repeat(
        np.cumsum(lengths) - lengths, lengths
    )
    fut_pos = np.repeat(lengths, prediction_length) + np.tile(
        np.arange(prediction_length), len(ids)
    )
    context = pd.DataFrame(
        {
            "item_id": np.repeat(ids, lengths),
            "timestamp": ts[ctx_pos],
            "target": np.concatenate([w.T.ravel() for w in windows]),
        }
    )
    future = pd.DataFrame(
        {"item_id": np.repeat(ids, prediction_length), "timestamp": ts[fut_pos]}
    )

    pred = forecaster.predict_df(context, future_df=future, quantiles=qs)

    # Rows in input order (tabicl sorts item ids), prediction_length per item.
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
        """Return an unfitted classifier placed on ``device``.

        It must accept ``random_state``, set from benchopt's seed at each run.
        """

    @abstractmethod
    def make_regressor(self):
        """Return an unfitted regressor; tabicl assigns it a device per worker.

        It must accept ``random_state``, set from benchopt's seed at each run.
        """

    @staticmethod
    @abstractmethod
    def predict_quantiles(model, X, levels):
        """Return the ``(n, Q)`` quantiles of ``X`` at ``levels``."""

    def set_objective(self, *args, **kwargs):
        super().set_objective(*args, **kwargs)
        # The model is cached across runs, so each run sets its own seed.
        seed = self.get_seed(use_repetition=True)
        if self.task == "classification":
            self.model[-1].set_params(random_state=seed)
        else:
            self.model.predictor._regressor.set_params(random_state=seed)

    def load_model(self, device, dtype):
        # Weights load lazily in fit(): fit a copy on a toy table so the
        # checkpoint download and load are not timed in run().
        X_toy, y_toy = np.arange(16.0).reshape(8, 2), np.arange(8) % 2
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
