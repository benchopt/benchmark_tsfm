"""Shared solver for tabular foundation models (TabICL, TabPFN, ...).

These models predict by in-context learning on a table of rows.

- **Classification** flattens each ``(T, C)`` sample into a ``(T*C,)`` row and
  fits the model's classifier on the training rows.
- **Forecasting** recasts each series as tabular regression on
  timestamp-derived features (calendar / Fourier / auto-seasonal / index),
  reusing tabicl's forecasting pipeline (``tabicl.forecast``) with the model's
  own regressor plugged in through :class:`_RegressorEngine`.

A new model subclasses :class:`TabularFMSolver`.
"""

import copy
import re
import warnings
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from tabicl.forecast import TabICLForecaster
from tabicl.forecast._engine import ForecastEngine, _to_numpy

from benchmark_utils.base_solver import BaseTSFMSolver
from benchmark_utils.forecasting_constants import CANONICAL_QUANTILE_LEVELS

# Anchor for synthesized timestamps (see forecast_windows). Pandas'
# ns-precision Timestamp ends in 2262, so yearly windows must stay under ~560
# steps: forecast_windows trims the history to max_context_length to ensure it.
SYNTH_TS_START = "1700-01-01"

# Pre-2.2 pandas aliases, rejected by pandas 3.
_LEGACY_FREQ = {
    "A": "YE", "Y": "YE", "AS": "YS", "Q": "QE", "M": "ME",
    "H": "h", "T": "min", "S": "s", "L": "ms", "U": "us", "N": "ns",
}  # fmt: skip
_FREQ_RE = re.compile(r"^(\d*)([A-Za-z]+)(-\w+)?$")


def to_tabular(X):
    """Flatten a list of ``(T, C)`` samples into an ``(N, T*C)`` matrix."""
    arr = np.asarray(X, dtype=np.float32)  # (N, T) or (N, T, C)
    if arr.ndim == 3:
        arr = arr.reshape(arr.shape[0], -1)  # flatten C into features
    return arr


class _RegressorEngine(ForecastEngine):
    """tabicl ForecastEngine that fits a given regressor on each series.

    Only the per-series inference routine changes; tabicl's feature
    engineering and dispatch to workers are inherited. The regressor is
    pickled to the workers, so it must be an unfitted library estimator.
    """

    def __init__(self, regressor, quantiles_kwarg):
        super().__init__()
        self._regressor = regressor
        self._quantiles_kwarg = quantiles_kwarg

    def _inference_routine(self, train_X, train_y, test_X, quantiles=None):
        train_X, train_y, test_X = map(_to_numpy, (train_X, train_y, test_X))
        # Fit a copy so the template never carries fitted weights to workers.
        model = copy.deepcopy(self._regressor).fit(train_X, train_y)
        q = model.predict(
            test_X, output_type="quantiles", **{self._quantiles_kwarg: quantiles}
        )
        # TabPFN returns one array per quantile, TabICL an (n_test, Q) array.
        q = np.stack(q, axis=1) if isinstance(q, list) else np.asarray(q)
        # tabicl requires a point column; the benchmark only reads quantiles.
        result = {"target": q.mean(axis=1)}
        result.update({level: q[:, i] for i, level in enumerate(quantiles)})
        return result


def safe_freq(freq):
    """Return a pandas-usable frequency alias, defaulting to hourly.

    The benchmark ForecastInput carries no timestamps, so we synthesize a
    regular index for the temporal encoders. Legacy aliases are mapped to
    their current spelling, and an unusable alias falls back with a warning.
    """
    if not freq:
        return "h"
    if m := _FREQ_RE.match(freq):
        mult, base, anchor = m.groups()
        freq = f"{mult}{_LEGACY_FREQ.get(base, base)}{anchor or ''}"
    try:
        pd.date_range(SYNTH_TS_START, periods=2, freq=freq)
        return freq
    except (ValueError, TypeError):
        warnings.warn(f"Unusable freq {freq!r}, synthesizing hourly timestamps.")
        return "h"


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
        # tabicl only uses the last max_context_length steps: trimming here
        # keeps the frames small and the synthetic timestamps in range.
        window = window[-forecaster.max_context_length :]
        # tabicl's seasonality detection needs at least 2 points.
        window = np.pad(window, ((max(0, 2 - len(window)), 0), (0, 0)), mode="edge")
        T, C = window.shape
        shapes.append(C)
        # Explicit future timestamps: tabicl would otherwise infer the freq,
        # which fails on histories shorter than 3 steps.
        ts = pd.date_range(SYNTH_TS_START, periods=T + prediction_length, freq=freq)
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

    outputs = []
    for j, C in enumerate(shapes):
        out = np.empty((prediction_length, C, len(qs)), dtype=np.float32)
        for c in range(C):
            grp = pred.xs(f"{j}:{c}", level="item_id")
            out[:, c, :] = grp[qs].to_numpy()[:prediction_length]
        outputs.append(out)
    return outputs


class TabularFMSolver(BaseTSFMSolver):
    """Base solver for tabular foundation models.

    A subclass, in its own ``solvers/`` file, sets:

    - ``name``, ``requirements`` and ``parameters`` as literals: benchopt reads
      them statically before the dependencies are installed, so they are not
      inherited. ``requirements`` must include ``pip::tabicl[forecast]`` and
      ``parameters`` must define ``max_context_length``.
    - ``make_classifier(device)``: an unfitted scikit-learn classifier.
    - ``make_regressor()``: an unfitted, picklable regressor whose
      ``predict(X, output_type="quantiles", <quantiles_kwarg>=levels)``
      returns an ``(n, Q)`` array or a list of ``Q`` arrays.
    - ``quantiles_kwarg``: the name of that quantile-levels argument.

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

    def load_model(self, device, dtype):
        # Weights load lazily in fit(): fit a copy on a toy table so the
        # checkpoint download and load are not timed in run().
        X_toy, y_toy = np.random.default_rng(0).normal(size=(8, 2)), np.arange(8) % 2
        if self.task == "classification":
            classifier = self.make_classifier(device)
            copy.deepcopy(classifier).fit(X_toy, y_toy)
            # The model is itself the classifier, fitted on flat tabular rows.
            model = make_pipeline(FunctionTransformer(to_tabular), classifier)
        else:
            regressor = self.make_regressor()
            copy.deepcopy(regressor).fit(X_toy, y_toy.astype(float))
            model = TabICLForecaster(max_context_length=self.max_context_length)
            model.predictor = _RegressorEngine(regressor, self.quantiles_kwarg)
        return model

    def build_adapter(self, task, model):
        if task != "classification":
            return super().build_adapter(task, model)
        return model.fit(self.X_train, np.asarray(self.y_train))

    def forecast_batch(self, inputs, covariates, prediction_length):
        preds = forecast_windows(
            self.model,
            inputs,
            prediction_length,
            self.quantile_levels,
            self.meta.get("freq"),
        )
        return [torch.from_numpy(p) for p in preds]  # each (H, C, Q)
