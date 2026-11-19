"""TabICL-v2 solver for UCR/UEA classification and forecasting.

TabICL is a tabular foundation model that classifies via in-context learning
(ICL) in a single forward pass. Each ``(T, C)`` sample is flattened to a
``(T*C,)`` feature row and the whole training matrix is passed to
``TabICLClassifier`` — the model *is* the classifier, so the default
embedding + linear-probe classification strategy does not apply and
``build_adapter`` is overridden to fit it directly.

For forecasting, TabICL ships a dedicated pipeline (``tabicl.forecast``) that
recasts the problem as tabular regression on timestamp features; the solver
exposes it through ``forecast_batch`` (see
:mod:`benchmark_utils.tabular_forecast`) and ``BaseTSFMSolver`` builds the
forecasting adapter automatically.

References:
    https://github.com/soda-inria/tabicl
    https://huggingface.co/tabicl
"""

import numpy as np
import torch
from tabicl import TabICLClassifier

from benchmark_utils.base_solver import BaseTSFMSolver
from benchmark_utils.tabular_forecast import (
    DEFAULT_FORECAST_QUANTILES,
    build_forecaster,
    forecast_windows,
    to_tabular,
)


class _TabICLAdapter:
    """Wrap a fitted TabICLClassifier behind the benchmark adapter API.

    ``predict`` receives the entire test list in one call because TabICL is an
    in-context learner — all test rows are scored jointly against the stored
    training context.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(to_tabular(X))


class Solver(BaseTSFMSolver):
    """TabICL-v2 in-context learning solver for UCR/UEA classification."""

    name = "TabICL-v2"

    requirements = ["pip::tabicl"]

    parameters = {
        "checkpoint_version": ["tabicl-classifier-v2-20260212.ckpt"],
        "n_estimators": [8],
        # Forecasting knobs: single-valued (not swept) but exposed so they
        # show up in results/parametrized names rather than being hidden
        # class attributes. A short context keeps per-series fitting
        # tractable; the temporal features already encode seasonality, so a
        # long raw history adds little.
        "max_context_length": [512],
        "point_estimate": ["mean"],
        # Push all of a dataset's series through the forecaster in one batch
        # so the GPU worker pool (and checkpoint load) spins up once, not
        # per 128 windows.
        "forecast_inference_batch_size": [4096],
    }

    @property
    def supported_tasks(self):
        return {"classification", "forecasting"}

    @property
    def quantile_levels(self):
        return DEFAULT_FORECAST_QUANTILES

    @property
    def model_id(self):
        return f"{self.task}:{self.checkpoint_version}:{self.n_estimators}"

    def load_model(self, device, dtype):
        # For forecasting the "model" is the tabicl forecasting pipeline; the
        # regressor itself loads lazily per series inside forecast_batch.
        if self.task == "forecasting":
            self.inference_batch_size = self.forecast_inference_batch_size
            return build_forecaster(
                "tabicl",
                max_context_length=self.max_context_length,
                point_estimate=self.point_estimate,
                config={"n_estimators": self.n_estimators},
            )
        # Classifier instantiation (and checkpoint download) is untimed.
        return TabICLClassifier(
            checkpoint_version=self.checkpoint_version,
            n_estimators=self.n_estimators,
            device=device,
            random_state=42,
            verbose=False,
        )

    def build_adapter(self, task, model):
        if task == "classification":
            # TabICL is itself the classifier: fit it on flat tabular rows.
            model.fit(to_tabular(self.X_train), np.asarray(self.y_train))
            return _TabICLAdapter(model)
        # Forecasting: BaseTSFMSolver builds the adapter from forecast_batch.
        return super().build_adapter(task, model)

    def forecast_batch(self, inputs, covariates, prediction_length):
        # self.model is the tabicl forecasting pipeline (see load_model).
        preds = forecast_windows(
            self.model, inputs, prediction_length,
            self.quantile_levels, self.meta.get("freq"),
        )
        return [torch.from_numpy(p) for p in preds]  # each (H, C, Q)
