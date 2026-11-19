"""TabPFN-v2 solver for UCR/UEA classification and forecasting.

TabPFN is a Prior-fitted Network (PFN) — a Transformer pre-trained over
synthetic tabular datasets via meta-learning. Each ``(T, C)`` time-series
sample is flattened to a ``(T*C,)`` feature row and the whole training matrix
is passed to ``TabPFNClassifier``. The model *is* the classifier, so the
default embedding + linear-probe classification strategy does not apply and
``build_adapter`` is overridden to fit it directly.

For forecasting there is no first-party TabPFN pipeline, so we reuse tabicl's
forecasting transforms/preprocessing and swap in ``TabPFNRegressor`` (see
:mod:`benchmark_utils.tabular_forecast`). This adds a ``tabicl`` requirement.

References:
    https://github.com/PriorLabs/TabPFN
    https://doi.org/10.1038/s41586-024-08328-6
"""

import numpy as np
import torch
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from benchmark_utils.base_solver import BaseTSFMSolver
from benchmark_utils.tabular_forecast import (
    DEFAULT_FORECAST_QUANTILES,
    build_forecaster,
    forecast_windows,
    to_tabular,
)


class _TabPFNAdapter:
    """Wrap a fitted TabPFNClassifier behind the benchmark adapter API.

    ``predict`` receives the entire test list in one call — scoring all test
    rows at once is significantly faster than individual calls because TabPFN
    recomputes the training context on every call.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(to_tabular(X))


class Solver(BaseTSFMSolver):
    """TabPFN-v2 in-context learning solver for UCR/UEA classification."""

    name = "TabPFN-v2"

    # tabicl provides the forecasting transforms reused for TabPFN forecasting.
    requirements = ["pip::tabpfn", "pip::tabicl"]

    parameters = {
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
        return f"{self.task}:tabpfn-v2:{self.n_estimators}"

    def load_model(self, device, dtype):
        # For forecasting the "model" is the (tabicl-transforms + TabPFN)
        # pipeline; the regressor itself loads lazily inside forecast_batch.
        if self.task == "forecasting":
            self.inference_batch_size = self.forecast_inference_batch_size
            return build_forecaster(
                "tabpfn",
                max_context_length=self.max_context_length,
                point_estimate=self.point_estimate,
                config={"n_estimators": self.n_estimators},
            )
        # Classifier instantiation (and checkpoint download) is untimed.
        # TabPFN handles device placement itself via device="auto".
        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V2,
            n_estimators=self.n_estimators,
            device="auto",
            random_state=42,
            ignore_pretraining_limits=True,
        )

    def build_adapter(self, task, model):
        if task == "classification":
            # TabPFN is itself the classifier: fit it on flat tabular rows.
            model.fit(to_tabular(self.X_train), np.asarray(self.y_train))
            return _TabPFNAdapter(model)
        # Forecasting: BaseTSFMSolver builds the adapter from forecast_batch.
        return super().build_adapter(task, model)

    def forecast_batch(self, inputs, covariates, prediction_length):
        # self.model is the (tabicl-transforms + TabPFN) pipeline.
        preds = forecast_windows(
            self.model,
            inputs,
            prediction_length,
            self.quantile_levels,
            self.meta.get("freq"),
        )
        return [torch.from_numpy(p) for p in preds]  # each (H, C, Q)
