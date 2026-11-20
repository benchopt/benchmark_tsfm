"""TabICL-v2 solver for UCR/UEA classification and forecasting.

TabICL is a tabular foundation model that predicts by in-context learning
(ICL) in a single forward pass. See :mod:`benchmark_utils.tabular_fm` for how
time series are cast to tables.

References:
    https://github.com/soda-inria/tabicl
    https://huggingface.co/tabicl
"""

from tabicl import TabICLClassifier, TabICLRegressor

from benchmark_utils.tabular_fm import TabularFMSolver


class Solver(TabularFMSolver):
    """TabICL-v2 in-context learning solver."""

    name = "TabICL-v2"

    requirements = ["pip::tabicl[forecast]"]

    parameters = {
        # Classification only: forecasting uses the default regressor ckpt.
        "checkpoint_version": ["tabicl-classifier-v2-20260212.ckpt"],
        "n_estimators": [8],
        # A short context keeps per-series fitting tractable; the temporal
        # features already encode seasonality.
        "max_context_length": [512],
        # One regressor is fitted per series: large batches let tabicl's
        # worker pool start once per dataset rather than per 128 windows.
        "inference_batch_size": [4096],
    }

    @staticmethod
    def predict_quantiles(model, X, levels):
        return model.predict(X, output_type="quantiles", alphas=levels)

    def make_classifier(self, device):
        return TabICLClassifier(
            checkpoint_version=self.checkpoint_version,
            n_estimators=self.n_estimators,
            device=device,
            random_state=42,
            verbose=False,
        )

    def make_regressor(self):
        return TabICLRegressor(n_estimators=self.n_estimators)
