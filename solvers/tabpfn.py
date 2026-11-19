"""TabPFN-v2 solver for UCR/UEA classification and forecasting.

TabPFN is a Prior-fitted Network (PFN), a Transformer pre-trained over
synthetic tabular datasets. See :mod:`benchmark_utils.tabular_fm` for how time
series are cast to tables. Forecasting reuses tabicl's pipeline, hence the
``tabicl`` requirement. The open V2 weights are used: newer versions (v2.5+)
are gated behind a license token.

References:
    https://github.com/PriorLabs/TabPFN
    https://doi.org/10.1038/s41586-024-08328-6
"""

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion

from benchmark_utils.tabular_fm import TabularFMSolver


class Solver(TabularFMSolver):
    """TabPFN-v2 in-context learning solver."""

    name = "TabPFN-v2"

    requirements = ["pip::tabpfn", "pip::tabicl[forecast]"]

    parameters = {
        "n_estimators": [8],
        # A short context keeps per-series fitting tractable; the temporal
        # features already encode seasonality.
        "max_context_length": [512],
        # One regressor is fitted per series: large batches let tabicl's
        # worker pool start once per dataset rather than per 128 windows.
        "inference_batch_size": [4096],
    }

    quantiles_kwarg = "quantiles"

    def make_classifier(self, device):
        # "auto" spreads the ensemble over all visible GPUs.
        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V2,
            n_estimators=self.n_estimators,
            device="auto",
            random_state=42,
            ignore_pretraining_limits=True,
        )

    def make_regressor(self):
        return TabPFNRegressor.create_default_for_version(
            ModelVersion.V2,
            n_estimators=self.n_estimators,
            ignore_pretraining_limits=True,
        )
