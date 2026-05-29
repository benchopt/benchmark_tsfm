"""The objective masks covariates to the adapter's declared capabilities.

Enforcement lives in ``Objective._eval_forecasting`` so it is guaranteed for
every forecasting model, not reimplemented per solver.
"""

import numpy as np
import pytest

from benchmark_utils.adapters.base import BaseTSFMAdapter
from benchmark_utils.capabilities import FUTURE_COVARIATES, HIST_COVARIATES
from benchmark_utils.covariates import Covariates
from benchmark_utils.outputs import ForecastOutput

H, C = 3, 1


class _RecordingAdapter(BaseTSFMAdapter):
    """Captures the covariates it is handed, returns a valid zero forecast."""

    def __init__(self, covariate_capabilities=frozenset()):
        self.covariate_capabilities = frozenset(covariate_capabilities)
        self.seen = None

    def predict(self, x):
        self.seen = x.covariates
        qs = [
            np.zeros((len(cutoffs), 1, H, C), dtype=np.float32)
            for cutoffs in x.cutoff_indexes
        ]
        return ForecastOutput(quantiles=qs, quantile_levels=(0.5,))


def _make_objective():
    from objective import Objective

    obj = Objective.get_instance()
    series = np.arange(10, dtype=np.float32)[:, None]  # (10, 1)
    cutoffs = [5]
    covariates = Covariates(
        static_covars=[],
        hist_covars=[np.zeros((10, 1), dtype=np.float32)],
        future_covars=[np.ones((10, 2), dtype=np.float32)],
    )
    obj.set_data(
        X_train=[series[:5]],
        y_train=[series[5:8]],
        X_test=[series],
        y_test=[np.zeros((1, H, C), dtype=np.float32)],
        cutoff_indexes=[cutoffs],
        covariates=covariates,
        task="forecasting",
        metrics=["mae"],
        prediction_length=H,
    )
    return obj


@pytest.mark.parametrize(
    "active, expect_hist, expect_future",
    [
        (frozenset(), False, False),
        ({HIST_COVARIATES}, True, False),
        ({FUTURE_COVARIATES}, False, True),
        ({HIST_COVARIATES, FUTURE_COVARIATES}, True, True),
    ],
)
def test_objective_masks_to_adapter_capabilities(active, expect_hist, expect_future):
    obj = _make_objective()
    adapter = _RecordingAdapter(active)

    obj.evaluate_result(adapter)

    assert (len(adapter.seen.hist_covars) > 0) is expect_hist
    assert (len(adapter.seen.future_covars) > 0) is expect_future


def test_default_adapter_sees_no_covariates():
    """An adapter that declares nothing (base default) runs univariate."""
    obj = _make_objective()
    adapter = _RecordingAdapter()  # inherits empty covariate_capabilities

    obj.evaluate_result(adapter)

    assert adapter.seen.hist_covars == []
    assert adapter.seen.future_covars == []
