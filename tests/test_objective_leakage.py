"""Integration test: the Objective disqualifies leaky forecasters.

A leakage-free model is scored normally (``leakage == 0``); a model that
reads the future target is flagged (``leakage == 1``) and every metric is
set to ``+inf`` so it cannot win the benchmark.
"""

import numpy as np
import pytest

benchopt = pytest.importorskip("benchopt")

from benchmark_utils.adapters.base import BaseTSFMAdapter  # noqa: E402
from benchmark_utils.inputs import ForecastInput  # noqa: E402
from benchmark_utils.outputs import ForecastOutput  # noqa: E402
from objective import Objective  # noqa: E402

H = 4


class _Clean(BaseTSFMAdapter):
    def predict(self, x: ForecastInput) -> ForecastOutput:
        qs = []
        for series, cutoffs in zip(x.x, x.cutoff_indexes):
            series = np.asarray(series)
            C = series.shape[1]
            arr = np.stack(
                [np.broadcast_to(series[:c][-1], (H, C)) for c in cutoffs]
            )[:, None, :, :]
            qs.append(arr.astype(np.float64))
        return ForecastOutput(quantiles=qs, quantile_levels=(0.5,))


class _Leaky(BaseTSFMAdapter):
    def predict(self, x: ForecastInput) -> ForecastOutput:
        qs = []
        for series, cutoffs in zip(x.x, x.cutoff_indexes):
            series = np.asarray(series)
            C = series.shape[1]
            arr = np.stack(
                [series[c:c + H].reshape(H, C) for c in cutoffs]
            )[:, None, :, :]
            qs.append(arr.astype(np.float64))
        return ForecastOutput(quantiles=qs, quantile_levels=(0.5,))


@pytest.fixture
def objective():
    rng = np.random.default_rng(0)
    X = [rng.standard_normal((40, 1)), rng.standard_normal((35, 1))]
    cutoffs = [[30], [25]]
    y_test = [
        np.stack([X[i][c:c + H] for c in cutoffs[i]])  # (n_cutoffs, H, C)
        for i in range(len(X))
    ]
    obj = Objective()
    obj.set_data(
        X_train=X,
        y_train=None,
        X_test=X,
        y_test=y_test,
        task="forecasting",
        metrics=["mae", "mse"],
        cutoff_indexes=cutoffs,
        prediction_length=H,
    )
    return obj


def test_clean_model_scored_normally(objective):
    result = objective.evaluate_result(_Clean())
    assert result["leakage"] == 0.0
    assert np.isfinite(result["mae"])
    assert np.isfinite(result["mse"])


def test_leaky_model_is_disqualified(objective):
    result = objective.evaluate_result(_Leaky())
    assert result["leakage"] == 1.0
    assert result["mae"] == float("inf")
    assert result["mse"] == float("inf")
    assert result["value"] == float("inf")
