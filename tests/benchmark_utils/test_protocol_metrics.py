"""Differential tests: ``gluonts_*`` and ``fev_*`` vs. the real upstream code.

The expected values below were produced once by the real upstream scorers
on the inputs built by :func:`make_case`: gluonts 0.15.1
(``evaluate_forecasts`` with GIFT-Eval's arguments, multivariate rows split
with ``to_univariate``) and fev at edcc51ad (``Task.evaluation_summary``
with fev-bench's metric config), in float64 on the float32 inputs. Neither
package is a dependency, so they are hardcoded. ``time_*`` is checked
against the vendored upstream function in ``test_time_metrics.py``.
"""

from pathlib import Path

import numpy as np
import pytest

from benchmark_utils import metrics as M
from benchmark_utils.adapters.base import BaseTSFMAdapter
from benchmark_utils.covariates import Covariates
from benchmark_utils.outputs import ForecastOutput

GLUONTS = [
    "gluonts_mase",
    "gluonts_crps",
    "gluonts_mae",
    "gluonts_mse",
    "gluonts_rmse",
    "gluonts_nd",
    "gluonts_smape",
    "gluonts_mape",
]
FEV = ["fev_sql", "fev_mase", "fev_wql", "fev_wape"]

# Chronos-2's 21 levels, as float32 (so they only match within 1e-6).
CHRONOS2_LEVELS = tuple(
    np.float32(
        [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        + [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    ).tolist()
)
NINE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def make_case(seed, n_variates, seasonality, levels):
    """A task with ragged windows and every NaN / degenerate-scale case.

    Windows follow fev-bench's layout: ``K`` global cutoffs ``H`` steps
    apart, anchored at each series end; a series keeps the windows with at
    least one past observation.
    """
    rng = np.random.default_rng(seed)
    H, K, C, amplitude = 6, 4, n_variates, 10.0
    lengths = [K * H + 30, K * H + 1, 2 * H + 3, H + 2, K * H + 15, K * H + 40, 3 * H]
    series, cutoffs, quantiles = [], [], []
    for i, T in enumerate(lengths):
        x = rng.normal(1.0, 0.5, size=(T, C)) * amplitude
        x[rng.random((T, C)) < 0.1] = np.nan  # gaps in history and labels
        cs = [T - (K - w) * H for w in range(K) if T - (K - w) * H >= 1]
        if i == 0:
            x[: cs[-1]] = 3.0 * amplitude  # constant history: zero scale
        if i == 1:
            x[1::2, 0] = np.nan  # every lag-1 pair has a NaN end
        if i == 2:
            x[cs[-1] :, 0] = np.nan  # all-NaN label window (variate 0)
        if i == 4:
            x[: cs[0], -1] = np.nan  # all-NaN history (last variate)
        if i == 5:
            x[cs[0] : cs[0] + H] = 0.0  # all-zero label window
        center = rng.normal(1.0, 0.5, size=(len(cs), H, C, 1)) * amplitude
        spread = rng.normal(0, 0.3 * amplitude, size=(len(cs), H, C, len(levels)))
        q = center + np.sort(spread, axis=-1)
        if i == 3:
            mid = len(levels) // 2
            q[0, 0] = q[0, 0, :, mid : mid + 1]  # degenerate quantiles
        series.append(x.astype(np.float32))
        cutoffs.append(cs)
        quantiles.append(q.astype(np.float32))
    return dict(
        series=series,
        cutoffs=cutoffs,
        quantiles=quantiles,
        levels=levels,
        H=H,
        seasonality=seasonality,
    )


# (case, upstream values). Case 0: univariate, Chronos-2 levels, history
# shorter than the season for some series. Case 1: bivariate, the nine
# levels, daily season on hourly data.
CASES = [
    (
        make_case(1, 1, 5, CHRONOS2_LEVELS),
        {
            "gluonts_mase": 0.9941664500733767,
            "gluonts_crps": 0.5673456960995416,
            "gluonts_mae": 8.474600174881163,
            "gluonts_mse": 116.48081779076911,
            "gluonts_rmse": 10.792627937197182,
            "gluonts_nd": 0.6355584361948409,
            "gluonts_smape": 0.7857367909824101,
            "gluonts_mape": 0.727665652195208,
            "fev_sql": 0.9317434820763635,
            "fev_mase": 1.0688810405273537,
            "fev_wql": 0.5783505545610811,
            "fev_wape": 0.6466849496541589,
        },
    ),
    (
        make_case(4, 2, 24, NINE_LEVELS),
        {
            "gluonts_mase": 1.363071471174737,
            "gluonts_crps": 0.528168418164953,
            "gluonts_mae": 8.10710589349542,
            "gluonts_mse": 107.5223373190901,
            "gluonts_rmse": 10.369297821891804,
            "gluonts_nd": 0.613148191006813,
            "gluonts_smape": 0.744112593322379,
            "gluonts_mape": 0.9619076265962048,
            "fev_sql": 1.1160096141377265,
            "fev_mase": 1.3369042941031917,
            "fev_wql": 0.544291332005133,
            "fev_wape": 0.6293396464224421,
        },
    ),
]


def _inputs(case, quantiles=None):
    """Objective-style inputs: (M, H, C) truth, forecast and metric kwargs."""
    H = case["H"]
    series, cutoffs = case["series"], case["cutoffs"]
    y_true = np.concatenate(
        [np.stack([x[c : c + H] for c in cs]) for x, cs in zip(series, cutoffs)]
    )
    forecast = ForecastOutput(
        quantiles=case["quantiles"] if quantiles is None else quantiles,
        quantile_levels=case["levels"],
    )
    kwargs = dict(
        contexts=[x[:c] for x, cs in zip(series, cutoffs) for c in cs],
        seasonality=case["seasonality"],
        row_index=[
            (i, len(cs) - 1 - j) for i, cs in enumerate(cutoffs) for j in range(len(cs))
        ],
    )
    return y_true, forecast, kwargs


def _ours(case, names, quantiles=None):
    y_true, forecast, kwargs = _inputs(case, quantiles)
    return {name: M.ALL_METRICS[name](y_true, forecast, **kwargs) for name in names}


def _assert_close(ours, expected):
    for name, value in expected.items():
        np.testing.assert_allclose(
            ours[name], value, rtol=1e-10, atol=0, equal_nan=True, err_msg=name
        )


# --- tests -------------------------------------------------------------------


def test_matches_stored_reference():
    for case, expected in CASES:
        _assert_close(_ours(case, GLUONTS + FEV), expected)


def test_nan_forecast_is_nan_with_warning():
    """Upstream raises; we report NaN, never a better number."""
    case, _ = CASES[1]
    quantiles = [q.copy() for q in case["quantiles"]]
    quantiles[1][0, 0, 0, :] = np.nan
    for name in ("gluonts_mase", "fev_sql"):
        with pytest.warns(RuntimeWarning, match="NaN or inf"):
            value = _ours(case, [name], quantiles)[name]
        assert np.isnan(value)


class _Replay(BaseTSFMAdapter):
    """Returns precomputed forecasts (leakage-free by construction)."""

    def __init__(self, forecast):
        self.forecast = forecast

    def predict(self, x):
        return self.forecast


def test_objective_passes_contexts_and_row_index():
    from benchopt.benchmark import Benchmark

    Objective = Benchmark(Path(__file__).parents[2]).get_benchmark_objective()
    case, expected = CASES[1]
    H = case["H"]
    series, cutoffs = case["series"], case["cutoffs"]
    obj = Objective.get_instance()
    obj.set_data(
        X_train=[x[: cs[0]] for x, cs in zip(series, cutoffs)],
        y_train=None,
        X_test=series,
        y_test=[np.stack([x[c : c + H] for c in cs]) for x, cs in zip(series, cutoffs)],
        cutoff_indexes=cutoffs,
        covariates=Covariates(),
        task="forecasting",
        metrics=FEV + GLUONTS,
        prediction_length=H,
        freq="H",
        seasonality=case["seasonality"],
    )
    forecast = ForecastOutput(
        quantiles=case["quantiles"], quantile_levels=case["levels"]
    )
    result = obj.evaluate_result(_Replay(forecast))
    _assert_close(result, expected)
    assert result["value"] == result["fev_sql"]
