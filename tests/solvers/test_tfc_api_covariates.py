"""TFC-API adapter threads covariates through to the SDK.

Uses a fake ``cross_validate`` that records its kwargs and returns a minimal
forecast frame — no network, no API key. The adapter receives whatever
covariates the objective hands it (already masked), so passing an empty
``Covariates`` here exercises the deactivated path.
"""

import importlib

import numpy as np
import pandas as pd

from benchmark_utils.covariates import Covariates
from benchmark_utils.inputs import ForecastInput

tfc_api = importlib.import_module("solvers.tfc_api")
_TFCAPIForecaster = tfc_api._TFCAPIForecaster

MODEL = "mymodel"  # a str has no ``supports_batching`` → per-series path
H = 3


class _FakeClient:
    def __init__(self):
        self.calls = []

    def cross_validate(self, train_df, *, model, horizon, freq, fcds, quantiles,
                       context, add_holidays, add_events, country_isocode,
                       historical_variables, future_variables, batch_size):
        self.calls.append({
            "columns": list(train_df.columns),
            "historical_variables": historical_variables,
            "future_variables": future_variables,
        })
        rows = []
        for uid in train_df["unique_id"].unique():
            ds_vals = sorted(train_df.loc[train_df["unique_id"] == uid, "ds"].unique())
            for fcd in fcds:
                future_ds = [d for d in ds_vals if d > fcd][:horizon]
                for d in future_ds:
                    rows.append(
                        {"unique_id": uid, "ds": d, "fcd": fcd, str(model): 0.0}
                    )
        return pd.DataFrame(rows)


def _adapter(client):
    return _TFCAPIForecaster(
        client=client, model=MODEL, prediction_length=H, freq="D",
        context=None, quantiles=None, add_holidays=False, add_events=False,
        country_isocode=None, batch_size=256,
    )


def _forecast_input(covariates):
    series = np.arange(10, dtype=np.float32)[:, None]  # (10, 1)
    return ForecastInput(
        x=[series], cutoff_indexes=[[5]], covariates=covariates
    )


def test_covariates_are_passed_to_sdk():
    client = _FakeClient()
    out = _adapter(client).predict(_forecast_input(Covariates(
        static_covars=[],
        hist_covars=[np.zeros((10, 1), dtype=np.float32)],
        future_covars=[np.ones((10, 2), dtype=np.float32)],
    )))

    call = client.calls[0]
    assert call["historical_variables"] == ["hist_0"]
    assert call["future_variables"] == ["future_0", "future_1"]
    assert {"hist_0", "future_0", "future_1"}.issubset(call["columns"])
    # Sanity: a well-formed forecast came back.
    assert out.point[0].shape == (1, H, 1)


def test_no_covariates_passes_none():
    client = _FakeClient()
    _adapter(client).predict(_forecast_input(Covariates()))

    call = client.calls[0]
    assert call["historical_variables"] is None
    assert call["future_variables"] is None
    assert not any(
        c.startswith(("hist_", "future_")) for c in call["columns"]
    )
