"""Forecasting helper for tabular in-context models (TabICL / TabPFN).

Both models forecast by recasting the task as tabular regression on
timestamp-derived features (calendar / Fourier / auto-seasonal / index).
Rather than re-implement that feature engineering, we reuse tabicl's
forecasting pipeline (``tabicl.forecast``):

- **TabICL** uses :class:`~tabicl.forecast.TabICLForecaster` directly.
- **TabPFN** reuses the *same* pipeline (transforms, preprocessing, per-series
  dispatch) but swaps the underlying regressor for ``TabPFNRegressor`` via
  :class:`_TabPFNForecastEngine`. This couples the TabPFN solver to ``tabicl``.

A solver only needs to build a forecaster (:func:`build_forecaster`) and call
:func:`forecast_windows` from its ``forecast_batch``; ``BaseTSFMSolver`` then
handles all ForecastInput/ForecastOutput plumbing automatically.
"""

import re
import warnings

import numpy as np
import pandas as pd

from tabicl.forecast import TabICLForecaster
from tabicl.forecast._engine import ForecastEngine, _to_numpy

# Quantile levels forecast by default; includes 0.5 so a median point estimate
# is always available to ForecastOutput.point.
DEFAULT_FORECAST_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

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


class _TabPFNForecastEngine(ForecastEngine):
    """tabicl ForecastEngine that predicts with ``TabPFNRegressor``.

    Only the per-series inference routine changes; all feature engineering and
    dispatch are inherited from tabicl.
    """

    def __init__(self, tabpfn_config=None, point_estimate="mean"):
        super().__init__(tabicl_config=None, point_estimate=point_estimate)
        self._tabpfn_config = tabpfn_config or {}

    def _inference_routine(self, train_X, train_y, test_X, quantiles=None):
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion

        if quantiles is None:
            quantiles = list(DEFAULT_FORECAST_QUANTILES)
        train_X, train_y, test_X = map(_to_numpy, (train_X, train_y, test_X))

        # Pin the open V2 regressor: newer versions (v2.5+) gate weights behind
        # a license token. V2 matches the classification solver and needs none.
        model = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2, ignore_pretraining_limits=True, **self._tabpfn_config
        )
        model.fit(train_X, train_y)
        q = np.asarray(
            model.predict(test_X, output_type="quantiles", quantiles=quantiles)
        )

        if 0.5 in quantiles:
            point = q[quantiles.index(0.5)]
        else:
            point = q.mean(axis=0)
        result = {"target": point}
        result.update({level: q[i] for i, level in enumerate(quantiles)})
        return result


def build_forecaster(
    kind,
    *,
    max_context_length,
    temporal_features=None,
    point_estimate="mean",
    config=None,
):
    """Build a tabicl-pipeline forecaster backed by TabICL or TabPFN."""
    forecaster = TabICLForecaster(
        max_context_length=max_context_length,
        temporal_features=temporal_features,
        point_estimate=point_estimate,
        tabicl_config=config if kind == "tabicl" else None,
    )
    if kind == "tabpfn":
        # Reuse the transforms built above; only the regressor differs.
        forecaster.predictor = _TabPFNForecastEngine(
            tabpfn_config=config, point_estimate=point_estimate
        )
    elif kind != "tabicl":
        raise ValueError(f"Unknown forecaster kind {kind!r}")
    return forecaster


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
    forecaster : object with ``predict_df``
        Built via :func:`build_forecaster`.
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
