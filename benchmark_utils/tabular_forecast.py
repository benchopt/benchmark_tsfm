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

import numpy as np
import pandas as pd

from tabicl.forecast import TabICLForecaster
from tabicl.forecast._engine import ForecastEngine, _to_numpy

# Quantile levels forecast by default; includes 0.5 so a median point estimate
# is always available to ForecastOutput.point.
DEFAULT_FORECAST_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# Anchor for synthesized timestamps (see forecast_windows). Pandas' ns-precision
# Timestamp only spans ~1677-2262; starting at 2000 leaves just ~262 years of
# forward headroom, which yearly-freq series (e.g. GiftEval m4_yearly, up to
# ~280 points) can overflow. 1700 gives ~560 years of headroom instead — the
# actual date is otherwise arbitrary, these timestamps are synthetic.
SYNTH_TS_START = "1700-01-01"


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
    regular index for the temporal encoders; the alias only needs to be valid.
    """
    if not freq:
        return "h"
    try:
        pd.date_range(SYNTH_TS_START, periods=2, freq=freq)
        return freq
    except (ValueError, TypeError):
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

    frames = []
    shapes = []
    for j, window in enumerate(inputs):
        window = np.asarray(window, dtype=np.float32)
        if window.ndim == 1:
            window = window[:, None]
        C = window.shape[1]
        shapes.append(C)
        ts = pd.date_range(SYNTH_TS_START, periods=window.shape[0], freq=freq)
        for c in range(C):
            frames.append(
                pd.DataFrame(
                    {
                        "item_id": f"{j}:{c}",
                        "timestamp": ts,
                        "target": window[:, c],
                    }
                )
            )

    if not frames:
        return []

    context_df = pd.concat(frames, ignore_index=True)
    pred = forecaster.predict_df(
        context_df, prediction_length=prediction_length, quantiles=qs
    )

    outputs = []
    for j, C in enumerate(shapes):
        out = np.empty((prediction_length, C, len(qs)), dtype=np.float32)
        for c in range(C):
            grp = pred.xs(f"{j}:{c}", level="item_id")
            out[:, c, :] = grp[qs].to_numpy()[:prediction_length]
        outputs.append(out)
    return outputs
