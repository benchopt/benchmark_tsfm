"""TFC API solver for the TSFM benchmark.

Calls The Forecasting Company's hosted inference API via the official
``theforecastingcompany`` Python SDK. Supports zero-shot forecasting.

Authentication
--------------
The SDK reads ``TFC_API_KEY`` from the environment by default. Sign in at
https://docs.retrocast.com/settings/api-keys to get one.

Batching
--------
Models that report ``supports_batching == True`` (chronos-2, moirai-2,
T0-1535, T0-1638) are sent in a single ``cross_validate`` call with all
series stacked into one DataFrame. Series are aligned so their cutoffs
share a common set of ``fcds``; the SDK then builds the (V, T) tensor
internally with one ``unique_id`` per series-channel acting as the
group id Chronos-2 keys on. When cutoff offsets-from-end are not
homogeneous across series, the solver falls back to a per-series loop.

Covariates
----------
Static / historical / future covariates from ``ForecastInput.covariates``
are forwarded to the SDK as extra ``train_df`` columns, named via its
``static_variables`` / ``historical_variables`` / ``future_variables``
parameters. Time-varying covariates span the full series timeline, so
``cross_validate`` reads each cutoff's future-covariate values directly
from ``train_df`` (no separate ``future_df`` is needed). Datasets without
covariates (e.g. Monash) carry empty sequences, so nothing is sent and
behaviour is unchanged. Whether a given model actually consumes a covariate
kind is left to the SDK.

Adding a new model
------------------
Pass any model id from ``theforecastingcompany.utils.TFCModels`` via the
``model`` parameter.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
from benchopt import BaseSolver

from benchmark_utils.adapters.base import BaseTSFMAdapter
from benchmark_utils.inputs import ForecastInput
from benchmark_utils.outputs import ForecastOutput

SUPPORTED_TASKS = {"forecasting"}

# Map benchmark freq codes to API-accepted pandas-like aliases.
_FREQ_REMAP = {"T": "min", "S": "10S"}

# pandas >=2 deprecates Y/Q/M/H short forms in ``pd.date_range``; use the
# long forms for synthetic indices but pass the original to the API.
_PD_FREQ_REMAP = {"Y": "YE", "Q": "QE", "M": "ME"}


def _to_api_freq(freq: str) -> str:
    return _FREQ_REMAP.get(freq, freq)


def _to_pandas_freq(api_freq: str) -> str:
    return _PD_FREQ_REMAP.get(api_freq, api_freq)


def _covariate_column_names(covariates) -> dict[str, list[str]]:
    """Map each covariate kind to stable train_df column names.

    Names are derived from the channel count of the first present series.
    A kind that is empty for every series yields an empty list. We assume a
    kind is either present (with the same channel count) for all series or
    absent for all of them — true for every benchmark dataset today.
    """

    def _names(seq, prefix):
        for arr in seq:
            arr = np.asarray(arr)
            n = arr.shape[-1] if arr.ndim else 1
            return [f"{prefix}_{k}" for k in range(n)]
        return []

    return {
        "future": _names(covariates.future_covars, "future"),
        "historical": _names(covariates.hist_covars, "hist"),
        "static": _names(covariates.static_covars, "static"),
    }


def _attach_covariates(frame, index_len, covariates, series_idx, col_names):
    """Add this series' covariate columns to a per-channel target ``frame``.

    Covariates are per-series, so the same values attach to every channel
    frame of ``series_idx``. ``hist``/``future`` arrays are ``(T, Ch)`` and
    must align with the ``index_len`` rows of ``frame``; ``static`` arrays
    are ``(Ch,)`` and broadcast over all rows.
    """
    if series_idx < len(covariates.future_covars):
        arr = np.asarray(covariates.future_covars[series_idx], dtype=np.float32)
        _set_timed_columns(frame, arr, index_len, col_names["future"], "future")
    if series_idx < len(covariates.hist_covars):
        arr = np.asarray(covariates.hist_covars[series_idx], dtype=np.float32)
        _set_timed_columns(frame, arr, index_len, col_names["historical"], "historical")
    if series_idx < len(covariates.static_covars):
        arr = np.asarray(covariates.static_covars[series_idx], dtype=np.float32)
        arr = arr.reshape(-1)
        for k, col in enumerate(col_names["static"]):
            frame[col] = arr[k]


def _set_timed_columns(frame, arr, index_len, columns, kind):
    """Attach time-varying covariate columns, validating their length."""
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] != index_len:
        raise ValueError(
            f"{kind} covariate has length {arr.shape[0]} but the series has "
            f"{index_len} steps; time-varying covariates must align with x."
        )
    for k, col in enumerate(columns):
        frame[col] = arr[:, k]


def _shared_offsets_from_end(x, cutoff_indexes):
    """Return per-series cutoff offsets if shared across series, else None."""
    if not cutoff_indexes:
        return None
    reference = None
    for series, cutoffs in zip(x, cutoff_indexes):
        T = np.asarray(series).shape[0]
        offsets = tuple(T - c for c in cutoffs)
        if reference is None:
            reference = offsets
        elif offsets != reference:
            return None
    return reference


class _TFCAPIForecaster(BaseTSFMAdapter):
    """Adapter calling the TFC SDK.

    Uses a single batched ``cross_validate`` call when the model supports
    batching and series share cutoff offsets; falls back to one call per
    series otherwise.
    """

    def __init__(
        self,
        client,
        model,
        prediction_length: int,
        freq: str,
        context: Optional[int],
        quantiles: Optional[list[float]],
        add_holidays: bool,
        add_events: bool,
        country_isocode: Optional[str],
        batch_size: int,
    ):
        self.client = client
        self.model = model  # TFCModels enum
        self.prediction_length = prediction_length
        self.freq = _to_api_freq(freq)
        if quantiles is None:
            quantiles = [0.5]
        elif 0.5 not in quantiles:
            quantiles = quantiles + [0.5]
        self.quantiles = quantiles
        self.context = context
        self.add_holidays = add_holidays
        self.add_events = add_events
        self.country_isocode = country_isocode
        self.batch_size = batch_size

    def predict(self, x: ForecastInput) -> ForecastOutput:
        # Static / historical / future covariates ride along as extra
        # ``train_df`` columns; the SDK reads them via its ``*_variables``
        # params. Datasets without covariates (e.g. Monash) carry empty
        # sequences, so the column lists are empty and nothing is sent.
        series_list, cutoff_indexes = x.x, x.cutoff_indexes
        covariates = x.covariates
        col_names = _covariate_column_names(covariates)
        pd_freq = _to_pandas_freq(self.freq)

        offsets = _shared_offsets_from_end(series_list, cutoff_indexes)
        if getattr(self.model, "supports_batching", False) and offsets is not None:
            per_series, levels = self._predict_batched(
                series_list, cutoff_indexes, pd_freq, offsets, covariates, col_names
            )
        else:
            per_series, levels = self._predict_per_series(
                series_list, cutoff_indexes, pd_freq, covariates, col_names
            )
        return ForecastOutput(quantiles=per_series, quantile_levels=levels)

    def _predict_per_series(self, x, cutoff_indexes, pd_freq, covariates, col_names):
        per_series = []
        levels = None
        for series_idx, (series, cutoffs) in enumerate(zip(x, cutoff_indexes)):
            series = np.asarray(series, dtype=np.float32)
            if series.ndim == 1:
                series = series[:, None]
            T, C = series.shape
            index = pd.date_range("2000-01-01", periods=T, freq=pd_freq)

            frames = []
            for c in range(C):
                frame = pd.DataFrame(
                    {
                        "unique_id": f"s{series_idx}_c{c}",
                        "ds": index,
                        "target": series[:, c],
                    }
                )
                _attach_covariates(frame, T, covariates, series_idx, col_names)
                frames.append(frame)
            train_df = pd.concat(frames, ignore_index=True)
            fcds = [pd.Timestamp(index[cutoff]) for cutoff in cutoffs]

            forecast_df = self.client.cross_validate(
                train_df,
                model=self.model,
                horizon=self.prediction_length,
                freq=self.freq,
                fcds=fcds,
                quantiles=self.quantiles,
                context=self.context,
                add_holidays=self.add_holidays,
                add_events=self.add_events,
                country_isocode=self.country_isocode,
                batch_size=self.batch_size,
                future_variables=col_names["future"] or None,
                historical_variables=col_names["historical"] or None,
                static_variables=col_names["static"] or None,
            )

            arr, series_levels = self._gather_series_output(
                forecast_df, series_idx, C, cutoffs, fcds
            )
            per_series.append(arr)
            levels = series_levels
        return per_series, (levels if levels is not None else (0.5,))

    def _predict_batched(
        self, x, cutoff_indexes, pd_freq, offsets, covariates, col_names
    ):
        """One ``cross_validate`` call covering every series in ``x``.

        Series are aligned to share an end date so all cutoffs collapse to
        the same set of timestamps. The SDK then groups by ``unique_id``
        when building Chronos-2's (V, T) tensor.
        """
        end = pd.Timestamp("2030-01-01")
        frames = []
        per_series_meta = []  # (series_idx, C, index, cutoffs)
        for series_idx, (series, cutoffs) in enumerate(zip(x, cutoff_indexes)):
            series = np.asarray(series, dtype=np.float32)
            if series.ndim == 1:
                series = series[:, None]
            T, C = series.shape
            index = pd.date_range(end=end, periods=T, freq=pd_freq)
            for c in range(C):
                frame = pd.DataFrame(
                    {
                        "unique_id": f"s{series_idx}_c{c}",
                        "ds": index,
                        "target": series[:, c],
                    }
                )
                _attach_covariates(frame, T, covariates, series_idx, col_names)
                frames.append(frame)
            per_series_meta.append((series_idx, C, index, cutoffs))

        train_df = pd.concat(frames, ignore_index=True)
        # ``offsets`` is (T - cutoff) for any series, so the corresponding
        # timestamp is end - (offset - 1) * delta. We let pandas pick the
        # delta by walking the date_range backwards from ``end``.
        ref_index = pd.date_range(end=end, periods=max(offsets) + 1, freq=pd_freq)
        fcds = sorted({pd.Timestamp(ref_index[-offset]) for offset in offsets})

        forecast_df = self.client.cross_validate(
            train_df,
            model=self.model,
            horizon=self.prediction_length,
            freq=self.freq,
            fcds=fcds,
            quantiles=self.quantiles,
            context=self.context,
            add_holidays=self.add_holidays,
            add_events=self.add_events,
            country_isocode=self.country_isocode,
            batch_size=self.batch_size,
            future_variables=col_names["future"] or None,
            historical_variables=col_names["historical"] or None,
            static_variables=col_names["static"] or None,
        )

        per_series = []
        levels = None
        for series_idx, C, index, cutoffs in per_series_meta:
            series_fcds = [pd.Timestamp(index[cutoff]) for cutoff in cutoffs]
            arr, series_levels = self._gather_series_output(
                forecast_df, series_idx, C, cutoffs, series_fcds
            )
            per_series.append(arr)
            levels = series_levels
        return per_series, (levels if levels is not None else (0.5,))

    def _gather_series_output(self, forecast_df, series_idx, C, cutoffs, fcds):
        # Discover which quantile columns the SDK returned; fall back to
        # the mean column only when no quantiles are present.
        levels, quantile_cols = [], []
        for q in self.quantiles:
            col = f"{self.model}_q{q}"
            if col in forecast_df.columns:
                levels.append(q)
                quantile_cols.append(col)
        if not quantile_cols:
            mean_col = str(self.model)
            if mean_col not in forecast_df.columns:
                raise ValueError(
                    "TFC API response missing expected columns;"
                    f" got {list(forecast_df.columns)!r}"
                )
            levels = [0.5]
            quantile_cols = [mean_col]

        Q = len(levels)
        preds = np.empty((len(cutoffs), self.prediction_length, C, Q), dtype=np.float32)
        for c in range(C):
            channel = forecast_df.loc[forecast_df["unique_id"] == f"s{series_idx}_c{c}"]
            for k, fcd in enumerate(fcds):
                window = (
                    channel.loc[channel["fcd"] == fcd]
                    .sort_values("ds")
                    .head(self.prediction_length)
                )
                for q_idx, col in enumerate(quantile_cols):
                    preds[k, :, c, q_idx] = window[col].to_numpy(dtype=np.float32)
        return preds, tuple(levels)


class Solver(BaseSolver):
    """TFC hosted-API solver.

    Parameters
    ----------
    model : str
        Model id served by the TFC API — must match a value in
        ``theforecastingcompany.utils.TFCModels`` (e.g. ``"chronos-2"``,
        ``"timesfm-2p5"``, ``"tfc-global"``, ``"moirai-2"``).
    context : int or None
        Number of history steps to send to the model. ``None`` lets the
        model use its native maximum.
    add_holidays, add_events : bool
        Whether to attach TFC holiday / event covariates. Requires
        ``country_isocode`` to be set.
    country_isocode : str or None
        ISO country code (e.g. ``"US"``) used by the holiday/event lookup.
    batch_size : int
        Series-per-batch for batching-enabled models (chronos-2, moirai-2).
    """

    name = "TFC-API"

    requirements = ["pip::theforecastingcompany"]

    parameters = {
        "model": ["chronos-2"],
        "context": [None],
        "add_holidays": [False],
        "add_events": [False],
        "country_isocode": [None],
        "batch_size": [256],
    }

    def skip(self, task, **kwargs):
        if task not in SUPPORTED_TASKS:
            return True, f"TFC-API solver does not support task={task!r}"
        if os.getenv("TFC_API_KEY") is None:
            return True, "TFC_API_KEY environment variable not set"
        return False, None

    def set_objective(self, X_train, y_train, task, **meta):
        from theforecastingcompany import TFCClient
        from theforecastingcompany.utils import TFCModels

        self.task = task
        self.X_train = X_train
        self.meta = meta

        try:
            self._model_enum = TFCModels(self.model)
        except ValueError as e:
            known = ", ".join(m.value for m in TFCModels)
            raise ValueError(
                f"Unknown TFC model '{self.model}'. Known SDK models: {known}."
            ) from e

        if not hasattr(self, "_client"):
            self._client = TFCClient()

    def run(self, _):
        self._adapter = _TFCAPIForecaster(
            client=self._client,
            model=self._model_enum,
            prediction_length=self.meta.get("prediction_length", 1),
            freq=self.meta.get("freq", "D"),
            context=self.context,
            quantiles=None,
            add_holidays=self.add_holidays,
            add_events=self.add_events,
            country_isocode=self.country_isocode,
            batch_size=self.batch_size,
        )

    def get_result(self):
        return {"model": self._adapter}
