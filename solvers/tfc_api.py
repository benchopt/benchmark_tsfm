"""TFC API solver for the TSFM benchmark.

Calls The Forecasting Company's hosted inference API via the official
``theforecastingcompany`` Python SDK. Supports zero-shot forecasting.

Authentication
--------------
The SDK reads ``TFC_API_KEY`` from the environment by default. Sign in at
https://docs.retrocast.com/settings/api-keys to get one.

Adding a new model
------------------
Pass any model id from ``theforecastingcompany.utils.TFCModels`` via the
``model`` parameter (e.g. ``"chronos-2"``, ``"timesfm-2p5"``,
``"tfc-global"``, ``"T0-1638-step-85000"``).
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
from benchopt import BaseSolver

from benchmark_utils.adapters.base import BaseTSFMAdapter


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


class _TFCAPIForecaster(BaseTSFMAdapter):
    """Batched adapter that calls ``client.cross_validate`` per series."""

    def __init__(
        self,
        client,
        model,
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

    def predict(self, x, cutoff_indexes, covariates, horizon):
        # TODO: thread ``covariates`` (static/hist/future) through to the SDK
        # once the benchmark datasets expose them. For now the dict is
        # ignored — Monash datasets carry no covariates.
        del covariates
        pd_freq = _to_pandas_freq(self.freq)

        results = []
        for series_idx, (series, cutoffs) in enumerate(zip(x, cutoff_indexes)):
            series = np.asarray(series, dtype=np.float32)
            if series.ndim == 1:
                series = series[:, None]
            T, C = series.shape
            index = pd.date_range("2000-01-01", periods=T, freq=pd_freq)

            frames = []
            for c in range(C):
                frames.append(
                    pd.DataFrame(
                        {
                            "unique_id": f"s{series_idx}_c{c}",
                            "ds": index,
                            "target": series[:, c],
                        }
                    )
                )
            train_df = pd.concat(frames, ignore_index=True)

            fcds = [pd.Timestamp(index[cutoff]) for cutoff in cutoffs]
            forecast_df = self.client.cross_validate(
                train_df,
                model=self.model,
                horizon=horizon,
                freq=self.freq,
                fcds=fcds,
                quantiles=self.quantiles,
                context=self.context,
                add_holidays=self.add_holidays,
                add_events=self.add_events,
                country_isocode=self.country_isocode,
                batch_size=self.batch_size,
            )

            value_col = f"{self.model}_q0.5"
            if value_col not in forecast_df.columns:
                value_col = str(self.model)
            if value_col not in forecast_df.columns:
                raise ValueError(
                    f"TFC API response missing expected columns; got {list(forecast_df.columns)!r}"
                )

            preds = np.empty((len(cutoffs), horizon, C), dtype=np.float32)
            for c in range(C):
                channel = forecast_df.loc[forecast_df["unique_id"] == f"s{series_idx}_c{c}"]
                for k, fcd in enumerate(fcds):
                    window = channel.loc[channel["fcd"] == fcd].sort_values("ds").head(horizon)
                    preds[k, :, c] = window[value_col].to_numpy(dtype=np.float32)
            results.append(preds)
        return results


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

    sampling_strategy = "run_once"

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
