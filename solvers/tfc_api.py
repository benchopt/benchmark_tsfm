"""TFC API solver for the TSFM benchmark.

Calls The Forecasting Company's hosted inference API via the official
``theforecastingcompany`` Python SDK. Supports zero-shot forecasting.

Authentication
--------------
The SDK reads ``TFC_API_KEY`` from the environment by default. Sign in at
https://docs.retrocast.com/settings/api-keys to get one.

Adding a new model
------------------
Pass any model id supported by the API (see
https://api.retrocast.com/docs/routes/index) via the ``model`` parameter,
e.g. ``"chronos-2"``, ``"timesfm-2p5"``, ``"tfc-global"``, or a custom
``"T0-<run>-step-<n>"`` checkpoint identifier.
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

# pandas >=2 deprecates Y/Q/M/H in favor of YE/QE/ME/h for ``pd.date_range``,
# but the TFC API still expects the short forms. Use the long forms only when
# generating synthetic indices.
_PD_FREQ_REMAP = {"Y": "YE", "Q": "QE", "M": "ME"}


def _to_api_freq(freq: str) -> str:
    return _FREQ_REMAP.get(freq, freq)


def _to_pandas_freq(api_freq: str) -> str:
    return _PD_FREQ_REMAP.get(api_freq, api_freq)


class _TFCAPIForecaster(BaseTSFMAdapter):
    """Per-test-series adapter that calls the TFC forecast endpoint."""

    def __init__(
        self,
        client,
        model: str,
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
        self.model = model
        self.prediction_length = prediction_length
        self.freq = _to_api_freq(freq)
        self.context = context
        # The objective takes the median, so always request 0.5; keep any
        # extra quantiles the caller asked for so the API call is identical
        # to a production setup.
        if quantiles is None:
            quantiles = [0.5]
        elif 0.5 not in quantiles:
            quantiles = quantiles + [0.5]
        self.quantiles = quantiles
        self.add_holidays = add_holidays
        self.add_events = add_events
        self.country_isocode = country_isocode
        self.batch_size = batch_size

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]
        T, C = x.shape

        # One unique_id per channel; let pandas pick a start date — the API
        # only uses ``ds`` for frequency alignment, not for absolute time.
        index = pd.date_range("2000-01-01", periods=T, freq=_to_pandas_freq(self.freq))
        frames = []
        for c in range(C):
            frames.append(
                pd.DataFrame(
                    {
                        "unique_id": f"c{c}",
                        "ds": index,
                        "target": x[:, c],
                    }
                )
            )
        train_df = pd.concat(frames, ignore_index=True)

        forecast_df = self.client.forecast(
            train_df,
            model=self.model,
            horizon=self.prediction_length,
            freq=self.freq,
            quantiles=self.quantiles,
            context=self.context,
            add_holidays=self.add_holidays,
            add_events=self.add_events,
            country_isocode=self.country_isocode,
            batch_size=self.batch_size,
        )

        # SDK names the median column "<model>_q0.5" and the mean column "<model>".
        # Prefer the median to match the convention used by the Chronos solver.
        value_col = f"{self.model}_q0.5"
        if value_col not in forecast_df.columns:
            value_col = self.model
        if value_col not in forecast_df.columns:
            raise ValueError(
                f"TFC API response missing expected columns; got {list(forecast_df.columns)!r}"
            )

        preds = np.empty((self.prediction_length, C), dtype=np.float32)
        for c in range(C):
            channel = forecast_df.loc[forecast_df["unique_id"] == f"c{c}"]
            channel = channel.sort_values("ds").head(self.prediction_length)
            preds[:, c] = channel[value_col].to_numpy(dtype=np.float32)
        return preds


class Solver(BaseSolver):
    """TFC hosted-API solver.

    Parameters
    ----------
    model : str
        Model id served by the TFC API — e.g. ``"chronos-2"``,
        ``"timesfm-2p5"``, ``"tfc-global"``, ``"moirai-2"``, or a custom
        ``"T0-<run>-step-<n>"`` checkpoint.
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
                f"Unknown TFC model '{self.model}'. Known SDK models: {known}. "
                f"For custom T0 checkpoints, add the value to TFCModels first."
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
