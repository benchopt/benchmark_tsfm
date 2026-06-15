"""Enedis French electricity-consumption forecasting dataset (with holidays).

Wraps ``theforecastingcompany/enedis-with-holidays`` on the Hugging Face
Hub: five years (2020-07-18 → 2025-07-17) of total French electricity
consumption (``consommation_totale``) published by Enedis, at three
frequencies — ``30min``, ``6h`` and ``D``. Each frequency carries
holiday-anchored backtest windows and the matching covariates:

- past (history-only):   ``temperature_reelle_lissee``
- future (known-ahead):  ``is_france_holiday``, ``temperature_normale_lissee``

The Hub stores one row per frequency in a GluonTS-style layout. Each row
holds the full target series ``(1, T)`` plus nine forecast windows
anchored on three 2024 French holidays (Labour Day, Armistice,
Christmas/New Year), encoded as ``window_fcd_idxs`` (index of the last
*observed* step) and ``window_horizons`` (steps to forecast).

Single-horizon contract
-----------------------
The objective stacks every window's target into one array, so a single
``Dataset`` instance must expose windows that share one horizon. We
therefore select the windows whose ``window_horizons`` equals
``prediction_length``; the cutoff for a window is ``fcd_idx + 1`` (the
first forecast step). Sweep ``prediction_length`` to reach the other
holiday windows — :data:`_HORIZONS_BY_FREQ` lists the available values.

Data contract output
--------------------
X_train         : List[np.ndarray (T0, 1)]    history before the first cutoff
y_train         : List[np.ndarray (H, 1)]     next-H targets after that cutoff
X_test          : List[np.ndarray (T, 1)]     full series — model uses
                                              ``x[:cutoff]`` as history
cutoff_indexes  : List[List[int]]             per-series holiday cutoffs
y_test          : List[np.ndarray (n_cutoffs, H, 1)]
covariates      : Covariates                  hist_covars (T, 1) +
                                              future_covars (T, 2) per series
task            : "forecasting"
metrics         : ["mae", "mse", "mase", "smape"]
prediction_length : int
freq            : str  ("30min", "6h", "D")
seasonality     : int  (seasonal period used for MASE)
"""

import numpy as np
import pandas as pd
from benchopt import BaseDataset

# Requirement checks
import fsspec  # noqa: F401
from huggingface_hub import hf_hub_download  # noqa: F401

from benchmark_utils.covariates import Covariates

_HF_PARQUET = (
    "hf://datasets/theforecastingcompany/enedis-with-holidays/"
    "data/train-00000-of-00001.parquet"
)

_ITEM_ID = {
    "30min": "enedis_bilan_30min",
    "6h": "enedis_bilan_6h",
    "D": "enedis_bilan_D",
}

# MASE seasonal period (one full day) per frequency.
_SEASONALITY = {"30min": 48, "6h": 4, "D": 7}

# Forecast-window horizons available per frequency (steps). Sweep
# ``prediction_length`` over these to reach every holiday window.
_HORIZONS_BY_FREQ = {
    "30min": [240, 480, 624, 864, 960, 1344],
    "6h": [20, 40, 52, 72, 80, 112],
    "D": [5, 10, 13, 18, 20, 28],
}

# Default horizon per frequency — the "10 days ahead" window, comparable
# across frequencies (Labour Day + Armistice ⇒ two cutoffs each).
_DEFAULT_HORIZON = {"30min": 480, "6h": 40, "D": 10}


def _stack_channels(seq) -> np.ndarray:
    """Stack a GluonTS ``Sequence(Sequence(float32))`` cell into ``(T, C)``."""
    return np.stack([np.asarray(channel, dtype=np.float32) for channel in seq], axis=-1)


class Dataset(BaseDataset):
    """Enedis electricity forecasting dataset with holiday windows.

    Parameters
    ----------
    freq : str
        Which frequency series to load — one of ``"30min"``, ``"6h"``,
        ``"D"``.
    prediction_length : int or None
        Forecast horizon in steps. Only windows whose horizon equals this
        value are used. ``None`` selects the default 10-day-ahead horizon
        for the frequency (see :data:`_DEFAULT_HORIZON`).
    debug : bool
        If True, keep only the first matching window for fast iteration.
    """

    name = "Enedis"

    requirements = ["pip::huggingface_hub", "fsspec"]

    parameters = {
        "freq": ["D", "6h", "30min"],
        "prediction_length": [None],
        "debug": [False],
    }

    def get_data(self):
        if self.freq not in _ITEM_ID:
            raise ValueError(
                f"Unknown freq {self.freq!r}; expected one of {sorted(_ITEM_ID)}."
            )

        df = pd.read_parquet(_HF_PARQUET)
        rows = df.loc[df["item_id"] == _ITEM_ID[self.freq]]
        if rows.empty:
            raise ValueError(
                f"No row for item_id {_ITEM_ID[self.freq]!r} in the Enedis "
                f"parquet; got {df['item_id'].tolist()!r}."
            )
        row = rows.iloc[0]

        target = _stack_channels(row["target"])  # (T, 1)
        hist_covar = _stack_channels(row["past_feat_dynamic_real"])  # (T, 1)
        future_covar = _stack_channels(row["feat_dynamic_real"])  # (T, 2)
        T = target.shape[0]

        pred_len = self.prediction_length
        if pred_len is None:
            pred_len = _DEFAULT_HORIZON[self.freq]

        fcd_idxs = [int(i) for i in row["window_fcd_idxs"]]
        horizons = [int(h) for h in row["window_horizons"]]
        # cutoff = first forecast step = last-observed-step index + 1.
        cutoffs = [
            fcd + 1
            for fcd, h in zip(fcd_idxs, horizons)
            if h == pred_len and fcd + 1 + h <= T
        ]
        if not cutoffs:
            raise ValueError(
                f"No window with horizon {pred_len} for freq {self.freq!r}. "
                f"Available horizons: {_HORIZONS_BY_FREQ[self.freq]}."
            )
        if self.debug:
            cutoffs = cutoffs[:1]

        y_windows = np.stack(
            [target[c : c + pred_len] for c in cutoffs], axis=0
        )  # (n_cutoffs, H, 1)

        first_cut = min(cutoffs)
        X_train = [target[:first_cut]]
        y_train = [target[first_cut : first_cut + pred_len]]

        covariates = Covariates(
            static_covars=[],
            hist_covars=[hist_covar],
            future_covars=[future_covar],
        )

        return dict(
            X_train=X_train,
            y_train=y_train,
            X_test=[target],
            y_test=[y_windows],
            cutoff_indexes=[cutoffs],
            covariates=covariates,
            task="forecasting",
            metrics=["mae", "mse", "mase", "smape"],
            prediction_length=pred_len,
            freq=self.freq,
            seasonality=_SEASONALITY[self.freq],
        )
