"""Monash Time Series Forecasting Archive dataset.

Uses ``aeon.datasets.load_forecasting`` to download directly from
https://forecastingdata.org/ and produces rolling-window evaluation splits.
The full historical context is passed as each test entry so models with long
context windows can use it.

Dataset names follow the aeon convention: ``<name>_dataset``
(e.g. ``"m1_yearly_dataset"``, ``"m4_weekly_dataset"``).
A full list is at https://www.timeseriesclassification.com/dataset.php
or via ``aeon.datasets.forecasting_dataset_names()``.

Data contract output
--------------------
X_train         : List[np.ndarray (T_i, C)]      training portions of each series
y_train         : List[np.ndarray (H, C)]        next-H targets aligned with X_train
X_test          : List[np.ndarray (T_i, C)]      full series — model uses
                                                  ``x[:cutoff]`` as history
cutoff_indexes  : List[List[int]]                jagged: per-series cutoff
                                                  positions in X_test
y_test          : List[np.ndarray (n_cutoffs, H, C)]
                                                  ground-truth windows
covariates      : dict                           {static_covars, hist_covars,
                                                  future_covars} — all empty for
                                                  Monash today
task            : "forecasting"
metrics         : ["mae", "mse", "mase", "smape"]
prediction_length : int
freq            : str  (e.g. "Y", "M", "D")
seasonality     : int  (seasonal period used for MASE)
"""

import numpy as np
from benchopt import BaseDataset

from aeon.datasets import load_forecasting
from benchmark_utils.covariates import Covariates
from benchmark_utils.windowing import make_forecasting_splits


# Map aeon frequency strings → pandas-style freq codes and MASE seasonality
_FREQ_MAP = {
    "yearly": ("Y", 1),
    "quarterly": ("Q", 4),
    "monthly": ("M", 12),
    "weekly": ("W", 52),
    "daily": ("D", 7),
    "hourly": ("H", 24),
    "minutely": ("T", 1440),
    "seconds": ("S", 1),
}

_DEFAULT_HORIZON = {
    "Y": 6, "Q": 8, "M": 12, "W": 13, "D": 14, "H": 24, "T": 60,
}


class Dataset(BaseDataset):
    """Monash forecasting dataset (loaded via aeon).

    Parameters
    ----------
    dataset_name : str
        aeon dataset name, e.g. ``"m1_yearly_dataset"``.
    prediction_length : int or None
        Override the dataset's default forecast horizon.
    n_windows : int
        Number of rolling evaluation windows per series.
    debug : bool
        If True, keep only the first 5 series for fast iteration.
    """

    name = "Monash"

    # aeon is already a requirement of the objective
    requirements = []

    parameters = {
        "dataset_name": ["m1_yearly_dataset"],
        "prediction_length": [None],
        "n_windows": [1],
        "debug": [False],
    }

    def get_data(self):

        df, meta = load_forecasting(self.dataset_name, return_metadata=True)
        # df columns: series_name, start_timestamp, series_value
        # meta keys:  frequency, forecast_horizon,
        #             contain_missing_values, contain_equal_length

        aeon_freq = meta.get("frequency", "yearly")
        freq, seasonality = _FREQ_MAP.get(aeon_freq, ("D", 1))

        pred_len = self.prediction_length
        if pred_len is None:
            pred_len = int(
                meta.get("forecast_horizon")
                or _DEFAULT_HORIZON.get(freq, 10)
            )

        series_list = []
        rows = df.iterrows() if not self.debug else list(df.iterrows())[:5]
        for _, row in rows:
            values = np.asarray(row["series_value"], dtype=np.float32)
            series_list.append(values.reshape(-1, 1))  # (T, 1) univariate

        if not series_list:
            raise ValueError(
                f"No series found for dataset {self.dataset_name!r}."
            )

        # Training portion: everything except the last test windows
        test_len = pred_len * self.n_windows
        X_train, y_train_list, full_series = [], [], []
        for ts in series_list:
            if ts.shape[0] < pred_len + 1:
                continue
            train_end = max(1, ts.shape[0] - test_len)
            X_train.append(ts[:train_end])
            y_train_list.append(ts[train_end: train_end + pred_len])
            full_series.append(ts)

        if not full_series:
            raise ValueError(
                "All series are shorter than prediction_length."
            )

        n_windows = 1 if self.debug else self.n_windows
        X_test, cutoff_indexes, y_test = make_forecasting_splits(
            full_series,
            prediction_length=pred_len,
            n_windows=n_windows,
        )

        return dict(
            X_train=X_train,
            y_train=y_train_list,
            X_test=X_test,
            y_test=y_test,
            cutoff_indexes=cutoff_indexes,
            covariates=Covariates(),
            task="forecasting",
            metrics=["mae", "mse", "mase", "smape"],
            prediction_length=pred_len,
            freq=freq,
            seasonality=seasonality,
        )
