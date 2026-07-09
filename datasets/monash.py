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
metrics         : ["mae", "mse", "rmse", "mase", "smape",
                   "crps", "wql", "mcis", "pinball", "skill_score_ratio"]
prediction_length : int
freq            : str  (e.g. "Y", "M", "D")
seasonality     : int  (seasonal period used for MASE)
"""

import numpy as np
from aeon.datasets import load_forecasting
from benchopt import BaseDataset

from benchmark_utils.covariates import Covariates
from benchmark_utils.forecasting_constants import FORECASTING_METRICS, from_aeon
from benchmark_utils.windowing import build_forecasting_data


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

    # Only dataset_name decides what aeon downloads; the other knobs
    # affect the in-memory split, not the file on disk.
    prepare_cache_ignore = ("prediction_length", "n_windows", "debug")

    def prepare(self):
        """Warm aeon's local cache for this dataset (download if missing).

        aeon writes the ``.tsf`` to
        ``~/.aeon/datasets/local_data/<name>/<name>.tsf`` on first use;
        we call it once and discard the parsed result so the cache layer
        in :func:`load_forecasting` handles the actual download.
        """
        load_forecasting(self.dataset_name, return_metadata=False)

    def get_data(self):
        df, meta = load_forecasting(self.dataset_name, return_metadata=True)
        # df columns: series_name, start_timestamp, series_value
        # meta keys:  frequency, forecast_horizon,
        #             contain_missing_values, contain_equal_length

        aeon_freq = meta.get("frequency", "yearly")
        freq, seasonality, default_h = from_aeon(aeon_freq)

        pred_len = self.prediction_length
        if pred_len is None:
            pred_len = int(meta.get("forecast_horizon") or default_h)

        series_list = []
        rows = df.iterrows() if not self.debug else list(df.iterrows())[:5]
        for _, row in rows:
            values = np.asarray(row["series_value"], dtype=np.float32)
            series_list.append(values.reshape(-1, 1))  # (T, 1) univariate

        if not series_list:
            raise ValueError(f"No series found for dataset {self.dataset_name!r}.")

        return dict(
            **build_forecasting_data(
                series_list,
                prediction_length=pred_len,
                n_windows=self.n_windows,
                debug=self.debug,
            ),
            covariates=Covariates(),
            task="forecasting",
            metrics=list(FORECASTING_METRICS),
            prediction_length=pred_len,
            freq=freq,
            seasonality=seasonality,
        )
