"""GIFT-Eval forecasting benchmark dataset (Salesforce/GiftEval on HF).

The HF repo organizes data per-dataset under top-level directories
(``m4_weekly``, ``etth1``, ``solar``, ...). Each directory holds a
single Arrow file with the test-set series.

Each entry exposes ``item_id``, ``start``, ``freq``, and ``target``.
``target`` is a flat ``List[float]`` for univariate configs and a
``List[List[float]]`` of shape ``(C, T)`` for multivariate ones (e.g.
``bitbrains_*``, ``electricity/*``, ``ett1/*``, ``ett2/*``,
``jena_weather/*``, ``solar/*``). Both shapes are handled — multivariate
entries are transposed to the ``(T, C)`` repo contract.

Cutoffs and windows follow the Monash recipe (we don't comply with
GIFT-Eval's prescribed test cutoff — same rolling-window logic via
:func:`benchmark_utils.windowing.make_forecasting_splits`).

Data contract output mirrors :mod:`datasets.monash`.
"""

import numpy as np
from benchopt import BaseDataset

from benchmark_utils.covariates import Covariates
from benchmark_utils.constants import (
    from_pandas,
    gift_eval_prediction_length,
)
from benchmark_utils.windowing import make_forecasting_splits


# Canonical list of GIFT-Eval evaluation configs. Each entry is the
# arrow-containing directory path inside the HF repo. Flat datasets are
# bare names (``m4_weekly``); datasets that ship multiple frequencies
# are encoded as ``<name>/<freq>`` (e.g. ``LOOP_SEATTLE/H``,
# ``LOOP_SEATTLE/D`` — these are genuinely distinct evaluations).
# Surfaced via ``get_parameter_choices`` so that ``dataset_name=all``
# and ``benchopt info -v`` work.
GIFTEVAL_DATASETS: tuple[str, ...] = (
    "LOOP_SEATTLE/5T", "LOOP_SEATTLE/D", "LOOP_SEATTLE/H",
    "M_DENSE/D", "M_DENSE/H",
    "SZ_TAXI/15T", "SZ_TAXI/H",
    "bitbrains_fast_storage/5T", "bitbrains_fast_storage/H",
    "bitbrains_rnd/5T", "bitbrains_rnd/H",
    "bizitobs_application",
    "bizitobs_l2c/5T", "bizitobs_l2c/H",
    "bizitobs_service",
    "car_parts_with_missing", "covid_deaths",
    "electricity/15T", "electricity/D", "electricity/H", "electricity/W",
    "ett1/15T", "ett1/D", "ett1/H", "ett1/W",
    "ett2/15T", "ett2/D", "ett2/H", "ett2/W",
    "hierarchical_sales/D", "hierarchical_sales/W",
    "hospital",
    "jena_weather",
    "jena_weather/10T", "jena_weather/D", "jena_weather/H",
    "kdd_cup_2018_with_missing/D", "kdd_cup_2018_with_missing/H",
    "m4_daily", "m4_hourly", "m4_monthly", "m4_quarterly",
    "m4_weekly", "m4_yearly",
    "restaurant",
    "saugeenday/D", "saugeenday/M", "saugeenday/W",
    "solar/10T", "solar/D", "solar/H", "solar/W",
    "temperature_rain_with_missing",
    "us_births/D", "us_births/M", "us_births/W",
)

GIFTEVAL_TERMS: tuple[str, ...] = ("short", "medium", "long")


class Dataset(BaseDataset):
    """GIFT-Eval forecasting dataset (loaded from HF Salesforce/GiftEval).

    Parameters
    ----------
    dataset_name : str
        Subdirectory name on the HF repo (e.g. ``"m4_weekly"``, ``"ett1"``,
        ``"solar"``). See https://huggingface.co/datasets/Salesforce/GiftEval
        for the full list.
    term : str
        GIFT-Eval forecast term — ``"short"`` (×1), ``"medium"`` (×10), or
        ``"long"`` (×15). Selects the prediction length via the canonical
        per-freq base, matching the GIFT-Eval leaderboard convention.
        Ignored when ``prediction_length`` is set explicitly.
    prediction_length : int or None
        Explicit override. ``None`` → resolved from (freq, term).
    n_windows : int
        Number of rolling evaluation windows per series.
    max_series : int or None
        Optional cap on the number of series — useful for very large
        configs (e.g. ``solar``). ``None`` = no cap.
    debug : bool
        If True, keep only the first 5 series for fast iteration.
    """

    name = "GiftEval"

    requirements = ["pip::datasets", "pip::huggingface-hub"]

    parameters = {
        "dataset_name": ["m4_weekly"],
        "term": ["short"],
        "prediction_length": [None],
        "n_windows": [1],
        "max_series": [None],
        "debug": [False],
    }

    @classmethod
    def get_all_parameter_values(cls, name):
        if name == "dataset_name":
            return list(GIFTEVAL_DATASETS)
        if name == "term":
            return list(GIFTEVAL_TERMS)
        return None

    def get_data(self):
        from datasets import Dataset as HFDataset
        from huggingface_hub import hf_hub_download, list_repo_files

        # Locate the Arrow file inside the requested directory. Match the
        # exact directory (no nested descent) — for datasets like
        # ``LOOP_SEATTLE`` that ship multiple freq subdirs, the user must
        # pick one (``LOOP_SEATTLE/H``, ``LOOP_SEATTLE/D``, ...), and
        # those are genuinely separate evaluation configs.
        files = list_repo_files(
            "Salesforce/GiftEval", repo_type="dataset"
        )
        prefix = f"{self.dataset_name}/"
        arrow_files = [
            f for f in files
            if f.startswith(prefix)
            and f.endswith(".arrow")
            and "/" not in f[len(prefix):]
        ]
        if not arrow_files:
            raise ValueError(
                f"No Arrow file found for GIFT-Eval dataset "
                f"{self.dataset_name!r}. Valid choices are in "
                f"GIFTEVAL_DATASETS."
            )

        # Download + load each shard; concatenate.
        rows = []
        for f in sorted(arrow_files):
            local = hf_hub_download(
                "Salesforce/GiftEval", filename=f, repo_type="dataset",
            )
            shard = HFDataset.from_file(local)
            rows.extend(shard)

        if self.debug:
            rows = rows[:5]
        elif self.max_series is not None:
            rows = rows[: int(self.max_series)]

        if not rows:
            raise ValueError(
                f"GIFT-Eval dataset {self.dataset_name!r} returned 0 series."
            )

        # Frequency / seasonality — take from the first entry (every series
        # in a GIFT-Eval subset shares the same freq).
        pandas_freq = rows[0].get("freq") or "D"
        freq, seasonality, _ = from_pandas(pandas_freq)

        pred_len = self.prediction_length
        if pred_len is None:
            pred_len = gift_eval_prediction_length(pandas_freq, self.term)

        # Build (T, C) series. Univariate entries arrive as flat
        # ``List[float]`` (ndim=1); multivariate entries arrive as
        # ``List[List[float]]`` of shape ``(C, T)``.
        series_list = []
        for r in rows:
            values = np.asarray(r["target"], dtype=np.float32)
            if values.ndim == 1:
                series = values.reshape(-1, 1)        # (T, 1)
            elif values.ndim == 2:
                series = values.T                       # (C, T) → (T, C)
            else:
                continue
            series_list.append(series)

        if not series_list:
            raise ValueError(
                f"All entries in GIFT-Eval dataset {self.dataset_name!r} "
                "had unsupported target shapes."
            )

        # Training portion: everything except the last test windows.
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
            covariates=Covariates(),  # GIFT-Eval HF schema has no covariates
            task="forecasting",
            metrics=["mae", "mse", "mase", "smape"],
            prediction_length=pred_len,
            freq=freq,
            seasonality=seasonality,
        )
