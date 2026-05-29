"""GIFT-Eval forecasting benchmark dataset (Salesforce/GiftEval on HF).

The HF repo organizes data per-dataset under top-level directories
(``m4_weekly``, ``etth1``, ``solar``, ...). Each directory holds a
single Arrow file with the test-set series.

Each entry exposes ``item_id``, ``start``, ``freq``, and ``target``
(a flat list of floats). For multivariate datasets, ``target`` is still
serialized as a flat list — GIFT-Eval handles those via separate file
layouts we don't unpack here; the MVP supports univariate only.

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

    def get_data(self):
        from datasets import Dataset as HFDataset
        from huggingface_hub import hf_hub_download, list_repo_files

        # Locate the Arrow file inside the requested subdirectory.
        files = list_repo_files(
            "Salesforce/GiftEval", repo_type="dataset"
        )
        arrow_files = [
            f for f in files
            if f.startswith(f"{self.dataset_name}/")
            and f.endswith(".arrow")
        ]
        if not arrow_files:
            raise ValueError(
                f"No Arrow file found for GIFT-Eval dataset "
                f"{self.dataset_name!r}. Available top-level dirs: "
                f"{sorted({f.split('/')[0] for f in files if '/' in f})}"
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

        # Build (T, C) series. Univariate only in the MVP.
        series_list = []
        for r in rows:
            values = np.asarray(r["target"], dtype=np.float32)
            if values.ndim != 1:
                # Skip multivariate entries until we add explicit handling.
                continue
            series_list.append(values.reshape(-1, 1))

        if not series_list:
            raise ValueError(
                f"All entries in GIFT-Eval dataset {self.dataset_name!r} "
                "were skipped (multivariate not yet supported)."
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
