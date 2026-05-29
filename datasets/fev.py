"""AutoGluon fev_datasets forecasting benchmark
(huggingface.co/datasets/autogluon/fev_datasets).

The HF repo organizes data either:
  - per-freq: ``<dataset>/<freq>/train-*.parquet``
    (e.g. ``ETT/1H``, ``LOOP_SEATTLE/5T``)
  - flat: ``<dataset>/train-*.parquet``
    (e.g. ``australian_tourism``)
  - or with an arbitrary subdir that is NOT a freq (e.g. ``boomlet/<N>``
    where ``<N>`` is a series id, not a frequency).

We accept the directory path directly as ``dataset_name`` (e.g.
``"ETT/1H"``, ``"australian_tourism"``) and infer the actual freq from
each series' ``timestamp`` column rather than parsing the path.

Each parquet row is one series; columns vary:
  - Always: ``id``, ``timestamp``
  - Univariate: a ``target`` column (list of floats)
  - Multivariate (e.g. ``ETT``): no ``target`` column — each channel is
    its own column (``HUFL``, ..., ``OT``). Channel columns are stacked
    on the last axis to form ``(T, C)``.

Rolling-window splits match :mod:`datasets.monash`. The default
``prediction_length`` is the freq-based heuristic from
:func:`benchmark_utils.constants.from_pandas`; FEV does not publish a
per-dataset horizon spec, so we don't try to mirror one. Pass
``prediction_length=N`` explicitly to override.
"""

import numpy as np
import pandas as pd
from benchopt import BaseDataset

from benchmark_utils.covariates import Covariates
from benchmark_utils.constants import from_pandas
from benchmark_utils.windowing import make_forecasting_splits


_METADATA_COLS = ("id", "timestamp")


# Canonical list of FEV evaluation configs — directory paths inside
# https://huggingface.co/datasets/autogluon/fev_datasets that contain at
# least one ``train-*.parquet`` file. Surfaced via
# ``get_parameter_choices`` so that ``dataset_name=all`` and ``benchopt
# info -v`` work.
FEV_DATASETS: tuple[str, ...] = (
    "ETT/15T", "ETT/1D", "ETT/1H", "ETT/1W",
    "LOOP_SEATTLE/1D", "LOOP_SEATTLE/1H", "LOOP_SEATTLE/5T",
    "M_DENSE/1D", "M_DENSE/1H",
    "SZ_TAXI/15T", "SZ_TAXI/1H",
    "australian_tourism",
    "bizitobs_l2c/1H", "bizitobs_l2c/5T",
    "boomlet/1062", "boomlet/1209", "boomlet/1225", "boomlet/1230",
    "boomlet/1282", "boomlet/1487", "boomlet/1631", "boomlet/1676",
    "boomlet/1855", "boomlet/1975", "boomlet/2187",
    "boomlet/285", "boomlet/619", "boomlet/772", "boomlet/963",
    "ecdc_ili",
    "entsoe/15T", "entsoe/1H", "entsoe/30T",
    "epf_be", "epf_de", "epf_fr", "epf_np", "epf_pjm",
    "ercot/1D", "ercot/1H", "ercot/1M", "ercot/1W",
    "favorita_stores/1D", "favorita_stores/1M", "favorita_stores/1W",
    "favorita_transactions/1D", "favorita_transactions/1M",
    "favorita_transactions/1W",
    "fred_md_2025", "fred_qd_2025",
    "gvar", "hermes",
    "hierarchical_sales/1D", "hierarchical_sales/1W",
    "hospital",
    "hospital_admissions/1D", "hospital_admissions/1W",
    "jena_weather/10T", "jena_weather/1D", "jena_weather/1H",
    "kdd_cup_2022/10T", "kdd_cup_2022/1D", "kdd_cup_2022/30T",
    "m5/1D", "m5/1M", "m5/1W",
    "proenfo_bull", "proenfo_cockatoo",
    "proenfo_gfc12", "proenfo_gfc14", "proenfo_gfc17",
    "proenfo_hog", "proenfo_pdb",
    "redset/15T", "redset/1H", "redset/5T",
    "restaurant",
    "rohlik_orders/1D", "rohlik_orders/1W",
    "rohlik_sales/1D", "rohlik_sales/1W",
    "rossmann/1D", "rossmann/1W",
    "solar/1D", "solar/1W",
    "solar_with_weather/15T", "solar_with_weather/1H",
    "uci_air_quality/1D", "uci_air_quality/1H",
    "uk_covid_nation/1D", "uk_covid_nation/1W",
    "uk_covid_utla/1D", "uk_covid_utla/1W",
    "us_consumption/1M", "us_consumption/1Q", "us_consumption/1Y",
    "walmart",
    "world_co2_emissions", "world_life_expectancy", "world_tourism",
)


def _infer_freq(timestamps) -> str:
    """Best-effort freq inference from a series' timestamp column.

    Falls back to ``"D"`` when pandas cannot infer. Uses the first 5
    points to keep the check cheap on long series.
    """
    try:
        idx = pd.DatetimeIndex(timestamps[:5])
        return pd.infer_freq(idx) or "D"
    except Exception:
        return "D"


class Dataset(BaseDataset):
    """AutoGluon fev forecasting dataset.

    Parameters
    ----------
    dataset_name : str
        Directory path inside the HF repo. Per-freq paths look like
        ``"ETT/1H"`` / ``"LOOP_SEATTLE/5T"``; flat paths like
        ``"australian_tourism"`` / ``"hospital"``. See ``FEV_DATASETS``
        for the full list (also discoverable via ``benchopt info -v``).
    prediction_length : int or None
        Explicit override. ``None`` → resolved from the inferred freq
        via :func:`benchmark_utils.constants.from_pandas` (same heuristic
        used by Monash). FEV does not publish its own per-dataset
        horizon matrix, so we don't try to align with a leaderboard
        spec here.
    n_windows : int
        Number of rolling evaluation windows per series.
    max_series : int or None
        Optional cap on the number of series.
    debug : bool
        If True, keep only the first 5 series.
    """

    name = "FEV"

    requirements = ["pip::pyarrow", "pip::huggingface-hub"]

    parameters = {
        "dataset_name": ["LOOP_SEATTLE/1H"],
        "prediction_length": [None],
        "n_windows": [1],
        "max_series": [None],
        "debug": [False],
    }

    # Cache prepare() by dataset_name only — the other knobs shape the
    # in-memory view, not the downloaded files.
    prepare_cache_ignore = (
        "prediction_length", "n_windows", "max_series", "debug",
    )

    @classmethod
    def get_all_parameter_values(cls, name):
        if name == "dataset_name":
            return list(FEV_DATASETS)
        return None

    def prepare(self):
        """Pre-download parquet shards for this config into HF's cache."""
        self._snapshot()

    def _snapshot(self) -> "list[str]":
        """Snapshot-download parquet files for this dataset_name and
        return their local paths. Idempotent."""
        from huggingface_hub import snapshot_download
        from pathlib import Path

        local_root = snapshot_download(
            "autogluon/fev_datasets",
            repo_type="dataset",
            allow_patterns=f"{self.dataset_name}/*.parquet",
        )
        return sorted(
            str(p) for p in (Path(local_root) / self.dataset_name).glob("*.parquet")
        )

    def get_data(self):
        parquet_files = self._snapshot()
        if not parquet_files:
            raise ValueError(
                f"No parquet found at {self.dataset_name!r} in "
                "autogluon/fev_datasets. Valid choices are in FEV_DATASETS."
            )

        df = pd.concat(
            [pd.read_parquet(f) for f in parquet_files],
            ignore_index=True,
        )

        if self.debug:
            df = df.head(5)
        elif self.max_series is not None:
            df = df.head(int(self.max_series))

        if df.empty:
            raise ValueError(f"{self.dataset_name!r} contained 0 series.")

        # Channel cols = non-metadata columns whose entries are numeric
        # array-likes. Some FEV datasets carry extra scalar/string fields
        # (``type``, ``Security``) or arrays of strings (holiday names in
        # ``favorita_stores``, etc.). We treat covariates as out of scope
        # for the MVP.
        def _is_numeric_array_col(c):
            v = df.iloc[0][c]
            if not hasattr(v, "__len__") or isinstance(v, (str, bytes)):
                return False
            if len(v) == 0:
                return False
            return isinstance(v[0], (int, float, np.integer, np.floating))

        channel_cols = [
            c for c in df.columns
            if c not in _METADATA_COLS and _is_numeric_array_col(c)
        ]
        if not channel_cols:
            raise ValueError(
                f"{self.dataset_name!r} has no channel columns "
                f"(only {_METADATA_COLS} present)."
            )

        # Infer freq from the first series' timestamps — same for the
        # whole config (FEV groups by freq at the directory level for
        # nested configs, and flat configs are single-freq).
        inferred_freq = _infer_freq(df.iloc[0]["timestamp"])
        canonical_freq, seasonality, default_h = from_pandas(inferred_freq)

        pred_len = self.prediction_length
        if pred_len is None:
            pred_len = int(default_h)

        # Build (T, C) series. Each row's per-channel array has the same
        # length (T_i); stack on the last axis.
        series_list = []
        for _, row in df.iterrows():
            channels = [np.asarray(row[c], dtype=np.float32) for c in channel_cols]
            T = channels[0].shape[0]
            if any(ch.shape[0] != T for ch in channels):
                continue
            series_list.append(np.stack(channels, axis=-1))

        if not series_list:
            raise ValueError("All series were skipped (inconsistent channel lengths).")

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
            raise ValueError("All series are shorter than prediction_length.")

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
            freq=canonical_freq,
            seasonality=seasonality,
        )
