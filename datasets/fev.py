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

``dataset_name`` is a fev-bench task (``benchmarks/fev_bench/tasks.yaml`` in
github.com/autogluon/fev). Its target columns, horizon, number of windows,
minimum context and seasonality follow the task. The rolling windows match
:mod:`datasets.monash`. Unlike fev-bench, the windows of the 4 rohlik tasks
end at the end of each series instead of starting at their ``initial_cutoff``
date, and the metrics are the benchmark's.

The task's covariate columns are exposed as :class:`Covariates`:
``past_dynamic_columns`` as ``hist_covars``, ``known_dynamic_columns`` as
``future_covars`` and ``static_columns`` as ``static_covars``. Dynamic arrays
span the whole series, and categorical values are encoded as integer codes.
"""

import functools
import re
import warnings

import numpy as np
import pooch
import yaml
import pandas as pd
from benchopt import BaseDataset, config

from benchmark_utils.covariates import Covariates
from benchmark_utils.download_hf import snapshot_hf_files
from benchmark_utils.forecasting_constants import (
    FORECASTING_METRICS,
    autogluon_seasonality,
    from_pandas,
)
from benchmark_utils.windowing import MIN_CONTEXT, build_forecasting_data


_METADATA_COLS = ("id", "timestamp")

# fev-bench task definitions, pinned to a commit of autogluon/fev.
_TASKS_URL = (
    "https://raw.githubusercontent.com/autogluon/fev/"
    "844509e5211292620fbd9d5f24695f1d27da767a/benchmarks/fev_bench/tasks.yaml"
)
_TASKS_SHA256 = "c7160f61a5e1ded66a3954ef1c514d55d13be18534b34fca817356312a6520a9"


# Underscore of a fev-bench dataset_config that is a subdirectory of
# autogluon/fev_datasets: before a frequency (ETT_1H -> ETT/1H) or a boomlet
# series id (boomlet_1062 -> boomlet/1062).
_HF_SUBDIR = re.compile(r"_(?=\d+[A-Za-z]+$)|(?<=^boomlet)_")

# Context kept by fev-bench's statsforecast baselines.
BASELINE_CONTEXT_LENGTH = 1000


def _baseline_seasonality(seasonality: int, freq: str) -> int:
    """Season of fev-bench's Seasonal Naive: the larger of the task's and
    the frequency's, or 1 when it exceeds the context the baseline sees."""
    season = max(seasonality, autogluon_seasonality(freq))
    return season if season <= BASELINE_CONTEXT_LENGTH else 1


def _infer_freq(timestamps) -> str:
    """Best-effort freq inference from a series' timestamp column.

    Tries ``pd.infer_freq`` on the first points, then the most common
    consecutive delta (robust to isolated gaps). Warns and falls back to
    ``"D"`` when nothing can be inferred.
    """
    try:
        idx = pd.DatetimeIndex(timestamps[:20])
        freq = pd.infer_freq(idx[:5]) if len(idx) >= 3 else None
        if freq:
            return freq
        deltas = pd.Series(idx).diff().dropna()
        if len(deltas):
            step = deltas.mode().iloc[0]
            return pd.tseries.frequencies.to_offset(step).freqstr
    except Exception:
        pass
    warnings.warn(
        "Could not infer the sampling frequency from timestamps; "
        "defaulting to daily ('D')."
    )
    return "D"


def _is_numeric_array_col(df, c) -> bool:
    """True if column ``c`` holds non-empty numeric array-likes.

    Decided from the first informative entry among the first 5 rows, so
    a null or empty first row does not drop a valid channel.
    """
    for v in df[c].head(5):
        if v is None or isinstance(v, (str, bytes)) or not hasattr(v, "__len__"):
            continue
        if len(v) == 0:
            continue
        return isinstance(v[0], (int, float, np.integer, np.floating))
    return False


@functools.cache
def _fev_bench_tasks() -> "dict[str, dict]":
    """Task name → fev-bench task, from the pinned (disk-cached) tasks.yaml."""
    path = pooch.retrieve(
        _TASKS_URL, _TASKS_SHA256, path=config.get_data_path(key="fev")
    )
    with open(path) as f:
        tasks = yaml.safe_load(f)["tasks"]
    return {t.get("task_name", t["dataset_config"]): t for t in tasks}


def _select_channel_cols(df, target=None, covariate_cols=()) -> "list[str]":
    """Columns to stack as forecast channels.

    The task's ``target`` (one column or a list) gives the channels. Without
    it, an explicit ``target`` column is the sole target, as other numeric
    array columns are exogenous covariates (e.g. the epf_* load forecasts).
    Without either (e.g. ETT), every numeric array column that is not one of
    the task's ``covariate_cols`` is a channel.
    """
    if target is not None:
        return [target] if isinstance(target, str) else list(target)
    if "target" in df.columns:
        return ["target"]
    return [
        c for c in df.columns
        if c not in (*_METADATA_COLS, *covariate_cols)
        and _is_numeric_array_col(df, c)
    ]


def _encode(values) -> np.ndarray:
    """Float values of a covariate column, categories as integer codes.

    Callers pass the values of all series at once, so a category gets the
    same code in every series.
    """
    values = pd.Series(values)
    try:
        return pd.to_numeric(values).to_numpy(np.float32)
    except (ValueError, TypeError):
        codes = pd.factorize(values)[0]  # -1 for missing values
        return np.where(codes < 0, np.nan, codes).astype(np.float32)


def _dynamic_covariates(df, cols) -> "list[np.ndarray] | None":
    """Per-series ``(T, len(cols))`` arrays of the dynamic columns ``cols``."""
    if not cols:
        return None
    lengths = df[cols[0]].map(len).to_numpy()
    split = np.cumsum(lengths)[:-1]
    columns = [np.split(_encode(np.concatenate(df[c].to_numpy())), split) for c in cols]
    return [np.stack(arrays, axis=-1) for arrays in zip(*columns)]


def _static_covariates(df, cols) -> "list[np.ndarray] | None":
    """Per-series ``(len(cols),)`` arrays of the static columns ``cols``."""
    if not cols:
        return None
    return list(np.stack([_encode(df[c]) for c in cols], axis=-1))



class Dataset(BaseDataset):
    """AutoGluon fev forecasting dataset.

    Parameters
    ----------
    dataset_name : str
        fev-bench task name, e.g. ``"LOOP_SEATTLE_1H"`` or
        ``"uk_covid_nation_1D/new"`` (``benchopt info -v`` lists them).
    prediction_length : int or None
        Explicit override. ``None`` → the task's horizon.
    n_windows : int or None
        Number of rolling evaluation windows per series. ``None`` → the
        task's number of windows.
    max_series : int or None
        Optional cap on the number of series.
    debug : bool
        If True, keep only the first 5 series.
    """

    name = "FEV"
    # Read by the solvers' skip without loading the data.
    task = "forecasting"

    requirements = ["pip::huggingface-hub", "pip::pooch", "pip::pyyaml"]

    parameters = {
        "dataset_name": ["LOOP_SEATTLE_1H"],
        "prediction_length": [None],
        "n_windows": [None],
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
            return list(_fev_bench_tasks())
        return None

    def prepare(self):
        """Pre-download parquet shards for this config into HF's cache."""
        self._snapshot()

    def _snapshot(self) -> "list[str]":
        """Snapshot-download parquet files for this dataset_name and
        return their local paths (cache-first, see ``snapshot_hf_files``)."""
        config = _fev_bench_tasks()[self.dataset_name]["dataset_config"]
        hf_path = _HF_SUBDIR.sub("/", config)
        return snapshot_hf_files("autogluon/fev_datasets", hf_path, "*.parquet")

    def get_data(self):
        task = _fev_bench_tasks()[self.dataset_name]
        parquet_files = self._snapshot()
        if not parquet_files:
            raise ValueError(
                f"No parquet found at {self.dataset_name!r} in "
                "autogluon/fev_datasets."
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

        covariate_cols = [
            c
            for key in ("past_dynamic_columns", "known_dynamic_columns")
            for c in task.get(key, [])
        ]
        channel_cols = _select_channel_cols(df, task.get("target"), covariate_cols)
        if not channel_cols:
            raise ValueError(
                f"{self.dataset_name!r} has no channel columns "
                f"(only {_METADATA_COLS} present)."
            )

        # Infer freq from the first series' timestamps — same for the
        # whole config (FEV groups by freq at the directory level for
        # nested configs, and flat configs are single-freq).
        inferred_freq = _infer_freq(df.iloc[0]["timestamp"])
        canonical_freq, _, _ = from_pandas(inferred_freq)

        pred_len = self.prediction_length or task["horizon"]
        n_windows = self.n_windows or task["num_windows"]

        # Build (T, C) series. Each row's per-channel array has the same
        # length (T_i); stack on the last axis.
        series_list, kept = [], []
        for i, (_, row) in enumerate(df.iterrows()):
            channels = [np.asarray(row[c], dtype=np.float32) for c in channel_cols]
            T = channels[0].shape[0]
            if any(ch.shape[0] != T for ch in channels):
                continue
            series_list.append(np.stack(channels, axis=-1))
            kept.append(i)

        if not series_list:
            raise ValueError("All series were skipped (inconsistent channel lengths).")

        covariates = Covariates(
            static_covars=_static_covariates(df, task.get("static_columns")),
            hist_covars=_dynamic_covariates(df, task.get("past_dynamic_columns")),
            future_covars=_dynamic_covariates(df, task.get("known_dynamic_columns")),
        ).select(kept)

        return dict(
            **build_forecasting_data(
                series_list,
                prediction_length=pred_len,
                n_windows=n_windows,
                debug=self.debug,
                # fev-bench's own minimum, but never below the benchmark's.
                min_context=max(task.get("min_context_length", 1), MIN_CONTEXT),
                covariates=covariates,
            ),
            task=self.task,
            # fev-bench's aggregation, next to the pooled one of GIFT-Eval.
            metrics=[*FORECASTING_METRICS, "mase_fev", "wql_fev"],
            prediction_length=pred_len,
            freq=canonical_freq,
            seasonality=task["seasonality"],
            baseline_seasonality=_baseline_seasonality(
                task["seasonality"], inferred_freq
            ),
            baseline_context_length=BASELINE_CONTEXT_LENGTH,
        )
