"""TIME forecasting benchmark dataset (Real-TSF/TIME on HF).

TIME (ICML 2026, arXiv 2602.12147) is a GIFT-Eval-style benchmark of 50
real-world ``<Dataset>/<freq>`` configs scored on 98 (config, term)
tasks. Data and task specs come from two upstream sources:

* Data — HF dataset repo ``Real-TSF/TIME``, laid out as
  ``<Dataset>/<freq>/data-00000-of-00001.arrow`` (e.g.
  ``Australia_Solar/H``, ``CPHL/15T``, ``Crypto/D``).
* Tasks — ``src/timebench/config/datasets.yaml`` in
  github.com/zqiao11/TIME, read at the pinned commit
  :data:`TIME_CONFIG_SHA` for reproducibility.

Parametrization
---------------
* ``dataset_name`` — one of the 50 ``<Dataset>/<freq>`` keys of
  ``datasets.yaml`` (e.g. ``"NE_China_Wind/H"``, ``"Oil_Price/B"``).
  The full list is discoverable via ``benchopt info -v``.
* ``term`` — one of ``short`` / ``medium`` / ``long``. Unlike
  GIFT-Eval, horizons are not a ×1/×10/×15 rule: each config lists an
  explicit ``prediction_length`` per term it defines.

Both are surfaced via ``get_all_parameter_values`` so that
``-d "TIME[dataset_name=all,term=all]"`` and ``benchopt info -v`` work.

Task table and gating
---------------------
Per config, ``datasets.yaml`` gives a ``test_length`` (shared by all
terms, so every term is scored on the same test region) and a subset of
``short`` / ``medium`` / ``long`` with their ``prediction_length``.
Low-frequency configs (``M``, ``Q``, ``W``, most ``D``, ``B``) define
``short`` only. The yaml is fetched once (by ``prepare`` or on first
use) and stored under benchopt's data path. When ``(dataset_name,
term)`` is not defined, ``get_data()`` returns a placeholder carrying a
``_skip_reason`` field, which :meth:`Objective.skip` (see
``objective.py``) honors. So ``-d "TIME[dataset_name=all,term=all]"``
makes 150 attempts, runs the 98 real tasks and skips 52.

Cutoffs and windows (upstream parity)
-------------------------------------
Unlike :mod:`datasets.gifteval`, we reproduce the upstream test split
(``timebench.evaluation.data.Dataset.test_data``) exactly. The test
region of each series is its last ``test_length`` steps, and
``floor(test_length / prediction_length)`` non-overlapping windows are
laid out **forward** from ``T - test_length``:

    cutoff_w = T - test_length + w * prediction_length

When ``test_length`` is not a multiple of ``prediction_length`` (12
tasks, e.g. ``Crypto/D`` short: 273 = 9 × 30 + 3), the last remainder
steps are never scored. Upstream drops them, and so do we. This is why
we don't use :func:`benchmark_utils.windowing.make_forecasting_splits`,
which anchors windows at the series end. ``X_train`` is everything
before ``T - test_length``, so it is the same for every term of a
config. ``n_windows`` keeps the first ``n`` upstream windows.

Schema
------
Each arrow entry has ``item_id``, ``start``, ``freq``, ``target``.
``target`` is ``List[float]`` for univariate configs and
``List[List[float]]`` of shape ``(C, T)`` for multivariate ones. Series
within a config share the same variates. Every variate is a target
(TIME has no exogenous covariates). Multivariate entries are transposed
to the repo's ``(T, C)`` contract. Missing values are kept as NaN. Many
configs have NaNs, some inside test windows (e.g. ``Australia_Solar/H``,
``Smart_Manufacturing/H``); upstream masks them out of its metrics.

The arrow files are Arrow IPC streams (written by HF ``datasets``),
read directly with ``pyarrow``.

Metrics (upstream parity)
-------------------------
Only upstream-exact metrics are reported, with a ``time_`` prefix so they
are never confused with the repo's generic ``mase`` / ``crps``:
``time_mase`` (first, so it is benchopt's ``value``), ``time_crps``,
``time_mae``, ``time_mse``, ``time_rmse``, ``time_mape``, ``time_smape``
and ``time_nd``. See :mod:`benchmark_utils.protocol_metrics`. They port
``compute_per_window_metrics_from_quantiles``
(``src/timebench/evaluation/metrics.py``): one value per (series, window,
variate) cell, then ``np.nanmean`` over all cells as in
``scripts/compute_local_leaderboard.py``. NaN ground-truth steps are
masked, and an all-NaN cell is NaN and drops out of the mean, so tasks
with missing values (e.g. ``Australia_Solar/H``) give finite numbers.
A forecast that emits NaN at a valid step also makes its cell NaN and
drops it, exactly as upstream does. MASE scales each cell by the
seasonal-naive MAE of that window's full history ``x[:cutoff]``, NaNs
removed first. The objective passes this history as the ``contexts``
keyword. CRPS uses exactly the levels 0.1, ..., 0.9, picked from the
forecast's levels. A median-only forecaster (``Naive``,
``SeasonalNaive``) is scored as nine equal quantiles, so its
``time_crps`` equals its ``time_nd``.

``seasonality`` is ``gluonts.time_feature.get_seasonality(freq)``
(gluonts 0.15.1, as upstream), via
:func:`benchmark_utils.protocol_metrics.gluonts_seasonality`. It is
not :func:`~benchmark_utils.forecasting_constants.from_pandas`'s value:
``D`` and ``W`` give 1, not 7 / 52. It is the scale's lag only. The
``SeasonalNaive`` solver uses its own ``season_length`` parameter and the
``Naive`` solver its own ``seasonality`` parameter; neither reads this
field.

Parity was checked on released Real-TSF/TIME-Output predictions. Our
windows, truth and history give the stored per-cell metrics within the
float16 storage error of ``predictions.npz``. The MASE scale, which does
not depend on the predictions, matches to 1e-15. Upstream's
leaderboard also divides each task by the ``seasonal_naive`` task value
and takes a geometric mean over tasks. That step is not done here.
Upstream's ``seasonal_naive`` is statsforecast with prediction
intervals, sampled 100 times at random, so the repo's ``SeasonalNaive``
does not reproduce it. Normalize by the published TIME-Output values.

License
-------
TIME data is released under CC BY-NC 4.0 (non-commercial use only).

Data contract output mirrors :mod:`datasets.gifteval`.
"""

import urllib.request
from pathlib import Path

import numpy as np
from benchopt import BaseDataset, config

from benchmark_utils.covariates import Covariates
from benchmark_utils.download_hf import snapshot_hf_files
from benchmark_utils.forecasting_constants import from_pandas
from benchmark_utils.protocol_metrics import gluonts_seasonality, protocol_metric_names


# ---------------------------------------------------------------------------
# Task table — the upstream ``datasets.yaml`` (50 configs, 98 tasks),
# pinned to a commit of github.com/zqiao11/TIME. Fetched once, stored
# under benchopt's data path by ``prepare`` (or on first use, e.g. when
# expanding ``dataset_name=all``).
# ---------------------------------------------------------------------------
_HF_REPO = "Real-TSF/TIME"
TIME_CONFIG_SHA = "6d3ba1150dfe8e551bc280cba2971bdb276f4f6b"
_TASK_YAML_URL = (
    "https://raw.githubusercontent.com/zqiao11/TIME/"
    f"{TIME_CONFIG_SHA}/src/timebench/config/datasets.yaml"
)
_tasks_cache: "dict[str, dict] | None" = None

TIME_TERMS: tuple[str, ...] = ("short", "medium", "long")


def _task_yaml_path() -> Path:
    # SHA in the filename so bumping the pin invalidates the cached copy.
    name = f"datasets-{TIME_CONFIG_SHA[:12]}.yaml"
    return Path(config.get_data_path(key="time")) / name


def _download_task_yaml(path: Path):
    """Fetch the pinned upstream ``datasets.yaml`` to ``path``."""
    with urllib.request.urlopen(_TASK_YAML_URL, timeout=60) as resp:
        content = resp.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


def _parse_task_yaml(path: Path) -> "dict[str, dict]":
    """Parse ``datasets.yaml`` into ``{dataset_name: spec}``.

    ``spec`` is ``{"test_length": int, "terms": {term: prediction_length}}``,
    where ``terms`` holds only the terms the config defines.
    """
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)["datasets"]
    return {
        name: dict(
            test_length=int(entry["test_length"]),
            terms={
                term: int(entry[term]["prediction_length"])
                for term in TIME_TERMS
                if term in entry
            },
        )
        for name, entry in raw.items()
    }


def _tasks() -> "dict[str, dict]":
    """dataset_name → task spec (disk-cached, see :func:`_parse_task_yaml`)."""
    global _tasks_cache
    if _tasks_cache is None:
        yaml_path = _task_yaml_path()
        if not yaml_path.exists():
            _download_task_yaml(yaml_path)
        _tasks_cache = _parse_task_yaml(yaml_path)
    return _tasks_cache


# Upstream TIME metrics, MASE first so it is the objective's ``value``.
TIME_METRIC_NAMES = protocol_metric_names("time")


def _skip_placeholder(reason: str) -> dict:
    """Flag the combo for skipping via ``Objective.skip``, which benchopt
    calls before ``set_data`` — no other data field is needed."""
    return dict(_skip_reason=reason)


def _window_cutoffs(
    T: int, test_length: int, prediction_length: int, n_windows=None
) -> "list[int]":
    """Upstream TIME test cutoffs for a series of length ``T``.

    Windows are non-overlapping and laid out forward from
    ``T - test_length``. There are ``floor(test_length /
    prediction_length)`` of them, so any remainder at the series end is
    left unscored. ``n_windows`` keeps the first ``n`` of them.
    """
    max_windows = test_length // prediction_length
    if max_windows < 1:
        raise ValueError(
            f"prediction_length ({prediction_length}) exceeds the TIME "
            f"test_length ({test_length})."
        )
    n = max_windows if n_windows is None else int(n_windows)
    if not 1 <= n <= max_windows:
        raise ValueError(
            f"n_windows must be in [1, {max_windows}] for test_length="
            f"{test_length} and prediction_length={prediction_length}; "
            f"got {n_windows}."
        )
    start = T - test_length
    return [start + w * prediction_length for w in range(n)]


def _build_time_splits(
    series: "list[np.ndarray]",
    test_length: int,
    prediction_length: int,
    n_windows=None,
    debug: bool = False,
) -> dict:
    """Train/test split fields of the forecasting data dict, using
    upstream TIME windows (see :func:`_window_cutoffs`).

    ``X_train`` is each series up to ``T - test_length``. Series with no
    history before the test region are dropped. ``y_train`` is ``None``,
    as in :func:`benchmark_utils.windowing.build_forecasting_data`.
    """
    X_train, X_test, cutoff_indexes, y_test = [], [], [], []
    for ts in series:
        T = ts.shape[0]
        if T - test_length < 1:
            continue
        cutoffs = _window_cutoffs(
            T, test_length, prediction_length, 1 if debug else n_windows
        )
        X_train.append(ts[: T - test_length])
        X_test.append(ts)
        cutoff_indexes.append(cutoffs)
        y_test.append(np.stack([ts[c : c + prediction_length] for c in cutoffs]))

    if not X_test:
        raise ValueError(
            f"All series are shorter than the TIME test_length ({test_length})."
        )
    return dict(
        X_train=X_train,
        y_train=None,
        X_test=X_test,
        y_test=y_test,
        cutoff_indexes=cutoff_indexes,
    )


def _read_arrow(files: "list[str]"):
    """Concatenate HF-written arrow files (IPC streams) into one table."""
    import pyarrow as pa

    tables = []
    for f in files:
        with pa.memory_map(f) as source:
            tables.append(pa.ipc.open_stream(source).read_all())
    return pa.concat_tables(tables)


def _target_to_series(target) -> "list[np.ndarray]":
    """Decode an arrow ``target`` column into ``(T, C)`` float32 series.

    Univariate entries (``List[float]``) become ``(T, 1)``; multivariate
    ones (``List[List[float]]``, shape ``(C, T)``) are transposed to
    ``(T, C)``. Decoding goes through flat arrow buffers, not Python
    floats, and nulls become NaN.
    """
    import pyarrow as pa

    series_list = []
    for chunk in target.chunks:
        for entry in chunk:
            values = entry.values
            if pa.types.is_list(values.type) or pa.types.is_large_list(values.type):
                n_variates = len(values)
                flat = values.flatten().to_numpy(zero_copy_only=False)
                arr = flat.astype(np.float32).reshape(n_variates, -1).T  # (T, C)
            else:
                arr = values.to_numpy(zero_copy_only=False)
                arr = arr.astype(np.float32).reshape(-1, 1)  # (T, 1)
            series_list.append(arr)
    return series_list


class Dataset(BaseDataset):
    """TIME forecasting dataset (loaded from HF Real-TSF/TIME).

    Parameters
    ----------
    dataset_name : str
        One of the 50 ``<Dataset>/<freq>`` configs of the upstream
        ``datasets.yaml``, e.g. ``"NE_China_Wind/H"``. See
        ``benchopt info -v``.
    term : str
        ``"short"`` / ``"medium"`` / ``"long"``. Terms the config does
        not define are skipped (placeholder + objective ``skip``).
    prediction_length : int or None
        Explicit override. ``None`` → the term's ``prediction_length``
        from ``datasets.yaml``.
    n_windows : int or None
        ``None`` → upstream ``floor(test_length / prediction_length)``
        windows. An int keeps only the first ``n_windows`` of them.
    max_series : int or None
        Optional cap on the number of series.
    debug : bool
        If True, keep only the first 5 series and a single window.
    """

    name = "TIME"

    requirements = ["pip::huggingface-hub", "pip::pyarrow", "pip::pyyaml"]

    parameters = {
        "dataset_name": ["NE_China_Wind/H"],
        "term": ["short"],
        "prediction_length": [None],
        "n_windows": [None],
        "max_series": [None],
        "debug": [False],
    }

    # ``prepare()`` depends on ``dataset_name`` only — ``term`` and the
    # other knobs shape the in-memory view, not the downloaded files.
    prepare_cache_ignore = (
        "term",
        "prediction_length",
        "n_windows",
        "max_series",
        "debug",
    )

    @classmethod
    def get_all_parameter_values(cls, name):
        if name == "dataset_name":
            return sorted(_tasks())
        if name == "term":
            return list(TIME_TERMS)
        return None

    def prepare(self):
        """Fetch the task yaml and pre-download the arrow file for this
        config into HF's cache."""
        _tasks()
        self._snapshot()

    def _snapshot(self) -> "list[str]":
        """Snapshot-download the arrow files for this config and return
        their local paths (cache-first, see ``snapshot_hf_files``)."""
        return snapshot_hf_files(_HF_REPO, self.dataset_name, "data-*.arrow")

    def get_data(self):
        spec = _tasks().get(self.dataset_name)
        if spec is None:
            raise ValueError(
                f"Unknown TIME dataset {self.dataset_name!r}. See "
                "`benchopt info -v` for valid choices."
            )
        # Short-circuit undefined terms so heavy parsing doesn't run.
        if self.term not in spec["terms"]:
            return _skip_placeholder(
                f"non-canonical TIME combo: {self.dataset_name!r} does not "
                f"define term {self.term!r} in datasets.yaml"
            )

        arrow_files = self._snapshot()
        if not arrow_files:
            raise ValueError(
                f"No Arrow file found for TIME dataset {self.dataset_name!r}."
            )
        table = _read_arrow(arrow_files)

        # Slice the Arrow table (zero-copy) before decoding anything.
        n_keep = 5 if self.debug else self.max_series
        if n_keep is not None:
            table = table.slice(0, int(n_keep))
        if table.num_rows == 0:
            raise ValueError(f"TIME dataset {self.dataset_name!r} returned 0 series.")

        # Every series in a TIME config shares the same freq. Seasonality
        # is gluonts' (as upstream), not ``from_pandas``'s.
        pandas_freq = table.column("freq")[0].as_py()
        freq, _, _ = from_pandas(pandas_freq)

        pred_len = self.prediction_length
        if pred_len is None:
            pred_len = spec["terms"][self.term]

        return dict(
            **_build_time_splits(
                _target_to_series(table.column("target")),
                test_length=spec["test_length"],
                prediction_length=pred_len,
                n_windows=self.n_windows,
                debug=self.debug,
            ),
            covariates=Covariates(),  # TIME has no exogenous covariates
            task="forecasting",
            metrics=list(TIME_METRIC_NAMES),
            prediction_length=pred_len,
            freq=freq,
            seasonality=gluonts_seasonality(pandas_freq),
        )
