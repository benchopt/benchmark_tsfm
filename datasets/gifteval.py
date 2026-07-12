"""GIFT-Eval forecasting benchmark dataset (Salesforce/GiftEval on HF).

Parametrization
---------------
The class exposes two orthogonal parameters that drive the leaderboard
matrix:

* ``dataset_name`` — one of 55 canonical ``<name>/<freq>`` paths (e.g.
  ``"m4_weekly/W"``, ``"loop_seattle/H"``). The full list is derived
  from the leaderboard CSV and discoverable via ``benchopt info -v``.
* ``term`` — one of ``short`` / ``medium`` / ``long``, controlling the
  forecast horizon (×1, ×10, ×15 of the per-freq base).

Both are surfaced via ``get_all_parameter_values`` so that
``-d "GiftEval[dataset_name=all,term=short]"`` and ``benchopt info -v``
work.

Canonical-combo gating
----------------------
GIFT-Eval scores only **97** of the 55 × 3 = 165 possible ``(path,
term)`` combinations on its public leaderboard. The 34 short-only paths
do not define ``medium`` / ``long``. The canonical set is derived
from the leaderboard's results CSV (fetched from the HF Space
``Salesforce/GIFT-Eval`` by ``prepare`` or on first use, and stored as a
small CSV under benchopt's data path) and gates runs at the dataset
level: when
``(dataset_name, term)`` is not canonical, ``get_data()`` short-circuits
and returns a placeholder dict carrying a ``_skip_reason`` field.
:meth:`Objective.skip` (see ``objective.py``) honors that field and
skips the combo cleanly.

So:

* ``-d "GiftEval[dataset_name=all,term=short]"`` → 55 canonical runs.
* ``-d "GiftEval[dataset_name=all,term=long]"``  → 55 attempts,
  21 canonical runs, 34 skipped.
* ``-d "GiftEval[dataset_name=all,term=all]"``   → 165 attempts,
  97 canonical runs, 68 skipped.

Leaderboard names vs HF directory names
---------------------------------------
The leaderboard uses lowercase, paper-style identifiers (e.g.
``loop_seattle/H``, ``m_dense/D``, ``car_parts/M``) while the HF repo
``Salesforce/GiftEval`` uses mixed-case directory names that don't
always match (``LOOP_SEATTLE/H``, ``M_DENSE/D``,
``car_parts_with_missing/``). We accept leaderboard names — that's what
appears in the paper, the leaderboard, and the gift-eval README — and
translate to HF paths internally via :data:`_LEADERBOARD_TO_HF`. Cases:

  * Pure case difference: ``loop_seattle`` → ``LOOP_SEATTLE``,
    ``m_dense`` → ``M_DENSE``, ``sz_taxi`` → ``SZ_TAXI``.
  * Missing-data suffix: ``car_parts`` → ``car_parts_with_missing``,
    ``kdd_cup_2018`` → ``kdd_cup_2018_with_missing``,
    ``temperature_rain`` → ``temperature_rain_with_missing``.
  * Rename: ``saugeen`` → ``saugeenday``.
  * Leaderboard adds a freq segment for HF-flat datasets: leaderboard
    ``m4_yearly/A`` → HF flat ``m4_yearly`` (the freq is implicit in the
    data, not the path). Likewise for the other ``m4_*``,
    ``car_parts/M``, ``covid_deaths/D``, ``hospital/M``,
    ``restaurant/D``, ``temperature_rain/D``,
    ``bizitobs_application/10S``, ``bizitobs_service/10S``.

Schema
------
Each HF entry exposes ``item_id``, ``start``, ``freq``, ``target``.
``target`` is a flat ``List[float]`` for univariate configs and a
``List[List[float]]`` of shape ``(C, T)`` for multivariate ones (e.g.
``bitbrains_*``, ``electricity/*``, ``ett1/*``, ``ett2/*``,
``jena_weather/*``, ``solar/*``). Both shapes are handled — multivariate
entries are transposed to the repo's ``(T, C)`` contract.

Cutoffs and windows
-------------------
We don't comply with GIFT-Eval's prescribed test cutoff; we use the same
rolling-window logic as Monash via
:func:`benchmark_utils.windowing.make_forecasting_splits`. The
``prediction_length`` for a given (freq, term) follows GIFT-Eval's
canonical ``base × multiplier`` rule via
:func:`benchmark_utils.forecasting_constants.gift_eval_prediction_length`.

Data contract output mirrors :mod:`datasets.monash`.
"""

import csv
from pathlib import Path

import numpy as np
from benchopt import BaseDataset, config

from benchmark_utils.covariates import Covariates
from benchmark_utils.download_hf import snapshot_hf_files
from benchmark_utils.forecasting_constants import (
    FORECASTING_METRICS,
    from_pandas,
    gift_eval_prediction_length,
)
from benchmark_utils.windowing import build_forecasting_data


# ---------------------------------------------------------------------------
# Canonical (dataset_name, term) table — derived from the GIFT-Eval
# leaderboard Space (results/seasonal_naive/all_results.csv: 55 paths,
# 97 combos, 34 short-only). Fetched once, stored as a small CSV under
# benchopt's data path by ``prepare`` (or on first use, e.g. when
# expanding ``dataset_name=all``).
# ---------------------------------------------------------------------------
_LEADERBOARD_REPO = "Salesforce/GIFT-Eval"
_LEADERBOARD_FILE = "results/seasonal_naive/all_results.csv"
_leaderboard_cache: "dict[str, tuple[str, ...]] | None" = None

GIFTEVAL_TERMS: tuple[str, ...] = ("short", "medium", "long")


def _leaderboard_csv_path() -> Path:
    return Path(config.get_data_path(key="gifteval")) / "leaderboard_combos.csv"


def _build_leaderboard_csv(path: Path):
    """Fetch the leaderboard results CSV and store the (dataset_name, term)
    combos it scores."""
    from huggingface_hub import hf_hub_download

    src = hf_hub_download(_LEADERBOARD_REPO, _LEADERBOARD_FILE, repo_type="space")
    with open(src) as f:
        # "dataset" column is "<name>/<freq>/<term>".
        combos = [row["dataset"].rsplit("/", 1) for row in csv.DictReader(f)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset_name", "term"])
        writer.writerows(combos)


def _parse_leaderboard_csv(path: Path) -> "dict[str, tuple[str, ...]]":
    table: dict = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            table.setdefault(row["dataset_name"], []).append(row["term"])
    return {name: tuple(terms) for name, terms in table.items()}


def _leaderboard() -> "dict[str, tuple[str, ...]]":
    """dataset_name → terms it defines on the leaderboard (disk-cached)."""
    global _leaderboard_cache
    if _leaderboard_cache is None:
        csv_path = _leaderboard_csv_path()
        if not csv_path.exists():
            _build_leaderboard_csv(csv_path)
        _leaderboard_cache = _parse_leaderboard_csv(csv_path)
    return _leaderboard_cache


# ---------------------------------------------------------------------------
# Leaderboard ``<name>`` → HF top-level directory name. Only entries that
# differ from the lowercase identity mapping appear here.
# ---------------------------------------------------------------------------
_LEADERBOARD_TO_HF: dict[str, str] = {
    "loop_seattle":     "LOOP_SEATTLE",
    "m_dense":          "M_DENSE",
    "sz_taxi":          "SZ_TAXI",
    "car_parts":        "car_parts_with_missing",
    "kdd_cup_2018":     "kdd_cup_2018_with_missing",
    "temperature_rain": "temperature_rain_with_missing",
    "saugeen":          "saugeenday",
}


# ---------------------------------------------------------------------------
# Datasets that live as a single arrow file directly under the dataset
# name (no per-freq subdir on HF). The leaderboard still adds a freq
# segment to their paths (e.g. ``m4_yearly/A``, ``hospital/M``), which we
# strip before locating the file.
# ---------------------------------------------------------------------------
_HF_FLAT_DATASETS: frozenset[str] = frozenset({
    "bizitobs_application", "bizitobs_service",
    "car_parts_with_missing", "covid_deaths", "hospital",
    "m4_daily", "m4_hourly", "m4_monthly", "m4_quarterly",
    "m4_weekly", "m4_yearly",
    "restaurant", "temperature_rain_with_missing",
})


def _hf_arrow_directory(leaderboard_path: str) -> str:
    """Resolve a leaderboard ``<name>/<freq>`` path to the actual HF
    directory containing the arrow file.

    Examples
    --------
        ``"m4_weekly/W"``       → ``"m4_weekly"`` (HF-flat, drops freq)
        ``"loop_seattle/H"``    → ``"LOOP_SEATTLE/H"`` (case-renamed)
        ``"car_parts/M"``       → ``"car_parts_with_missing"`` (HF-flat + suffix)
    """
    leaderboard_name, _, freq_segment = leaderboard_path.partition("/")
    hf_name = _LEADERBOARD_TO_HF.get(leaderboard_name, leaderboard_name)
    if hf_name in _HF_FLAT_DATASETS:
        return hf_name
    if freq_segment:
        return f"{hf_name}/{freq_segment}"
    return hf_name


def _skip_placeholder(reason: str) -> dict:
    """Flag the combo for skipping via ``Objective.skip``, which benchopt
    calls before ``set_data`` — no other data field is needed."""
    return dict(_skip_reason=reason)


class Dataset(BaseDataset):
    """GIFT-Eval forecasting dataset (loaded from HF Salesforce/GiftEval).

    Parameters
    ----------
    dataset_name : str
        One of 55 canonical leaderboard paths — ``<name>/<freq>``, e.g.
        ``"m4_weekly/W"``, ``"loop_seattle/H"``. See
        ``benchopt info -v``.
    term : str
        ``"short"`` / ``"medium"`` / ``"long"``. Combos not on the
        leaderboard are skipped (placeholder + objective
        ``skip``), so ``dataset_name=all, term=long`` runs only the 21
        paths that define ``long``.
    prediction_length : int or None
        Explicit override. ``None`` → resolved from (freq, term) via
        :func:`benchmark_utils.forecasting_constants.gift_eval_prediction_length`.
    n_windows : int
        Number of rolling evaluation windows per series.
    max_series : int or None
        Optional cap on the number of series.
    debug : bool
        If True, keep only the first 5 series for fast iteration.
    """

    name = "GiftEval"

    requirements = ["pip::datasets", "pip::huggingface-hub"]

    parameters = {
        "dataset_name": ["m4_weekly/W"],
        "term": ["short"],
        "prediction_length": [None],
        "n_windows": [1],
        "max_series": [None],
        "debug": [False],
    }

    # ``prepare()`` depends on ``dataset_name`` only — ``term`` and the
    # other knobs shape the in-memory view, not the downloaded files.
    prepare_cache_ignore = (
        "term", "prediction_length", "n_windows", "max_series", "debug",
    )

    @classmethod
    def get_all_parameter_values(cls, name):
        if name == "dataset_name":
            return sorted(_leaderboard())
        if name == "term":
            return list(GIFTEVAL_TERMS)
        return None

    def prepare(self):
        """Build the leaderboard-combos CSV and pre-download the arrow
        shards for this config into HF's cache."""
        _leaderboard()
        self._snapshot()

    def _snapshot(self) -> "list[str]":
        """Snapshot-download the arrow files for this dataset and return
        their local paths (cache-first, see ``snapshot_hf_files``).

        data-*.arrow only: the hub repo also holds stray HF map-cache
        shards (cache-*.arrow, e.g. under electricity/15T) with
        duplicated rows that must not be loaded.
        """
        return snapshot_hf_files(
            "Salesforce/GiftEval",
            _hf_arrow_directory(self.dataset_name),
            "data-*.arrow",
        )

    def get_data(self):
        # Short-circuit non-canonical combos so heavy parsing doesn't run.
        if self.term not in _leaderboard().get(self.dataset_name, ()):
            return _skip_placeholder(
                f"non-canonical GIFT-Eval combo: {self.dataset_name!r} does "
                f"not define term {self.term!r} on the leaderboard"
            )

        from datasets import Dataset as HFDataset, concatenate_datasets

        arrow_files = self._snapshot()
        if not arrow_files:
            raise ValueError(
                f"No Arrow file found for GIFT-Eval dataset "
                f"{self.dataset_name!r}. See `benchopt info` for valid choices."
            )

        parts = [HFDataset.from_file(f) for f in arrow_files]
        ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)

        # Slice on the Arrow table (zero-copy) before decoding anything.
        n_keep = 5 if self.debug else self.max_series
        if n_keep is not None:
            ds = ds.select(range(min(int(n_keep), len(ds))))

        if len(ds) == 0:
            raise ValueError(
                f"GIFT-Eval dataset {self.dataset_name!r} returned 0 series."
            )

        # Frequency / seasonality — every series in a GIFT-Eval subset
        # shares the same freq, so taking it from the first entry is safe.
        pandas_freq = ds[0].get("freq") or "D"
        freq, seasonality, _ = from_pandas(pandas_freq)

        pred_len = self.prediction_length
        if pred_len is None:
            pred_len = gift_eval_prediction_length(
                pandas_freq, self.term, dataset_name=self.dataset_name
            )

        # Build (T, C) series with columnar numpy access — avoids
        # round-tripping every float through python objects. Univariate
        # entries arrive as flat arrays (ndim=1); multivariate as (C, T).
        series_list = []
        for values in ds.with_format("numpy")["target"]:
            values = np.asarray(values, dtype=np.float32)
            if values.ndim == 1:
                series_list.append(values.reshape(-1, 1))         # (T, 1)
            elif values.ndim == 2:
                series_list.append(values.T)                        # (C,T)→(T,C)

        if not series_list:
            raise ValueError(
                f"All entries in GIFT-Eval dataset {self.dataset_name!r} "
                "had unsupported target shapes."
            )

        return dict(
            **build_forecasting_data(
                series_list,
                prediction_length=pred_len,
                n_windows=self.n_windows,
                debug=self.debug,
            ),
            covariates=Covariates(),  # GIFT-Eval HF schema has no covariates
            task="forecasting",
            metrics=list(FORECASTING_METRICS),
            prediction_length=pred_len,
            freq=freq,
            seasonality=seasonality,
        )
