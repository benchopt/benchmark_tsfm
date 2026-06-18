"""GIFT-Eval forecasting benchmark dataset (Salesforce/GiftEval on HF).

Parametrization
---------------
The class exposes two orthogonal parameters that drive the leaderboard
matrix:

* ``dataset_name`` — one of 55 canonical ``<name>/<freq>`` paths (e.g.
  ``"m4_weekly/W"``, ``"loop_seattle/H"``). The full list is
  :data:`GIFTEVAL_DATASETS`.
* ``term`` — one of ``short`` / ``medium`` / ``long``, controlling the
  forecast horizon (×1, ×10, ×15 of the per-freq base).

Both are surfaced via ``get_all_parameter_values`` so that
``-d "GiftEval[dataset_name=all,term=short]"`` and ``benchopt info -v``
work.

Canonical-combo gating
----------------------
GIFT-Eval scores only **97** of the 55 × 3 = 165 possible ``(path,
term)`` combinations on its public leaderboard. The 34 short-only paths
do not define ``medium`` / ``long``. We track the canonical set in
:data:`CANONICAL_COMBOS` and gate runs at the dataset level: when
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
:func:`benchmark_utils.constants.gift_eval_prediction_length`.

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


# ---------------------------------------------------------------------------
# Single source of truth: leaderboard ``<name>/<freq>`` path → tuple of
# terms that path defines. Derived from
# gift-eval/results/*/all_results.csv. 55 paths, 97 (path, term) triples;
# 34 paths are short-only, 21 define all three.
# ---------------------------------------------------------------------------
_LEADERBOARD: dict[str, tuple[str, ...]] = {
    "bitbrains_fast_storage/5T":   ("short", "medium", "long"),
    "bitbrains_fast_storage/H":    ("short",),
    "bitbrains_rnd/5T":            ("short", "medium", "long"),
    "bitbrains_rnd/H":             ("short",),
    "bizitobs_application/10S":    ("short", "medium", "long"),
    "bizitobs_l2c/5T":             ("short", "medium", "long"),
    "bizitobs_l2c/H":              ("short", "medium", "long"),
    "bizitobs_service/10S":        ("short", "medium", "long"),
    "car_parts/M":                 ("short",),
    "covid_deaths/D":              ("short",),
    "electricity/15T":             ("short", "medium", "long"),
    "electricity/D":               ("short",),
    "electricity/H":               ("short", "medium", "long"),
    "electricity/W":               ("short",),
    "ett1/15T":                    ("short", "medium", "long"),
    "ett1/D":                      ("short",),
    "ett1/H":                      ("short", "medium", "long"),
    "ett1/W":                      ("short",),
    "ett2/15T":                    ("short", "medium", "long"),
    "ett2/D":                      ("short",),
    "ett2/H":                      ("short", "medium", "long"),
    "ett2/W":                      ("short",),
    "hierarchical_sales/D":        ("short",),
    "hierarchical_sales/W":        ("short",),
    "hospital/M":                  ("short",),
    "jena_weather/10T":            ("short", "medium", "long"),
    "jena_weather/D":              ("short",),
    "jena_weather/H":              ("short", "medium", "long"),
    "kdd_cup_2018/D":              ("short",),
    "kdd_cup_2018/H":              ("short", "medium", "long"),
    "loop_seattle/5T":             ("short", "medium", "long"),
    "loop_seattle/D":              ("short",),
    "loop_seattle/H":              ("short", "medium", "long"),
    "m4_daily/D":                  ("short",),
    "m4_hourly/H":                 ("short",),
    "m4_monthly/M":                ("short",),
    "m4_quarterly/Q":              ("short",),
    "m4_weekly/W":                 ("short",),
    "m4_yearly/A":                 ("short",),
    "m_dense/D":                   ("short",),
    "m_dense/H":                   ("short", "medium", "long"),
    "restaurant/D":                ("short",),
    "saugeen/D":                   ("short",),
    "saugeen/M":                   ("short",),
    "saugeen/W":                   ("short",),
    "solar/10T":                   ("short", "medium", "long"),
    "solar/D":                     ("short",),
    "solar/H":                     ("short", "medium", "long"),
    "solar/W":                     ("short",),
    "sz_taxi/15T":                 ("short", "medium", "long"),
    "sz_taxi/H":                   ("short",),
    "temperature_rain/D":          ("short",),
    "us_births/D":                 ("short",),
    "us_births/M":                 ("short",),
    "us_births/W":                 ("short",),
}


# Public derived constants — what users and CLI tooling reference.
GIFTEVAL_DATASETS: tuple[str, ...] = tuple(sorted(_LEADERBOARD))
GIFTEVAL_TERMS: tuple[str, ...] = ("short", "medium", "long")
CANONICAL_COMBOS: frozenset[tuple[str, str]] = frozenset(
    (path, term) for path, terms in _LEADERBOARD.items() for term in terms
)


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
    """Return a minimal data dict that satisfies ``Objective.set_data``
    but flags the combo for skipping via ``Objective.skip``."""
    return dict(
        X_train=[],
        y_train=[],
        X_test=[],
        y_test=[],
        cutoff_indexes=[],
        covariates=Covariates(),
        task="forecasting",
        metrics=[],
        prediction_length=1,
        freq="D",
        seasonality=1,
        _skip_reason=reason,
    )


class Dataset(BaseDataset):
    """GIFT-Eval forecasting dataset (loaded from HF Salesforce/GiftEval).

    Parameters
    ----------
    dataset_name : str
        One of 55 canonical leaderboard paths — ``<name>/<freq>``, e.g.
        ``"m4_weekly/W"``, ``"loop_seattle/H"``. See
        :data:`GIFTEVAL_DATASETS`.
    term : str
        ``"short"`` / ``"medium"`` / ``"long"``. Combos not in
        :data:`CANONICAL_COMBOS` are skipped (placeholder + objective
        ``skip``), so ``dataset_name=all, term=long`` runs only the 21
        paths that define ``long``.
    prediction_length : int or None
        Explicit override. ``None`` → resolved from (freq, term) via
        :func:`benchmark_utils.constants.gift_eval_prediction_length`.
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
            return list(GIFTEVAL_DATASETS)
        if name == "term":
            return list(GIFTEVAL_TERMS)
        return None

    def prepare(self):
        """Pre-download arrow shards for this config into HF's cache."""
        self._snapshot()

    def _snapshot(self) -> "list[str]":
        """Snapshot-download the arrow files for this dataset and return
        their local paths. Idempotent — HF caches by content hash."""
        from huggingface_hub import snapshot_download
        from pathlib import Path

        hf_path = _hf_arrow_directory(self.dataset_name)
        local_root = snapshot_download(
            "Salesforce/GiftEval",
            repo_type="dataset",
            allow_patterns=f"{hf_path}/*.arrow",
        )
        return sorted(str(p) for p in (Path(local_root) / hf_path).glob("*.arrow"))

    def get_data(self):
        from datasets import Dataset as HFDataset

        # Short-circuit non-canonical combos so heavy parsing doesn't run.
        if (self.dataset_name, self.term) not in CANONICAL_COMBOS:
            return _skip_placeholder(
                f"non-canonical GIFT-Eval combo: {self.dataset_name!r} does "
                f"not define term {self.term!r} on the leaderboard"
            )

        arrow_files = self._snapshot()
        if not arrow_files:
            raise ValueError(
                f"No Arrow file found for GIFT-Eval dataset "
                f"{self.dataset_name!r}. Valid choices are in GIFTEVAL_DATASETS."
            )

        rows = []
        for f in arrow_files:
            rows.extend(HFDataset.from_file(f))

        if self.debug:
            rows = rows[:5]
        elif self.max_series is not None:
            rows = rows[: int(self.max_series)]

        if not rows:
            raise ValueError(
                f"GIFT-Eval dataset {self.dataset_name!r} returned 0 series."
            )

        # Frequency / seasonality — every series in a GIFT-Eval subset
        # shares the same freq, so taking it from the first entry is safe.
        pandas_freq = rows[0].get("freq") or "D"
        freq, seasonality, _ = from_pandas(pandas_freq)

        pred_len = self.prediction_length
        if pred_len is None:
            pred_len = gift_eval_prediction_length(pandas_freq, self.term)

        # Build (T, C) series. Univariate entries arrive as flat
        # ``List[float]`` (ndim=1); multivariate as ``List[List[float]]``
        # of shape ``(C, T)``.
        series_list = []
        for r in rows:
            values = np.asarray(r["target"], dtype=np.float32)
            if values.ndim == 1:
                series_list.append(values.reshape(-1, 1))         # (T, 1)
            elif values.ndim == 2:
                series_list.append(values.T)                        # (C,T)→(T,C)

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
