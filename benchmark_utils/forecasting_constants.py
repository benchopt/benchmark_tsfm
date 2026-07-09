"""Shared forecasting constants: frequency / seasonality tables and metrics.

Two sources name frequencies differently:
  - aeon (used by Monash) uses words: "yearly", "weekly", "minutely", ...
  - GIFT-Eval (and pandas) use offset aliases: "Y", "W-SUN", "5T", ...

This module exposes a single canonical (freq, seasonality) lookup keyed on
the canonical pandas-style base alias (e.g. "Y", "W", "D"), plus two
adapters that normalize each source onto that canonical key.
"""

import re

# Metrics reported by every forecasting dataset (names from
# benchmark_utils.metrics.ALL_METRICS).
FORECASTING_METRICS = (
    "mae", "mse", "rmse", "mase", "smape",
    "crps", "wql", "mcis", "pinball", "skill_score_ratio",
)

# Canonical base alias → (display_freq, MASE seasonality, default forecast horizon)
_BASE = {
    "Y": ("Y",   1,  6),
    "Q": ("Q",   4,  8),
    "M": ("M",  12, 12),
    "W": ("W",  52, 13),
    "D": ("D",   7, 14),
    "H": ("H",  24, 24),
    "T": ("T", 1440, 60),   # minutes
    "S": ("S",   1, 60),
}

# aeon's spelled-out names → pandas offset alias. Sub-hourly words map to
# multiplied aliases so the seasonality accounts for the step size.
_AEON_TO_ALIAS = {
    "yearly":      "Y",
    "quarterly":   "Q",
    "monthly":     "M",
    "weekly":      "W",
    "daily":       "D",
    "hourly":      "H",
    "half_hourly": "30T",
    "minutely":    "T",
    "10_minutes":  "10T",
    "seconds":     "S",
    "4_seconds":   "4S",
}


def from_aeon(freq_word: str) -> tuple[str, int, int]:
    """Look up (freq, seasonality, default_horizon) from an aeon freq word.

    Unknown words default to daily.
    """
    return from_pandas(_AEON_TO_ALIAS.get(freq_word, "D"))


# Pandas offset aliases: capture the leading multiplier and the unit,
# ignoring any anchor suffix (e.g. "5T" → (5, "T"), "W-SUN" → (1, "W")).
_PANDAS_ALIAS_RE = re.compile(r"^(\d*)([A-Za-z]+)")
_NORMALIZE_BASE = {
    # Newer pandas spellings → legacy single-letter aliases used in _BASE.
    "YE": "Y", "YS": "Y", "A": "Y", "AS": "Y",
    "QE": "Q", "QS": "Q",
    "ME": "M", "MS": "M",
    "min": "T", "MIN": "T",
}


def from_pandas(freq_alias: str) -> tuple[str, int, int]:
    """Look up (freq, seasonality, default_horizon) from a pandas freq alias.

    Anchors ("W-SUN", "QS-OCT") are stripped before lookup. A multiplier
    scales the step size, so the seasonality is divided by it: at "15T"
    one day is 1440/15 = 96 steps, not 1440. The original alias is
    returned as freq so calendar-building consumers (``pd.date_range``)
    keep the true sampling rate. Unknown aliases default to daily.
    """
    if not freq_alias:
        return _BASE["D"]
    m = _PANDAS_ALIAS_RE.match(freq_alias.split("-", 1)[0])
    if not m:
        return _BASE["D"]
    mult = int(m.group(1)) if m.group(1) else 1
    base = _NORMALIZE_BASE.get(m.group(2), m.group(2)[:1].upper())
    if base not in _BASE:
        return _BASE["D"]
    _, seasonality, default_h = _BASE[base]
    return freq_alias, max(1, seasonality // max(mult, 1)), default_h


# ---------------------------------------------------------------------------
# GIFT-Eval term resolution
#
# Mirrors the canonical table in the upstream time-series repo: prediction
# length is a function of pandas freq, then scaled by a term multiplier
# (short=1, medium=10, long=15). Used by datasets/gifteval.py so reported
# numbers line up with the GIFT-Eval leaderboard.
# ---------------------------------------------------------------------------

GIFT_EVAL_PRED_LENGTH_MAP: dict[str, int] = {
    "M":  12, "MS": 12,
    "W":   8, "W-SUN": 8, "W-MON": 8,
    "D":  30,
    "H":  48, "6H": 48,
    "T":  48, "5T": 48, "10T": 48, "15T": 48, "30T": 48,
    "S":  60, "4S": 60,
    "Q":   8, "Q-DEC": 8,
    "A":   4, "A-DEC": 4,
    "Y":   4,
}

# M4-competition horizons differ from the generic table; upstream
# gift-eval selects this map whenever "m4" is in the dataset name.
M4_PRED_LENGTH_MAP: dict[str, int] = {
    "A": 6, "Y": 6,
    "Q": 8,
    "M": 18,
    "W": 13,
    "D": 14,
    "H": 48,
}

GIFT_EVAL_TERM_MULTIPLIER: dict[str, int] = {
    "short":  1,
    "medium": 10,
    "long":   15,
}


def gift_eval_prediction_length(
    freq: str, term: str, dataset_name: str = ""
) -> int:
    """Resolve the GIFT-Eval prediction length for a (freq, term) pair.

    ``freq`` is a pandas-style alias (e.g. ``"5T"``, ``"1H"``, ``"W-SUN"``).
    Lookup falls back through: exact match → strip leading "1" multiplier
    ("1H" → "H") → collapse any multi-X alias to its base X ("10S" → "S",
    "30T" → "T") → default 48. ``term`` must be one of ``"short"``,
    ``"medium"``, ``"long"``. When ``dataset_name`` contains "m4", the
    M4-competition horizons are used, mirroring upstream gift-eval.
    """
    if term not in GIFT_EVAL_TERM_MULTIPLIER:
        raise ValueError(
            f"term must be one of {list(GIFT_EVAL_TERM_MULTIPLIER)}; got {term!r}"
        )
    pred_length_map = (
        M4_PRED_LENGTH_MAP if "m4" in dataset_name
        else GIFT_EVAL_PRED_LENGTH_MAP
    )
    base = pred_length_map.get(freq)
    if base is None:
        m = _PANDAS_ALIAS_RE.match(freq.split("-", 1)[0])
        if m:
            head = m.group(2)
            # Normalize new pandas spellings ("QE"→"Q", "ME"→"M", ...)
            # before falling back through the map.
            head = _NORMALIZE_BASE.get(head, head)
            base = pred_length_map.get(head)
    if base is None:
        base = 48
    return base * GIFT_EVAL_TERM_MULTIPLIER[term]
