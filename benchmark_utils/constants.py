"""Shared frequency / seasonality tables for forecasting datasets.

Two sources name frequencies differently:
  - aeon (used by Monash) uses words: "yearly", "weekly", "minutely", ...
  - GIFT-Eval (and pandas) use offset aliases: "Y", "W-SUN", "5T", ...

This module exposes a single canonical (freq, seasonality) lookup keyed on
the canonical pandas-style base alias (e.g. "Y", "W", "D"), plus two
adapters that normalize each source onto that canonical key.
"""

import re

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

# aeon's spelled-out names → canonical base alias
_AEON_TO_BASE = {
    "yearly":    "Y",
    "quarterly": "Q",
    "monthly":   "M",
    "weekly":    "W",
    "daily":     "D",
    "hourly":    "H",
    "minutely":  "T",
    "seconds":   "S",
}


def from_aeon(freq_word: str) -> tuple[str, int, int]:
    """Look up (freq, seasonality, default_horizon) from an aeon freq word."""
    base = _AEON_TO_BASE.get(freq_word, "D")
    return _BASE[base]


# Pandas offset aliases: strip a leading multiplier and any anchor suffix
# (e.g. "5T" → "T", "W-SUN" → "W", "QS-OCT" → "Q", "YE" → "Y").
_PANDAS_ALIAS_RE = re.compile(r"^\d*([A-Za-z]+)")
_NORMALIZE_BASE = {
    # Newer pandas spellings → legacy single-letter aliases used in _BASE.
    "YE": "Y", "YS": "Y", "A": "Y", "AS": "Y",
    "QE": "Q", "QS": "Q",
    "ME": "M", "MS": "M",
    "min": "T", "MIN": "T",
}


def from_pandas(freq_alias: str) -> tuple[str, int, int]:
    """Look up (freq, seasonality, default_horizon) from a pandas freq alias.

    Handles multipliers ("5T") and anchors ("W-SUN", "QS-OCT") by stripping
    them before lookup. Unknown aliases default to daily.
    """
    if not freq_alias:
        return _BASE["D"]
    m = _PANDAS_ALIAS_RE.match(freq_alias.split("-", 1)[0])
    if not m:
        return _BASE["D"]
    head = m.group(1)
    base = _NORMALIZE_BASE.get(head, head[:1].upper())
    return _BASE.get(base, _BASE["D"])


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

GIFT_EVAL_TERM_MULTIPLIER: dict[str, int] = {
    "short":  1,
    "medium": 10,
    "long":   15,
}


def gift_eval_prediction_length(freq: str, term: str) -> int:
    """Resolve the GIFT-Eval prediction length for a (freq, term) pair.

    ``freq`` is a pandas-style alias (e.g. ``"5T"``, ``"1H"``, ``"W-SUN"``).
    Lookup falls back through: exact match → strip leading "1" multiplier
    ("1H" → "H") → collapse any multi-X alias to its base X ("10S" → "S",
    "30T" → "T") → default 48. ``term`` must be one of ``"short"``,
    ``"medium"``, ``"long"``.
    """
    if term not in GIFT_EVAL_TERM_MULTIPLIER:
        raise ValueError(
            f"term must be one of {list(GIFT_EVAL_TERM_MULTIPLIER)}; got {term!r}"
        )
    base = GIFT_EVAL_PRED_LENGTH_MAP.get(freq)
    if base is None:
        m = _PANDAS_ALIAS_RE.match(freq.split("-", 1)[0])
        if m:
            head = m.group(1)
            # Normalize new pandas spellings ("QE"→"Q", "ME"→"M", ...)
            # before falling back through the map.
            head = _NORMALIZE_BASE.get(head, head)
            base = GIFT_EVAL_PRED_LENGTH_MAP.get(head)
    if base is None:
        base = 48
    return base * GIFT_EVAL_TERM_MULTIPLIER[term]
