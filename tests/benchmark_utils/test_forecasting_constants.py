"""Conversion tests for the shared frequency / seasonality tables."""

import pytest

from benchmark_utils.forecasting_constants import (
    from_aeon,
    from_pandas,
    gift_eval_prediction_length,
)


@pytest.mark.parametrize(
    "alias, expected",
    [
        ("H", ("H", 24, 24)),
        ("D", ("D", 7, 14)),
        ("W-SUN", ("W-SUN", 52, 13)),
        ("QS-OCT", ("QS-OCT", 4, 8)),
        ("YE", ("YE", 1, 6)),
        # Multipliers scale the seasonality; freq keeps the true step.
        ("5T", ("5T", 288, 60)),
        ("15T", ("15T", 96, 60)),
        ("30T", ("30T", 48, 60)),
        ("15min", ("15min", 96, 60)),
        ("6H", ("6H", 4, 24)),
        ("10S", ("10S", 1, 60)),
        # Unknown or empty aliases fall back to daily.
        ("", ("D", 7, 14)),
        ("??", ("D", 7, 14)),
    ],
)
def test_from_pandas(alias, expected):
    assert from_pandas(alias) == expected


@pytest.mark.parametrize(
    "word, expected",
    [
        ("yearly", ("Y", 1, 6)),
        ("monthly", ("M", 12, 12)),
        ("hourly", ("H", 24, 24)),
        # Sub-hourly aeon words resolve through multiplied aliases.
        ("half_hourly", ("30T", 48, 60)),
        ("10_minutes", ("10T", 144, 60)),
        ("4_seconds", ("4S", 1, 60)),
        ("unknown_word", ("D", 7, 14)),
    ],
)
def test_from_aeon(word, expected):
    assert from_aeon(word) == expected


def test_gift_eval_prediction_length_terms():
    assert gift_eval_prediction_length("H", "short") == 48
    assert gift_eval_prediction_length("5T", "medium") == 480
    assert gift_eval_prediction_length("H", "long") == 720
    # Multiplied aliases missing from the map fall back to their base.
    assert gift_eval_prediction_length("20T", "short") == 48
    with pytest.raises(ValueError):
        gift_eval_prediction_length("H", "weekly")


def test_gift_eval_prediction_length_m4():
    # m4 datasets follow the M4-competition horizons, not the generic map.
    assert gift_eval_prediction_length("A", "short", "m4_yearly/A") == 6
    assert gift_eval_prediction_length("Q", "short", "m4_quarterly/Q") == 8
    assert gift_eval_prediction_length("M", "short", "m4_monthly/M") == 18
    assert gift_eval_prediction_length("W", "short", "m4_weekly/W") == 13
    assert gift_eval_prediction_length("D", "short", "m4_daily/D") == 14
    assert gift_eval_prediction_length("H", "short", "m4_hourly/H") == 48
