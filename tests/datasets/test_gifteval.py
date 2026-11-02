"""Tests for the GIFT-Eval leaderboard-combos loading (offline)."""

import inspect
from pathlib import Path

import numpy as np
import pytest
from benchopt.benchmark import Benchmark

BENCHMARK_DIR = Path(__file__).parents[2]
(Dataset,) = Benchmark(BENCHMARK_DIR).check_dataset_patterns(
    ["GiftEval"], class_only=True
)
gifteval = inspect.getmodule(Dataset)


def test_parse_leaderboard_csv(tmp_path):
    csv_path = tmp_path / "leaderboard_combos.csv"
    csv_path.write_text(
        "dataset_name,term\n"
        "m4_weekly/W,short\n"
        "electricity/15T,short\n"
        "electricity/15T,medium\n"
        "electricity/15T,long\n"
    )
    table = gifteval._parse_leaderboard_csv(csv_path)
    assert table == {
        "m4_weekly/W": ("short",),
        "electricity/15T": ("short", "medium", "long"),
    }


def test_non_canonical_combo_skips(monkeypatch):
    monkeypatch.setattr(gifteval, "_leaderboard_cache", {"m4_weekly/W": ("short",)})
    ds = Dataset.get_instance(dataset_name="m4_weekly/W", term="long")
    data = ds.get_data()
    assert set(data) == {"_skip_reason"}
    assert "does not define term 'long'" in data["_skip_reason"]


def test_hf_arrow_directory():
    assert gifteval._hf_arrow_directory("m4_weekly/W") == "m4_weekly"
    assert gifteval._hf_arrow_directory("loop_seattle/H") == "LOOP_SEATTLE/H"
    assert gifteval._hf_arrow_directory("car_parts/M") == "car_parts_with_missing"


@pytest.mark.parametrize(
    "series, expected",
    [
        # forward-fill, leading flat-fill, all-NaN channel → 0.0
        (
            np.array([[np.nan], [1.0], [np.nan], [3.0]]),
            np.array([[1.0], [1.0], [1.0], [3.0]]),
        ),
        (
            np.array([[np.nan], [np.nan], [2.0], [np.nan]]),
            np.array([[2.0], [2.0], [2.0], [2.0]]),
        ),
        (
            np.array([[np.nan], [np.nan]]),
            np.array([[0.0], [0.0]]),
        ),
        # multivariate: channels are filled independently
        (
            np.array([[1.0, np.nan], [np.nan, 5.0], [3.0, np.nan]]),
            np.array([[1.0, 5.0], [1.0, 5.0], [3.0, 5.0]]),
        ),
    ],
)
def test_impute_nan(series, expected):
    out = gifteval._impute_nan(series.astype(np.float64))
    np.testing.assert_allclose(out, expected)


def test_impute_nan_read_only_view():
    # zero-copy arrow views are read-only: _impute_nan must copy
    series = np.array([[np.nan], [1.0], [np.nan]])
    view = series[:]
    view.flags.writeable = False
    out = gifteval._impute_nan(view)
    assert out is not view
    np.testing.assert_allclose(out, [[1.0], [1.0], [1.0]])


@pytest.mark.parametrize(
    "min_len, pred_len, dataset_name, expected",
    [
        (1000, 48, "electricity/15T", 3),
        (1050, 48, "electricity/15T", 3),
        (9600, 48, "electricity/15T", 20),
        (100, 13, "m4_weekly/W", 1),
        (5000, 48, "m4_hourly/H", 1),
        # short series still get at least one window
        (50, 48, "hospital/D", 1),
    ],
)
def test_gifteval_n_windows(min_len, pred_len, dataset_name, expected):
    ds = Dataset.get_instance(
        dataset_name=dataset_name, term="short", n_windows="gifteval"
    )
    series_list = [np.zeros((min_len, 1)) for _ in range(3)]
    n_windows = ds._resolve_n_windows(series_list, pred_len)
    assert n_windows == expected


def test_gifteval_n_windows_int_passthrough():
    ds = Dataset.get_instance(dataset_name="m4_weekly/W", term="short", n_windows=4)
    assert ds._resolve_n_windows([np.zeros((100, 1))], 13) == 4
