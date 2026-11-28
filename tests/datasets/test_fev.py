"""Tests for the FEV dataset helpers (freq inference, channel selection)."""

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchopt.benchmark import Benchmark

BENCHMARK_DIR = Path(__file__).parents[2]
Dataset, = Benchmark(BENCHMARK_DIR).check_dataset_patterns(
    ["FEV"], class_only=True
)
fev = inspect.getmodule(Dataset)


class TestInferFreq:
    def test_regular_hourly(self):
        ts = pd.date_range("2024-01-01", periods=10, freq="h")
        assert fev._infer_freq(ts).upper().startswith("H")

    def test_gap_in_first_points_uses_delta_mode(self):
        # One missing stamp among the first 5 breaks pd.infer_freq; the
        # most-common-delta fallback should still find hourly.
        ts = pd.date_range("2024-01-01", periods=10, freq="h").delete(2)
        assert fev._infer_freq(ts).upper().startswith("H")

    def test_uninferable_warns_and_defaults_to_daily(self):
        with pytest.warns(UserWarning, match="defaulting to daily"):
            assert fev._infer_freq(pd.DatetimeIndex([])) == "D"


class TestChannelSelection:
    def test_target_column_is_sole_channel(self):
        # epf_*-style schema: target + numeric covariate columns.
        df = pd.DataFrame({
            "id": ["a"],
            "timestamp": [np.array(["2024-01-01"], dtype="datetime64[ns]")],
            "target": [np.array([1.0, 2.0])],
            "Load Forecast": [np.array([3.0, 4.0])],
        })
        assert fev._select_channel_cols(df) == ["target"]

    def test_no_target_stacks_numeric_array_cols(self):
        # ETT-style schema: one column per channel, no target.
        df = pd.DataFrame({
            "id": ["a"],
            "timestamp": [np.array(["2024-01-01"], dtype="datetime64[ns]")],
            "HUFL": [np.array([1.0])],
            "OT": [np.array([2.0])],
            "type": ["metadata-string"],
            "holidays": [np.array(["xmas"], dtype=object)],
        })
        assert fev._select_channel_cols(df) == ["HUFL", "OT"]

    def test_empty_first_row_does_not_drop_channel(self):
        df = pd.DataFrame({
            "id": ["a", "b"],
            "timestamp": [None, None],
            "OT": [np.array([]), np.array([1.0, 2.0])],
        })
        assert fev._select_channel_cols(df) == ["OT"]


@pytest.mark.parametrize(
    "target, expected", [("sales", ["sales"]), (["a", "b"], ["a", "b"])]
)
def test_task_target_gives_channels(target, expected):
    df = pd.DataFrame({"id": ["a"], "target": [np.array([1.0])]})
    assert fev._select_channel_cols(df, target) == expected


@pytest.mark.parametrize(
    "config, path",
    [
        ("ETT_1H", "ETT/1H"),
        ("uk_covid_nation_1D", "uk_covid_nation/1D"),
        ("LOOP_SEATTLE_5T", "LOOP_SEATTLE/5T"),
        ("boomlet_1062", "boomlet/1062"),
        ("fred_md_2025", "fred_md_2025"),
        ("proenfo_gfc12", "proenfo_gfc12"),
    ],
)
def test_hf_path_of_dataset_config(config, path):
    assert fev._HF_SUBDIR.sub("/", config) == path


def test_covariate_columns_are_not_channels():
    df = pd.DataFrame({
        "id": ["a"],
        "load": [np.array([1.0, 2.0])],
        "temperature": [np.array([3.0, 4.0])],
    })
    assert fev._select_channel_cols(df, None, ["temperature"]) == ["load"]


def test_covariates_encode_categories_as_codes():
    df = pd.DataFrame({
        "store": ["x", "y", "x"],
        "promo": [np.array([0, 1]), np.array([1, 1, 0]), np.array([0])],
        "holiday": [np.array(["a", "0"]), np.array(["0", "0", "b"]), np.array(["a"])],
        "snap": [np.array([True, False]), np.array([False] * 3), np.array([True])],
        "code": [np.array(["1", "2"]), np.array(["2", "2", "1"]), np.array(["1"])],
    })
    static = fev._static_covariates(df, ["store"])
    assert [s.tolist() for s in static] == [[0.0], [1.0], [0.0]]
    dynamic, categorical = fev._dynamic_covariates(df, ["promo", "holiday"])
    assert [a.shape for a in dynamic] == [(2, 2), (3, 2), (1, 2)]
    # categories coded in sorted order: "0", "a", "b"
    np.testing.assert_array_equal(dynamic[1], [[1, 0], [1, 0], [0, 2]])
    assert categorical.tolist() == [False, True]
    # as fev, only numeric dtypes are numeric: booleans and digit strings are not
    dynamic, categorical = fev._dynamic_covariates(df, ["snap", "code"])
    np.testing.assert_array_equal(dynamic[0], [[1, 0], [0, 1]])
    assert categorical.tolist() == [True, True]
    assert fev._dynamic_covariates(df, None) == (None, None)


@pytest.mark.parametrize(
    "seasonality, freq, expected",
    [
        (1, "D", 7),
        (1, "MS", 12),
        (1, "QS-OCT", 4),
        (1, "W-SUN", 1),
        (96, "15min", 96),
        (1, "2h", 12),
        (1, "7h", 1),
        # longer than the baseline context: no season
        (1, "min", 1),
    ],
)
def test_baseline_seasonality(seasonality, freq, expected):
    assert fev._baseline_seasonality(seasonality, freq) == expected
