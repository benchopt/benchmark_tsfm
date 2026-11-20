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


def test_every_fev_bench_task_has_hf_data():
    assert {t["dataset_config"] for t in fev._fev_bench_tasks().values()} <= set(
        fev._HF_PATHS
    )


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
    })
    static = fev._static_covariates(df, ["store"])
    assert [s.tolist() for s in static] == [[0.0], [1.0], [0.0]]
    dynamic = fev._dynamic_covariates(df, ["promo", "holiday"])
    assert [a.shape for a in dynamic] == [(2, 2), (3, 2), (1, 2)]
    np.testing.assert_array_equal(dynamic[1], [[1, 1], [1, 1], [0, 2]])
    assert fev._dynamic_covariates(df, None) is None
