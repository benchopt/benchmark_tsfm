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
