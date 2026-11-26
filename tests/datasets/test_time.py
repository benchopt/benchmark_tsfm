"""Offline tests for the TIME dataset."""

import inspect
from pathlib import Path

import numpy as np
import pyarrow as pa
from benchopt.benchmark import Benchmark


BENCHMARK_DIR = Path(__file__).parents[2]
(Dataset,) = Benchmark(BENCHMARK_DIR).check_dataset_patterns(["TIME"], class_only=True)
time_ds = inspect.getmodule(Dataset)


def _write_arrow(path, targets, freq):
    """Write a tiny HF-style (IPC stream) arrow file."""
    table = pa.table(
        {
            "item_id": [f"item_{i}" for i in range(len(targets))],
            "freq": [freq] * len(targets),
            "target": targets,
        }
    )
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    return str(path)


def test_undefined_term_skips(monkeypatch):
    monkeypatch.setattr(
        time_ds,
        "_tasks_cache",
        {"Crypto/D": dict(test_length=273, terms={"short": 30})},
    )
    ds = Dataset.get_instance(dataset_name="Crypto/D", term="long")
    data = ds.get_data()
    assert set(data) == {"_skip_reason"}


def test_get_data_end_to_end(monkeypatch, tmp_path):
    a = np.arange(600, dtype=np.float32).reshape(2, 300)
    f = _write_arrow(tmp_path / "data-00000-of-00001.arrow", [a.tolist()], "D")
    monkeypatch.setattr(
        time_ds,
        "_tasks_cache",
        {"Crypto/D": dict(test_length=273, terms={"short": 30})},
    )
    monkeypatch.setattr(Dataset, "_snapshot", lambda self: [f])
    data = Dataset.get_instance(dataset_name="Crypto/D", term="short").get_data()
    assert data["prediction_length"] == 30
    assert data["freq"] == "D"
    assert data["task"] == "forecasting"
    assert data["cutoff_indexes"] == [[27 + 30 * w for w in range(9)]]
    assert data["X_train"][0].shape == (27, 2)
    assert data["y_test"][0].shape == (9, 30, 2)
    assert data["metrics"] == list(time_ds.TIME_METRIC_NAMES)
    assert data["metrics"][0] == "time_mase"  # the objective's ``value``
    assert data["seasonality"] == 1  # gluonts: D -> 1 (from_pandas says 7)


def test_gluonts_seasonality():
    # gluonts 0.15.1 (pinned upstream) values for the freqs of the TIME
    # configs; each equals the ``seasonality`` in upstream TIME-Output.
    expected = {
        "5T": 288, "10T": 144, "15T": 96, "20T": 72, "30T": 48, "H": 24,
        "D": 1, "B": 5, "W": 1, "M": 12, "Q": 4,
    }  # fmt: skip
    assert {f: time_ds.gluonts_seasonality(f) for f in expected} == expected
