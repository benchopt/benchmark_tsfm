"""Tests for the TimEE solver (no model download required)."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("timee")

from benchopt.benchmark import Benchmark  # noqa: E402

(Solver,) = Benchmark(Path(__file__).parents[2]).check_solver_patterns(
    ["timee"], class_only=True
)


class _StubClassifier:
    def __init__(self, transforms=("interp256",)):
        self.transforms = transforms
        self.calls = []

    def predict(self, X_train, y_train, X_test):
        self.calls.append((X_train, y_train, X_test))
        return np.full(len(X_test), y_train[0]), np.ones((len(X_test), 1))


def test_model_id():
    assert Solver.get_instance().model_id == "liamsbhoo/timee:ensemble=True"
    mv = Solver.get_instance(
        checkpoint="liamsbhoo/timee-multivariate", use_ensemble=False
    )
    assert mv.model_id == "liamsbhoo/timee-multivariate:ensemble=False"


def test_adapter_shapes_and_ragged_padding():
    stub = _StubClassifier()
    solver = object.__new__(Solver)
    solver.X_train = [np.arange(4, dtype=np.float64), np.arange(6, dtype=np.float64)]
    solver.y_train = np.array([2, 0])
    preds = solver.build_adapter("classification", stub).predict([np.ones((5, 1))])
    tr_x, tr_y, te_x = stub.calls[0]
    assert tr_x.shape == (2, 1, 6) and tr_x.dtype == np.float32
    assert np.isnan(tr_x[0, 0, :2]).all()
    assert te_x.shape == (1, 1, 5)
    np.testing.assert_array_equal(tr_y, [2, 0])
    np.testing.assert_array_equal(preds, [2])


def test_adapter_aligns_lengths_without_ensemble():
    stub = _StubClassifier(transforms=[])
    solver = object.__new__(Solver)
    solver.X_train = [np.ones((6, 2), dtype=np.float32)]
    solver.y_train = np.array([1])
    preds = solver.build_adapter("classification", stub).predict(
        [np.ones((10, 2), dtype=np.float32)]
    )
    tr_x, _, te_x = stub.calls[0]
    assert tr_x.shape == (1, 2, 10) and te_x.shape == (1, 2, 10)
    assert np.isnan(tr_x[:, :, :4]).all()
    assert preds.shape == (1,)
