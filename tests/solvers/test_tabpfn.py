"""Shape and behaviour tests for the TabPFN-v2 solver."""

import importlib.util
import sys
import types
from enum import Enum
from pathlib import Path

import numpy as np


def _make_fake_tabpfn_module():
    """Build a lightweight fake ``tabpfn`` module for offline testing."""
    fake = types.ModuleType("tabpfn")
    fake_constants = types.ModuleType("tabpfn.constants")

    class ModelVersion(Enum):
        V2 = "v2"
        V2_5 = "v2_5"
        V2_6 = "v2_6"
        V3 = "v3"

    # The solver only uses V2; keep the full enum for completeness.

    fake_constants.ModelVersion = ModelVersion
    fake.constants = fake_constants

    class FakeTabPFNClassifier:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.fit_X = None
            self.fit_y = None
            self.predict_calls = []

        @classmethod
        def create_default_for_version(cls, version, **kwargs):
            obj = cls(**kwargs)
            obj._version = version
            return obj

        def fit(self, X, y):
            self.fit_X = np.asarray(X)
            self.fit_y = np.asarray(y)
            return self

        def predict(self, X):
            X_arr = np.asarray(X)
            self.predict_calls.append(X_arr.copy())
            return np.zeros(X_arr.shape[0], dtype=np.int64)

    fake.TabPFNClassifier = FakeTabPFNClassifier
    return fake, fake_constants


def _make_fake_benchopt_module():
    """Minimal fake ``benchopt`` module — only BaseSolver is needed."""
    fake = types.ModuleType("benchopt")

    class BaseSolver:
        pass

    fake.BaseSolver = BaseSolver
    return fake


def _load_solver(monkeypatch):
    """Import ``solvers/tabpfn.py`` with fake modules injected."""
    fake_tabpfn, fake_constants = _make_fake_tabpfn_module()
    monkeypatch.setitem(sys.modules, "benchopt", _make_fake_benchopt_module())
    monkeypatch.setitem(sys.modules, "tabpfn", fake_tabpfn)
    monkeypatch.setitem(sys.modules, "tabpfn.constants", fake_constants)
    solver_path = (
        Path(__file__).resolve().parents[2] / "solvers" / "tabpfn.py"
    )
    spec = importlib.util.spec_from_file_location(
        "tabpfn_solver_under_test", solver_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_solver(module):
    """Instantiate Solver and inject default benchopt parameter attributes."""
    solver = module.Solver()
    solver.n_estimators = 8
    return solver


def test_tabpfn_solver_fit_shapes(monkeypatch):
    """_to_tabular stacks (T, 1) lists correctly; TabPFN receives (N, T)."""
    module = _load_solver(monkeypatch)
    solver = _make_solver(module)

    T = 6
    X_train = [np.arange(T, dtype=np.float32)[:, None] + i for i in range(4)]
    y_train = np.array([0, 1, 0, 1], dtype=np.int64)

    solver.set_objective("classification", X_train, y_train)
    solver.run(None)

    assert solver._classifier.fit_X.shape == (4, T)
    assert solver._classifier.fit_y.shape == (4,)


def test_tabpfn_solver_predict_shapes(monkeypatch):
    """Adapter.predict converts the test list to (N, T) before TabPFN."""
    module = _load_solver(monkeypatch)
    solver = _make_solver(module)

    T = 6
    X_train = [np.arange(T, dtype=np.float32)[:, None] + i for i in range(4)]
    y_train = np.array([0, 1, 0, 1], dtype=np.int64)

    solver.set_objective("classification", X_train, y_train)
    solver.run(None)

    X_test = [np.arange(T, dtype=np.float32)[:, None] + i for i in range(3)]
    adapter = solver.get_result()["model"]
    y_pred = adapter.predict(X_test)

    assert y_pred.shape == (3,)
    assert solver._classifier.predict_calls[-1].shape == (3, T)


def test_tabpfn_multivariate_shapes(monkeypatch):
    """_to_tabular flattens (T, C) into T*C features for multivariate data."""
    module = _load_solver(monkeypatch)
    solver = _make_solver(module)

    T, C = 5, 3
    X_train = [np.random.randn(T, C).astype(np.float32) for _ in range(4)]
    y_train = np.array([0, 1, 0, 1], dtype=np.int64)

    solver.set_objective("classification", X_train, y_train)
    solver.run(None)

    assert solver._classifier.fit_X.shape == (4, T * C)

    X_test = [np.random.randn(T, C).astype(np.float32) for _ in range(3)]
    adapter = solver.get_result()["model"]
    y_pred = adapter.predict(X_test)

    assert y_pred.shape == (3,)
    assert solver._classifier.predict_calls[-1].shape == (3, T * C)


def test_tabpfn_classifier_reuse_across_datasets(monkeypatch):
    """Classifier is reused when n_estimators is unchanged across datasets."""
    module = _load_solver(monkeypatch)
    solver = _make_solver(module)

    T = 6
    X = [np.arange(T, dtype=np.float32)[:, None]]
    y = np.array([0])

    solver.set_objective("classification", X, y)
    first_clf = solver._classifier

    solver.set_objective("classification", X, y)
    assert solver._classifier is first_clf


def test_tabpfn_solver_skip(monkeypatch):
    """Unsupported tasks are skipped with an informative message."""
    module = _load_solver(monkeypatch)
    solver = _make_solver(module)

    for task in ("forecasting", "anomaly_detection", "event_detection"):
        should_skip, reason = solver.skip(task)
        assert should_skip
        assert f"task={task!r}" in reason
