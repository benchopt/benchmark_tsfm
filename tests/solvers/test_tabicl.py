"""Shape and behaviour tests for the TabICL-v2 solver."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def _make_fake_tabicl_module():
    """Build a lightweight fake ``tabicl`` module for offline testing."""
    fake = types.ModuleType("tabicl")

    class FakeTabICLClassifier:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.fit_X = None
            self.fit_y = None
            self.predict_calls = []

        def fit(self, X, y):
            self.fit_X = np.asarray(X)
            self.fit_y = np.asarray(y)
            return self

        def predict(self, X):
            X_arr = np.asarray(X)
            self.predict_calls.append(X_arr.copy())
            return np.zeros(X_arr.shape[0], dtype=np.int64)

    fake.TabICLClassifier = FakeTabICLClassifier
    return fake


def _make_fake_benchopt_module():
    """Minimal fake ``benchopt`` module — only BaseSolver is needed."""
    fake = types.ModuleType("benchopt")

    class BaseSolver:
        pass

    fake.BaseSolver = BaseSolver
    return fake


def _make_fake_torch_module():
    """Minimal fake ``torch`` module — only cuda.is_available is needed."""
    fake = types.ModuleType("torch")
    fake_cuda = types.ModuleType("torch.cuda")
    fake_cuda.is_available = lambda: False
    fake.cuda = fake_cuda
    return fake


def _load_solver(monkeypatch):
    """Import ``solvers/tabicl.py`` with fake modules injected."""
    monkeypatch.setitem(sys.modules, "benchopt", _make_fake_benchopt_module())
    monkeypatch.setitem(sys.modules, "torch", _make_fake_torch_module())
    monkeypatch.setitem(sys.modules, "tabicl", _make_fake_tabicl_module())
    solver_path = (
        Path(__file__).resolve().parents[2] / "solvers" / "tabicl.py"
    )
    spec = importlib.util.spec_from_file_location(
        "tabicl_solver_under_test", solver_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_solver(module):
    """Instantiate Solver and inject default benchopt parameter attributes."""
    solver = module.Solver()
    solver.n_estimators = 8
    solver.checkpoint_version = "tabicl-classifier-v2-20260212.ckpt"
    return solver


def test_tabicl_solver_fit_shapes(monkeypatch):
    """_to_tabular stacks (T, 1) lists correctly; TabICL receives (N, T)."""
    module = _load_solver(monkeypatch)
    solver = _make_solver(module)

    T = 6
    X_train = [np.arange(T, dtype=np.float32)[:, None] + i for i in range(4)]
    y_train = np.array([0, 1, 0, 1], dtype=np.int64)

    solver.set_objective("classification", X_train, y_train)
    solver.run(None)

    assert solver._classifier.fit_X.shape == (4, T)
    assert solver._classifier.fit_y.shape == (4,)


def test_tabicl_solver_predict_shapes(monkeypatch):
    """Adapter.predict converts the test list to (N, T) before TabICL."""
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


def test_tabicl_classifier_reuse_across_datasets(monkeypatch):
    """Classifier is reused when checkpoint_version and n_estimators are unchanged."""
    module = _load_solver(monkeypatch)
    solver = _make_solver(module)

    T = 6
    X = [np.arange(T, dtype=np.float32)[:, None]]
    y = np.array([0])

    solver.set_objective("classification", X, y)
    first_clf = solver._classifier

    solver.set_objective("classification", X, y)
    assert solver._classifier is first_clf


def test_tabicl_solver_skip(monkeypatch):
    """Unsupported tasks are skipped with an informative message."""
    module = _load_solver(monkeypatch)
    solver = _make_solver(module)

    for task in ("forecasting", "anomaly_detection", "event_detection"):
        should_skip, reason = solver.skip(task)
        assert should_skip
        assert f"task={task!r}" in reason
