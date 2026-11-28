"""TimesFM wrapper tests: no model weights, network access, or GPU required."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("benchopt")


class FakeForecaster:
    """Encode channel history, horizon position, and quantile in each output."""

    def __init__(self):
        self.calls = []
        self.configs = []

    def compile(self, config):
        self.configs.append(config)

    def forecast(self, horizon, inputs):
        self.calls.append([x.copy() for x in inputs])
        values = np.stack([
            x[-1] + np.arange(horizon)[:, None] * 10 + np.arange(10)
            for x in inputs
        ])
        # Real TimesFM can mutate the supplied list to pad its final batch.
        inputs.append(np.zeros(3))
        return values[..., 5], values


@pytest.fixture
def solver(monkeypatch):
    fake_module = ModuleType("timesfm")
    fake_module.ForecastConfig = SimpleNamespace
    fake_module.TimesFM_2p5_200M_torch = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "timesfm", fake_module)
    path = Path(__file__).resolve().parents[2] / "solvers" / "timesfm.py"
    spec = importlib.util.spec_from_file_location("_timesfm_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    defaults = {key: values[0] for key, values in module.Solver.parameters.items()}
    instance = module.Solver(**defaults)
    instance.model = FakeForecaster()
    return instance


def test_channel_order_quantiles_and_batch_boundaries(solver):
    solver.batch_size = 2
    inputs = [torch.tensor([[1., 2., 3.], [101., 202., 303.]]),
              torch.tensor([[4.], [404.]])]
    outputs = solver.forecast_batch(inputs, [], 3)
    assert [len(call) for call in solver.model.calls] == [2, 2]
    assert [tuple(x.shape) for x in outputs] == [(3, 3, 9), (3, 1, 9)]
    assert solver.quantile_levels == tuple(q / 10 for q in range(1, 10))
    for output, last_values in zip(outputs, ([101, 202, 303], [404])):
        expected = (
            np.asarray(last_values)[None, :, None]
            + np.arange(3)[:, None, None] * 10
            + np.arange(1, 10)[None, None, :]
        )
        np.testing.assert_array_equal(output.numpy(), expected)


def test_context_truncation_and_partial_batch(solver):
    solver.context_length = 32
    solver.batch_size = 2
    history = torch.arange(80, dtype=torch.float32)
    original = history.clone()
    outputs = solver.forecast_batch([history], [], 2)
    np.testing.assert_array_equal(solver.model.calls[0][0], np.arange(48, 80))
    assert outputs[0].shape == (2, 1, 9)
    assert torch.equal(history, original)


def test_partial_missing_history_is_forwarded_without_future_imputation(solver):
    history = torch.tensor([float("nan"), 1., float("nan"), 2.])
    result = solver.forecast_batch([history], [], 2)
    np.testing.assert_array_equal(solver.model.calls[0][0], history.numpy())
    assert result[0].shape == (2, 1, 9)


def test_model_loading_uses_float32_eval_and_no_torch_compile(solver, monkeypatch):
    model = Mock()
    backend = SimpleNamespace(model=model)
    loader = Mock(return_value=backend)
    monkeypatch.setitem(solver.load_model.__globals__, "TimesFM_2p5_200M_torch",
                        SimpleNamespace(from_pretrained=loader))
    assert solver.load_model("cpu", torch.bfloat16) is backend
    loader.assert_called_once_with(solver.checkpoint, revision=solver.revision,
                                   torch_compile=False)
    model.to.assert_called_once_with(device="cpu", dtype=torch.float32)
    model.to.return_value.eval.assert_called_once()


@pytest.mark.parametrize("task", ["classification", "anomaly_detection",
                                  "event_detection"])
def test_unsupported_tasks_skip(solver, task):
    skipped, reason = solver.skip(task)
    assert skipped
    assert task in reason


@pytest.mark.parametrize("setting,value,horizon,message", [
    ("context_length", 33, 1, "multiple of 32"),
    ("batch_size", 0, 1, "batch_size"),
    ("context_length", 512, 0, "prediction_length"),
    ("context_length", 16352, 1, "rounded horizon"),
    ("context_length", 512, 1025, "continuous quantile head"),
])
def test_invalid_configurations_skip(solver, setting, value, horizon, message):
    setattr(solver, setting, value)
    skipped, reason = solver.skip("forecasting", prediction_length=horizon)
    assert skipped
    assert message in reason


def test_long_horizons_require_explicit_quantile_mode(solver):
    assert solver.skip("forecasting", prediction_length=1024) == (False, None)
    solver.use_continuous_quantile_head = False
    solver.context_length = 512
    assert solver.skip("forecasting", prediction_length=1344) == (False, None)


@pytest.mark.parametrize("history", [
    [], [float("inf")], [float("-inf")], [float("nan")],
    [[1., float("nan")], [2., float("nan")]],
    np.empty((3, 0)), np.zeros((2, 2, 2)),
])
def test_invalid_histories_fail_before_prediction(solver, history):
    with pytest.raises(ValueError, match="history|histories|channel"):
        solver.forecast_batch([torch.tensor(history)], [], 2)
    assert not solver.model.calls


@pytest.mark.parametrize("bad_output", [
    np.zeros((1, 2, 9)), np.full((1, 2, 10), np.nan),
    np.full((1, 2, 10), np.inf),
])
def test_invalid_predictions_fail(solver, monkeypatch, bad_output):
    monkeypatch.setattr(solver.model, "forecast",
                        lambda **kwargs: (None, bad_output))
    with pytest.raises(ValueError, match="returned|non-finite"):
        solver.forecast_batch([torch.ones(4, 1)], [], 2)


def test_objective_reconfigures_horizon_without_reloading(solver, monkeypatch):
    backend = solver.model
    loads = []

    def load(device, dtype):
        loads.append(device)
        return backend

    monkeypatch.setattr(solver, "load_model", load)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for horizon in (12, 48):
        solver.set_objective([], None, "forecasting", prediction_length=horizon)
    assert loads == ["cpu"]
    assert [config.max_horizon for config in backend.configs] == [1024, 1024]
    assert solver.dtype == torch.float32
    assert all(c.infer_is_positive and c.force_flip_invariance
               for c in backend.configs)


def test_default_uses_adaptive_context_and_reference_settings(solver):
    assert solver.context_length == 15360
    solver.batch_size = 2
    history = torch.arange(16000, dtype=torch.float32)
    outputs = solver.forecast_batch([torch.ones(33), torch.ones(65), history], [], 4)
    assert [c.max_context for c in solver.model.configs] == [96, 15360]
    for config in solver.model.configs:
        assert config.max_horizon == 1024
        assert config.infer_is_positive and config.force_flip_invariance
        assert config.normalize_inputs and config.fix_quantile_crossing
    assert [tuple(x.shape) for x in outputs] == [(4, 1, 9)] * 3
    np.testing.assert_array_equal(solver.model.calls[-1][0], np.arange(640, 16000))


def test_context_reserves_the_full_continuous_head(solver):
    solver.context_length = 15392
    assert solver.skip("forecasting", prediction_length=1)[0]
    solver.context_length = 15360
    assert solver.skip("forecasting", prediction_length=1024) == (False, None)


def test_autoregressive_quantiles_prepare_the_requested_horizon(solver):
    solver.context_length = 512
    solver.use_continuous_quantile_head = False
    solver.forecast_batch([torch.ones(65)], [], 1344)
    config = solver.model.configs[-1]
    assert config.max_horizon == 1344
    assert config.max_context == 96
    assert not config.use_continuous_quantile_head
    assert config.infer_is_positive and config.force_flip_invariance


def test_long_context_preserves_future_cutoff(solver):
    from benchmark_utils.inputs import ForecastInput

    solver.task = "forecasting"
    solver.meta = {"prediction_length": 3}
    solver.run(None)
    model = solver.get_result()["model"]
    series = np.arange(900, dtype=np.float32)[:, None]
    baseline = model.predict(ForecastInput(x=[series], cutoff_indexes=[[700]]))
    changed = series.copy()
    changed[700:] = -999
    altered = model.predict(ForecastInput(x=[changed], cutoff_indexes=[[700]]))
    np.testing.assert_array_equal(baseline.quantiles[0], altered.quantiles[0])
    assert all(len(x) == 700 for call in solver.model.calls for x in call)


def test_future_changes_do_not_affect_either_cutoff(solver):
    from benchmark_utils.inputs import ForecastInput

    solver.task = "forecasting"
    solver.meta = {"prediction_length": 3}
    solver.run(None)
    adapter = solver.get_result()["model"]
    series = np.arange(30, dtype=np.float32)[:, None]
    cutoffs = [10, 20]
    baseline = adapter.predict(ForecastInput(x=[series], cutoff_indexes=[cutoffs]))
    for position, cutoff in enumerate(cutoffs):
        changed = series.copy()
        changed[cutoff:] = -999
        output = adapter.predict(ForecastInput(x=[changed], cutoff_indexes=[cutoffs]))
        np.testing.assert_array_equal(
            output.quantiles[0][position], baseline.quantiles[0][position]
        )
