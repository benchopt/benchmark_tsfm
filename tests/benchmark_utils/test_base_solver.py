"""BaseTSFMSolver keeps its model across datasets (benchopt ``keep_state``)."""

import inspect

import pytest
from benchopt.utils.parametrized_name_mixin import ParametrizedNameMixin

from benchmark_utils.base_solver import BaseTSFMSolver

pytestmark = pytest.mark.skipif(
    "soft" not in inspect.signature(ParametrizedNameMixin._release_state).parameters,
    reason="benchopt without keep_state",
)


class _Solver(BaseTSFMSolver):
    name = "keep-state-test"
    supported_tasks = {"forecasting"}
    model_id = "m"

    def load_model(self, device, dtype):
        return object()


def test_model_kept_across_datasets_freed_with_solver():
    solver = _Solver.get_instance()
    solver.model, solver._loaded_model_id = object(), "m"
    model = solver.model
    # Same solver on another dataset: the model is kept, the data is not.
    solver.X_train = [0]
    solver._release_state(soft=True)
    assert solver.model is model and solver._loaded_model_id == "m"
    assert not hasattr(solver, "X_train")
    # The runs moved to another solver: everything is freed.
    solver._release_state()
    assert solver.model is None and solver._loaded_model_id is None
