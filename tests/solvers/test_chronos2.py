"""Tests for the Chronos2 solver wiring (no model load required)."""

from pathlib import Path

from benchopt.benchmark import Benchmark

from benchmark_utils.forecasting_constants import CANONICAL_QUANTILE_LEVELS

BENCHMARK_DIR = Path(__file__).parents[2]

(Solver,) = Benchmark(BENCHMARK_DIR).check_solver_patterns(
    ["chronos2"], class_only=True
)


def test_model_ids():
    assert (
        Solver.get_instance(model_size="small").model_id
        == "autogluon/chronos-2-small"
    )
    assert Solver.get_instance(model_size="base").model_id == "amazon/chronos-2"


def test_quantile_levels():
    assert Solver.get_instance(
        model_size="small"
    ).quantile_levels == CANONICAL_QUANTILE_LEVELS
