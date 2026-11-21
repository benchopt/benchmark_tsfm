"""Tests for the TabPFN-v2 skip (no data or model load required)."""

from pathlib import Path

import pytest
from benchopt.benchmark import Benchmark
from benchopt.utils.run_context import RunContext

BENCHMARK_DIR = Path(__file__).parents[2]
benchmark = Benchmark(BENCHMARK_DIR)
(Solver,) = benchmark.check_solver_patterns(["TabPFN-v2"], class_only=True)
(UCR,) = benchmark.check_dataset_patterns(["UCR"], class_only=True)


@pytest.mark.parametrize(
    "dataset_name, skipped",
    [("ECG5000", False), ("Adiac", True), ("NotInTheTable", False)],
)
def test_skip_reads_n_classes_from_dataset_meta(dataset_name, skipped):
    solver = Solver.get_instance()
    dataset = UCR.get_instance(dataset_name=dataset_name)
    with RunContext().for_run(None, dataset, solver, 0):
        assert (solver.skip() is not None) == skipped
