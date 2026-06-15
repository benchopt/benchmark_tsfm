"""Setup the tests for the benchmark.

In particular, for each test name TEST_NAME, defining test_TEST_NAME will
allow to skip particular configuration combinations that cannot be run
for some reason (e.g. missing API key, or too long to run in CI).
"""

import os

import pytest


# These datasets load fine locally, but their download hosts block / rate-limit
# CI runners (ucr: timeseriesclassification.com -> HTTP 401; mitdb: download
# timeout). So we run them locally and skip them *only in CI*.
_CI_FLAKY_DATASETS = {"ucr", "mitdb"}


def _skip_flaky_in_ci(name):
    if name.lower() in _CI_FLAKY_DATASETS and os.environ.get("CI"):
        pytest.skip(
            f"{name} download is blocked/rate-limited from CI runners "
            "(runs locally)."
        )


def check_test_dataset_get_data(benchmark, dataset_class):
    _skip_flaky_in_ci(dataset_class.name)


def check_test_benchmark_objective(benchmark, test_dataset_name):
    _skip_flaky_in_ci(test_dataset_name)


def check_test_solver_run(benchmark, solver_class, test_dataset_name):
    name = solver_class.name.lower()

    if name == "tfc-api" and os.environ.get("TFC_API_KEY") is None:
        # Skip tfc-api on monash since it doesn't support forecasting yet.
        pytest.skip("No TFC_API_KEY set, so cannot use the tfc-api solver.")

    _skip_flaky_in_ci(test_dataset_name)

    if name == "chronos2" and not _cuda_available():
        # Chronos-2's encoder produces non-finite (NaN) embeddings on CPU, which
        # the linear probe now rejects. It is a GPU model, so skip it when no
        # CUDA device is available (e.g. CPU-only CI runners).
        pytest.skip("Chronos2 requires CUDA (non-finite embeddings on CPU).")


def _cuda_available():
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()
