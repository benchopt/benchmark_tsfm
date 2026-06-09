"""Setup the tests for the benchmark.

In particular, for each test name TEST_NAME, defining test_TEST_NAME will
allow to skip particular configuration combinations that cannot be run
for some reason (e.g. missing API key, or too long to run in CI).
"""

import os

import pytest


def check_test_dataset_get_data(benchmark, dataset_class):
    if dataset_class.name.lower() == "mitdb":
        pytest.xfail("Download timeout due to rate limits")

def check_test_solver_run(benchmark, solver_class, test_dataset_name):
    name = solver_class.name.lower()

    if name == "tfc-api" and os.environ.get("TFC_API_KEY") is None:
        # Skip tfc-api on monash since it doesn't support forecasting yet.
        pytest.skip("No TFC_API_KEY set, so cannot use the tfc-api solver.")

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
