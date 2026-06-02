"""Setup the tests for the benchmark.

In particular, for each test name TEST_NAME, defining test_TEST_NAME will
allow to skip particular configuration combinations that cannot be run
for some reason (e.g. missing API key, or too long to run in CI).
"""

import os

import pytest


def check_test_solver_run(benchmark, solver_class, test_dataset_name):
    if solver_class.name.lower() == "tfc-api" and os.environ.get("TFC_API_KEY") is None:
        # Skip tfc-api on monash since it doesn't support forecasting yet.
        pytest.skip("No TFC_API_KEY set, so cannot use the tfc-api solver.")
