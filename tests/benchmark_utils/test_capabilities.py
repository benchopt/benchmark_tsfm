"""Tests for the covariate-masking helper."""

import numpy as np
import pytest

from benchmark_utils.capabilities import (
    FUTURE_COVARIATES,
    HIST_COVARIATES,
    mask_covariates,
)
from benchmark_utils.covariates import Covariates


@pytest.fixture
def covariates():
    return Covariates(
        static_covars=[np.array([1.0, 2.0])],
        hist_covars=[np.zeros((10, 1))],
        future_covars=[np.ones((10, 2))],
    )


def test_empty_active_drops_both_covariates(covariates):
    masked = mask_covariates(covariates, frozenset())
    assert masked.hist_covars == []
    assert masked.future_covars == []
    # static is not part of the vocabulary — passed through untouched.
    assert masked.static_covars is covariates.static_covars


def test_hist_only_keeps_hist_drops_future(covariates):
    masked = mask_covariates(covariates, {HIST_COVARIATES})
    assert masked.hist_covars is covariates.hist_covars
    assert masked.future_covars == []


def test_future_only_keeps_future_drops_hist(covariates):
    masked = mask_covariates(covariates, {FUTURE_COVARIATES})
    assert masked.future_covars is covariates.future_covars
    assert masked.hist_covars == []


def test_both_active_preserves_all(covariates):
    masked = mask_covariates(covariates, {HIST_COVARIATES, FUTURE_COVARIATES})
    assert masked.hist_covars is covariates.hist_covars
    assert masked.future_covars is covariates.future_covars


def test_input_not_mutated(covariates):
    mask_covariates(covariates, frozenset())
    # Original still intact.
    assert len(covariates.hist_covars) == 1
    assert len(covariates.future_covars) == 1
