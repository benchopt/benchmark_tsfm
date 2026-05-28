"""Shape tests for the Chronos encoder variants."""

import numpy as np
import pytest

import chronos
import torch

from solvers.chronos import ChronosEncoder  # noqa: E402


@pytest.fixture(scope="module")
def pipeline():
    return chronos.ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map="cpu",
        torch_dtype=torch.float32,
    )


@pytest.mark.parametrize("layer", [None, 0, -1])
@pytest.mark.parametrize("n_channels", [1, 3])
def test_encode_shape(pipeline, layer, n_channels):
    T = 64
    x = np.random.randn(T, n_channels).astype(np.float32)
    emb = ChronosEncoder(pipeline, layer=layer).encode(x)
    d_model = pipeline.model.model.config.d_model
    assert emb.shape == (T + 1, n_channels, d_model)
