"""Shape tests for poolers and the composed Encoder."""

import numpy as np
import pytest

from benchmark_utils.adapters import (
    Encoder,
    LastPooler,
    MaxPooler,
    MeanPooler,
    UnpooledEncoder,
)


class _DummyEncoder(UnpooledEncoder):
    """Deterministic ``(T_tok, C, D)`` output so we can check pooler math."""

    def __init__(self, T_tok=10, C=2, D=4):
        self.T_tok, self.C, self.D = T_tok, C, D

    def encode(self, x):
        return np.arange(
            self.T_tok * self.C * self.D, dtype=np.float32
        ).reshape(self.T_tok, self.C, self.D)


@pytest.fixture
def dummy():
    return _DummyEncoder()


@pytest.mark.parametrize("pooler_cls", [MeanPooler, MaxPooler, LastPooler])
def test_pooler_output_shape(dummy, pooler_cls):
    emb = dummy.encode(np.zeros((16, dummy.C)))
    pooled = pooler_cls().pool(emb)
    assert pooled.shape == (dummy.C, dummy.D)


@pytest.mark.parametrize("pooler_cls", [MeanPooler, MaxPooler, LastPooler])
def test_encoder_output_is_flat_feature_vector(dummy, pooler_cls):
    enc = Encoder(dummy, pooler_cls())
    out = enc.encode(np.zeros((16, dummy.C)))
    assert out.shape == (dummy.C * dummy.D,)
