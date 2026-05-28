"""Composable encoder: ``UnpooledEncoder`` + ``Pooler`` -> 1-D feature vector.

The :class:`UnpooledEncoder` ABC (in :mod:`.base`) exposes per-token
embeddings of shape ``(T_tok, C, D)``.  This module adds:

- :class:`BasePooler` and three concrete reducers (mean / max / last) over
  the time-token axis;
- :class:`Encoder`, which composes an unpooled encoder with a pooler and
  flattens channels & dim into a single 1-D feature vector, ready for
  sklearn-style linear heads.
"""

from abc import ABC, abstractmethod

import numpy as np

from .base import UnpooledEncoder


class BasePooler(ABC):
    """Reduce a per-token embedding sequence over the time-token axis."""

    @abstractmethod
    def pool(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce over axis 0.

        Parameters
        ----------
        embeddings : np.ndarray, shape (T_tok, C, D)

        Returns
        -------
        np.ndarray, shape (C, D)
        """


class MeanPooler(BasePooler):
    """Average over the time-token axis."""

    def pool(self, embeddings: np.ndarray) -> np.ndarray:
        return embeddings.mean(axis=0)


class MaxPooler(BasePooler):
    """Element-wise max over the time-token axis."""

    def pool(self, embeddings: np.ndarray) -> np.ndarray:
        return embeddings.max(axis=0)


class LastPooler(BasePooler):
    """Take the last token in the sequence (e.g. EOS for Chronos)."""

    def pool(self, embeddings: np.ndarray) -> np.ndarray:
        return embeddings[-1]


class Encoder:
    """Frozen feature extractor: :class:`UnpooledEncoder` + :class:`BasePooler`.

    Composes a sequence encoder (``(T_tok, C, D)``) with a pooler
    (reduces over ``T_tok``) and flattens channels & dim into a single
    1-D feature vector.

    Parameters
    ----------
    base_encoder : UnpooledEncoder
    pooler : BasePooler
    """

    def __init__(self, base_encoder: UnpooledEncoder, pooler: BasePooler):
        self.base_encoder = base_encoder
        self.pooler = pooler

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode one time series to a 1-D feature vector.

        ``(T, C) -> (C * D,)``.
        """
        embeddings = self.base_encoder.encode(x)   # (T_tok, C, D)
        pooled = self.pooler.pool(embeddings)      # (C, D)
        return pooled.reshape(-1)                  # (C * D,)
