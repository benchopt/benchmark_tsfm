"""Composable encoder: ``UnpooledEncoder`` + ``Pooler`` -> 1-D feature vector.

Shape conventions used below: ``B`` batch, ``T`` time / tokens,
``V`` variates (channels), ``D`` embedding dim.

This module defines:

- :class:`UnpooledEncoder` — ABC for frozen feature extractors that return
  per-token embeddings of shape ``(B, T, V, D)`` (no pooling);
- :class:`BasePooler` and three concrete reducers (mean / max / last) over
  the time axis;
- :class:`Encoder`, which composes an unpooled encoder with a pooler and
  flattens variates & embedding dim into a single feature vector per
  series, ready for sklearn-style linear heads.
"""

from abc import ABC, abstractmethod

import numpy as np


class UnpooledEncoder(ABC):
    """Frozen feature extractor returning *unpooled* embeddings.

    Subclasses must implement ``encode``. Implementations are expected
    to handle a batch of series and return embeddings with a leading
    batch axis; single samples are promoted to a batch of size one.
    """

    @abstractmethod
    def encode(self, X) -> np.ndarray:
        """Map a batch of time series to embedding sequences.

        Parameters
        ----------
        X : np.ndarray
            Single sample of shape ``(T, V)`` or a batch of shape
            ``(B, T, V)``.

        Returns
        -------
        np.ndarray, shape (B, T, V, D)
            Per-token, per-variate embeddings (``B=1`` for a single sample).
        """


class BasePooler(ABC):
    """Reduce a per-token embedding sequence over the time axis."""

    @abstractmethod
    def pool(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce over the time axis (``axis=-3``).

        Parameters
        ----------
        embeddings : np.ndarray, shape ``(..., T, V, D)``

        Returns
        -------
        np.ndarray, shape ``(..., V, D)``
        """


class MeanPooler(BasePooler):
    """Average over the time axis."""

    def pool(self, embeddings: np.ndarray) -> np.ndarray:
        return embeddings.mean(axis=-3)


class MaxPooler(BasePooler):
    """Element-wise max over the time axis."""

    def pool(self, embeddings: np.ndarray) -> np.ndarray:
        return embeddings.max(axis=-3)


class LastPooler(BasePooler):
    """Take the last token in the sequence (e.g. EOS for Chronos)."""

    def pool(self, embeddings: np.ndarray) -> np.ndarray:
        return embeddings[..., -1, :, :]


class Encoder:
    """Frozen feature extractor: :class:`UnpooledEncoder` + :class:`BasePooler`.

    Composes a sequence encoder (``(B, T, V, D)``) with a pooler
    (reduces over ``T``) and flattens variates & embedding dim into a
    feature vector per series.

    Parameters
    ----------
    base_encoder : UnpooledEncoder
    pooler : BasePooler
    """

    def __init__(self, base_encoder: UnpooledEncoder, pooler: BasePooler):
        self.base_encoder = base_encoder
        self.pooler = pooler

    def encode(self, X) -> np.ndarray:
        """Encode a batch of time series to feature vectors.

        ``(B, T, V) -> (B, V * D)``. A single ``(T, V)`` sample is
        promoted by the underlying encoder to ``B=1``.
        """
        embeddings = self.base_encoder.encode(X)   # (B, T, V, D)
        pooled = self.pooler.pool(embeddings)      # (B, V, D)
        return pooled.reshape(pooled.shape[0], -1)  # (B, V * D)
