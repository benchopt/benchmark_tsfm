"""Event-detection adapter for a frozen Chronos encoder + trainable head.

Architecture
------------
x (T, C)
  -> Chronos.embed per channel  (T_tok, D) x C
  -> mean-pool across channels  (T_tok, D)        [memory key/value]
  -> EventHead Transformer-decoder                 [10 learned queries]
  -> pos_head  : Linear(D,2) -> sigmoid  (start, length) in [0,1]
  -> cls_head  : Linear(D,k) -> logits   k binary class logits

Output shape per series: (N=10, 2+k)

Channel handling
----------------
Chronos is strictly univariate. We embed every channel independently and then
**mean-pool the per-channel (T_tok, D) tensors**. Alternatives are possible:

  * concat along D: increases D by C-fold, requires retraining head per C
  * attention pool: add a learned aggregation layer (more parameters)

Mean-pool is simple, parameter-free, and works for any C at inference time.

Target format
-------------
y per series: (N=10, 2+k) float32, all-zero rows = empty / no-event slots.
  col 0      : start  (normalised to [0,1] over T=512)
  col 1      : length (normalised to [0,1] over T=512)
  col 2..2+k : binary multi-class one-hot columns (sum >= 1 for real events)

Loss
----
For each of the 10 query slots:
  * has_event mask = (y_cls.sum(-1) > 0)  shape (B, N)
  * position loss  : smooth_l1 on (start, length), applied only where has_event
  * class loss     : BCEWithLogitsLoss on all k columns, applied to all slots
    (empty slots drive logits toward 0, which is the no-event baseline)

Combined: loss = pos_loss + lambda_cls * cls_loss  (lambda_cls=1.0 by default)
"""

import numpy as np
import torch
import torch.nn as nn

from .base import BaseTSFMAdapter


# ---------------------------------------------------------------------------
# EventHead
# ---------------------------------------------------------------------------

class EventHead(nn.Module):
    """Transformer-decoder that turns Chronos embeddings into span predictions.

    Parameters
    ----------
    d_model : int
        Embedding dimension coming out of the Chronos encoder.
    n_classes : int
        Number of binary event-class columns k.
    num_queries : int
        Fixed number of event slots N (default 10).
    num_decoder_layers : int
        Depth of the Transformer decoder (default 2).
    nhead : int
        Number of attention heads (default 8 when d_model >= 512).
    dim_feedforward : int
        FFN inner dimension (default 4 * d_model).
    dropout : float
        Dropout rate in the decoder (default 0.1).
    """

    def __init__(
        self,
        d_model: int,
        n_classes: int,
        num_queries: int = 10,
        num_decoder_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.num_queries = num_queries

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        # Ensure nhead divides d_model cleanly
        while d_model % nhead != 0 and nhead > 1:
            nhead //= 2

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        # N learnable query embeddings — one per event slot
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Output heads
        self.pos_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.query_embed.weight, std=0.02)

    def forward(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        memory : (B, T_tok, D) — Chronos encoder embeddings (mean-pooled over C)

        Returns
        -------
        pos_logits : (B, N, 2)  — raw (pre-sigmoid) span predictions
        cls_logits : (B, N, k)  — raw (pre-sigmoid) class logits
        """
        B = memory.size(0)
        # Expand queries across batch
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)

        decoded = self.decoder(tgt=queries, memory=memory)  # (B, N, D)

        pos_logits = self.pos_head(decoded)   # (B, N, 2)
        cls_logits = self.cls_head(decoded)   # (B, N, k)
        return pos_logits, cls_logits

    def compute_loss(
        self,
        pos_logits: torch.Tensor,
        cls_logits: torch.Tensor,
        y: torch.Tensor,
        lambda_cls: float = 1.0,
    ) -> torch.Tensor:
        """Compute combined position + classification loss.

        Parameters
        ----------
        pos_logits : (B, N, 2) — raw span predictions (sigmoid applied inside)
        cls_logits : (B, N, k) — raw class logits
        y          : (B, N, 2+k) — ground-truth targets, float32
        lambda_cls : float — weight for the class loss term

        Returns
        -------
        scalar loss tensor
        """
        y_pos = y[..., :2]         # (B, N, 2)  start, length
        y_cls = y[..., 2:]         # (B, N, k)  binary class targets

        # Mask: only penalise position loss on slots that have a real event
        has_event = (y_cls.sum(dim=-1) > 0).float()  # (B, N)

        pos_pred = torch.sigmoid(pos_logits)          # (B, N, 2) in [0,1]
        pos_loss_per = nn.functional.smooth_l1_loss(
            pos_pred, y_pos, reduction="none"
        ).mean(dim=-1)                                # (B, N)
        pos_loss = (pos_loss_per * has_event).sum() / (has_event.sum() + 1e-6)

        cls_loss = nn.functional.binary_cross_entropy_with_logits(
            cls_logits, y_cls, reduction="mean"
        )

        return pos_loss + lambda_cls * cls_loss


# ---------------------------------------------------------------------------
# ChronosEventAdapter
# ---------------------------------------------------------------------------

class ChronosEventAdapter(BaseTSFMAdapter):
    """Fitted adapter: frozen Chronos encoder + trained EventHead.

    Parameters
    ----------
    pipeline : ChronosPipeline
        A loaded (and frozen) Chronos pipeline instance.
    head : EventHead
        A trained EventHead instance on the target device.
    device : str
        torch device string, e.g. "cuda" or "cpu".
    n_classes : int
        Number of binary event-class columns k.
    T : int
        Input series length (used only for documentation; not enforced here).
    """

    def __init__(self, pipeline, head: EventHead, device: str,
                 n_classes: int, T: int = 512):
        self.pipeline = pipeline
        self.head = head
        self.device = device
        self.n_classes = n_classes
        self.T = T

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_series(self, x: np.ndarray) -> torch.Tensor:
        """Embed a single multivariate series via Chronos, mean-pool over C.

        Parameters
        ----------
        x : (T, C) float array

        Returns
        -------
        Tensor of shape (T_tok, D) on CPU, float32
        """
        import torch as _torch
        C = x.shape[1]
        channel_embs = []
        for c in range(C):
            ctx = _torch.tensor(x[:, c], dtype=_torch.float32)
            # pipeline.embed returns (1, T_tok, D), tokenizer_state
            emb, _ = self.pipeline.embed(ctx.unsqueeze(0))  # (1, T_tok, D)
            channel_embs.append(emb.squeeze(0).float().cpu())  # (T_tok, D)
        # Stack and mean-pool over channels
        stacked = _torch.stack(channel_embs, dim=0)   # (C, T_tok, D)
        return stacked.mean(dim=0)                     # (T_tok, D)

    # ------------------------------------------------------------------
    # BaseTSFMAdapter interface
    # ------------------------------------------------------------------

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run inference on a single multivariate series.

        Parameters
        ----------
        x : (T, C) float array

        Returns
        -------
        spans : (N=10, 2+k) float32 array
            Columns 0-1 : start and length in [0,1]
            Columns 2.. : binary class probabilities in [0,1]
        """
        self.head.eval()
        memory = self._embed_series(x)                    # (T_tok, D) cpu
        memory = memory.unsqueeze(0).to(self.device)      # (1, T_tok, D)

        with torch.no_grad():
            pos_logits, cls_logits = self.head(memory)    # (1,N,2), (1,N,k)

        pos = torch.sigmoid(pos_logits[0]).cpu().numpy()  # (N, 2)
        cls = torch.sigmoid(cls_logits[0]).cpu().numpy()  # (N, k)
        return np.concatenate([pos, cls], axis=-1).astype(np.float32)  # (N, 2+k)
