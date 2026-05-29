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

Loss — DETR-style Hungarian matching
-------------------------------------
Training uses per-sample optimal bipartite assignment (Hungarian algorithm)
between the N predicted slots and the M real ground-truth events (M <= N):

  1. Build cost matrix (N x M) per sample:
       cost_pos[n,m] = L1(sigmoid(pos_logits[n]), gt_pos[m])   (sum over 2 dims)
       cost_cls[n,m] = -sum_k sigmoid(cls_logits[n,k]) * gt_cls[m,k]
       cost[n,m]     = cost_pos[n,m] + lambda_cls * cost_cls[n,m]
  2. Solve: scipy.optimize.linear_sum_assignment(cost) -> (pred_idx, gt_idx)
  3. Position loss: smooth_l1 on matched (pred, gt) span pairs.
  4. Class loss: BCEWithLogitsLoss where
       - matched slots   -> their assigned GT class vector
       - unmatched slots -> all-zero target (no-event)
  5. Average losses over batch.

This gives the model permutation invariance: the N queries can predict events
in any order and the loss always finds the optimal pairing.
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
        """Compute combined position + classification loss with Hungarian matching.

        For each sample in the batch the optimal bipartite assignment between
        the N predicted slots and the M real ground-truth events is found via
        the Hungarian algorithm.  Matched slots are trained on their assigned
        GT event; unmatched slots are trained toward the no-event class target
        (all-zero class vector).

        Parameters
        ----------
        pos_logits : (B, N, 2) — raw span predictions (sigmoid applied inside)
        cls_logits : (B, N, k) — raw class logits
        y          : (B, N, 2+k) — ground-truth targets, float32;
                     all-zero rows are empty / no-event slots
        lambda_cls : float — weight for the class loss term

        Returns
        -------
        scalar loss tensor
        """
        from scipy.optimize import linear_sum_assignment

        B, N, _ = pos_logits.shape
        y_pos = y[..., :2]   # (B, N, 2)
        y_cls = y[..., 2:]   # (B, N, k)

        pos_pred = torch.sigmoid(pos_logits)   # (B, N, 2) in [0,1]
        cls_prob = torch.sigmoid(cls_logits)   # (B, N, k) in [0,1]

        total_pos_loss = pos_logits.new_zeros(())
        total_cls_loss = pos_logits.new_zeros(())

        for b in range(B):
            # Identify real GT events for this sample
            has_event_b = y_cls[b].sum(dim=-1) > 0   # (N,) bool mask over GT slots
            gt_pos = y_pos[b][has_event_b]            # (M, 2)
            gt_cls = y_cls[b][has_event_b]            # (M, k)
            M = gt_pos.shape[0]

            pred_pos_b = pos_pred[b]      # (N, 2)
            cls_logits_b = cls_logits[b]  # (N, k)
            cls_prob_b = cls_prob[b]       # (N, k) — used for cost matrix only

            if M == 0:
                # No GT events: drive all slots to no-event (zero target)
                total_cls_loss = total_cls_loss + nn.functional.binary_cross_entropy_with_logits(
                    cls_logits_b,
                    torch.zeros_like(cls_logits_b),
                    reduction="mean",
                )
                continue

            # --- cost matrix (N x M) ----------------------------------------
            # L1 position cost: sum of |pred - gt| over the 2 span dimensions
            with torch.no_grad():
                cost_pos = torch.cdist(pred_pos_b, gt_pos, p=1)          # (N, M)
                # Class cost: negative dot product of sigmoid probabilities and
                # GT class vectors — lower cost means better class agreement.
                cost_cls = -(cls_prob_b @ gt_cls.T)                       # (N, M)
                cost = cost_pos + lambda_cls * cost_cls                   # (N, M)

            pred_idx, gt_idx = linear_sum_assignment(cost.cpu().numpy())

            # --- position loss on matched pairs --------------------------------
            matched_pred_pos = pred_pos_b[pred_idx]   # (M, 2)
            matched_gt_pos = gt_pos[gt_idx]            # (M, 2)
            pos_loss_b = nn.functional.smooth_l1_loss(
                matched_pred_pos, matched_gt_pos, reduction="mean"
            )
            total_pos_loss = total_pos_loss + pos_loss_b

            # --- class loss: matched → GT class, unmatched → zero target -------
            cls_target = torch.zeros_like(cls_logits_b)    # (N, k) all zeros
            cls_target[pred_idx] = gt_cls[gt_idx].to(cls_target.dtype)
            cls_loss_b = nn.functional.binary_cross_entropy_with_logits(
                cls_logits_b, cls_target, reduction="mean"
            )
            total_cls_loss = total_cls_loss + cls_loss_b

        return (total_pos_loss + lambda_cls * total_cls_loss) / B


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


# ---------------------------------------------------------------------------
# Training helpers (used by solvers/chronos.py)
# ---------------------------------------------------------------------------

def _get_linear_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    """Linear warmup + cosine annealing LR scheduler (epoch-level)."""
    import math

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def precompute_embeddings(pipeline, X_train, batch_size=128):
    """Embed every training series via a frozen Chronos pipeline.

    Each series is embedded channel-by-channel and the per-channel
    (T_tok, D) tensors are mean-pooled to produce a single (T_tok, D)
    representation. Results are cached on CPU.

    Parameters
    ----------
    pipeline : ChronosPipeline or Chronos2Pipeline
        A loaded (and frozen) Chronos pipeline exposing ``.embed()``.
    X_train : List[np.ndarray (T, C)]
        Training time series.
    batch_size : int
        Number of series to embed in a single ``pipeline.embed()`` call.
        Each channel is treated as a separate univariate series, so the
        actual call size is ``batch_size * C``.

    Returns
    -------
    List[torch.Tensor]  — one (T_tok, D) float32 CPU tensor per series.
    """
    Z_train = []
    n = len(X_train)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = X_train[start:start + batch_size]
            # Flatten all channels across the batch into one list of 1D tensors.
            # Order: x0c0, x0c1, ..., x1c0, x1c1, ...
            contexts = []
            channel_counts = []
            for x in batch:
                x = np.asarray(x, dtype=np.float32)
                C = x.shape[1]
                channel_counts.append(C)
                for c in range(C):
                    contexts.append(torch.tensor(x[:, c], dtype=torch.float32))

            embs, _ = pipeline.embed(contexts)  # (sum(C_i), T_tok, D)
            embs = embs.float().cpu()

            idx = 0
            for C in channel_counts:
                channel_embs = embs[idx:idx + C]        # (C, T_tok, D)
                Z_train.append(channel_embs.mean(dim=0)) # (T_tok, D)
                idx += C

            print(f"  Embedded {min(start + batch_size, n)}/{n} series", end="\r")

    print()
    return Z_train


def _pad_or_truncate_labels(y: np.ndarray, num_queries: int) -> np.ndarray:
    """Pad or truncate a label array to exactly ``num_queries`` rows.

    Parameters
    ----------
    y : (N, 2+k) float array
        Raw event labels for one series.  ``N`` may be smaller or larger
        than ``num_queries``.
    num_queries : int
        Target number of event slots.

    Returns
    -------
    (num_queries, 2+k) float32 array
        Rows beyond the original ``N`` are filled with zeros (no-event).
        Rows beyond ``num_queries`` in the original are discarded.
    """
    y = np.asarray(y, dtype=np.float32)
    N, width = y.shape
    if N == num_queries:
        return y
    if N > num_queries:
        return y[:num_queries]
    # N < num_queries — pad with zero rows
    pad = np.zeros((num_queries - N, width), dtype=np.float32)
    return np.concatenate([y, pad], axis=0)


def fit_event_head(
    Z_train,
    y_train,
    n_classes,
    d_model,
    device,
    batch_size=32,
    num_epochs=100,
    lr=3e-4,
    weight_decay=1e-4,
    warmup_epochs=5,
    num_dec_layers=2,
    lambda_cls=1.0,
    num_queries=10,
):
    """Train an EventHead on pre-computed Chronos embeddings.

    Parameters
    ----------
    Z_train : List[torch.Tensor (T_tok, D)]
        Pre-computed encoder embeddings (CPU), one per training series.
    y_train : List[np.ndarray (N_i, 2+k)]
        Event targets, one per training series.  ``N_i`` may differ across
        series and need not equal ``num_queries``; labels are automatically
        padded (with zeros) or truncated to ``num_queries`` rows.
    n_classes : int
        Number of binary class columns k.
    d_model : int
        Encoder hidden dimension D.
    device : str
        Torch device, e.g. ``"cuda"`` or ``"cpu"``.
    batch_size, num_epochs, lr, weight_decay : training hyperparameters.
    warmup_epochs : int
        Linear warmup duration; cosine decay thereafter.
    num_dec_layers : int
        Transformer decoder depth.
    lambda_cls : float
        Weight of the classification loss relative to the position loss.
    num_queries : int
        Number of event slots (decoder queries); default 10.

    Returns
    -------
    EventHead  — trained, in eval mode, on ``device``.
    """
    # Normalise all labels to (num_queries, 2+k) once, before the training loop
    y_train = [_pad_or_truncate_labels(y, num_queries) for y in y_train]

    head = EventHead(
        d_model=d_model,
        n_classes=n_classes,
        num_queries=num_queries,
        num_decoder_layers=num_dec_layers,
        nhead=8,
    ).to(device)
    head.train()

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = _get_linear_cosine_scheduler(optimizer, warmup_epochs, num_epochs)

    N_train = len(Z_train)
    use_amp = "cuda" in device
    scaler = torch.amp.GradScaler(device, enabled=use_amp)
    num_batches_per_epoch = max(1, int(np.ceil(N_train / batch_size)))

    for epoch in range(num_epochs):
        indices = np.random.permutation(N_train)
        epoch_loss = 0.0
        num_batches = 0

        for batch_start in range(0, N_train, batch_size):
            batch_idx = indices[batch_start: batch_start + batch_size]

            embs = [Z_train[i] for i in batch_idx]
            max_ttok = max(e.shape[0] for e in embs)
            D = embs[0].shape[1]
            B = len(embs)

            memory = torch.zeros(B, max_ttok, D, dtype=torch.float32)
            for bi, e in enumerate(embs):
                memory[bi, : e.shape[0]] = e
            memory = memory.to(device)

            y_batch = torch.tensor(
                np.stack([y_train[i] for i in batch_idx]),
                dtype=torch.float32,
                device=device,
            )  # (B, N, 2+k)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pos_logits, cls_logits = head(memory)
                loss = head.compute_loss(
                    pos_logits, cls_logits, y_batch, lambda_cls=lambda_cls
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1

            print(
                f"  Epoch {epoch + 1:3d}/{num_epochs} | "
                f"step {num_batches:4d}/{num_batches_per_epoch} | "
                f"loss={loss.item():.4f}",
                end="\r",
            )

        scheduler.step()

        avg = epoch_loss / num_batches
        lr_now = scheduler.get_last_lr()[0]
        print(
            f"  Epoch {epoch + 1:3d}/{num_epochs} | "
            f"loss={avg:.4f} | lr={lr_now:.2e}"
            + " " * 20  # clear leftover step line
        )

    head.eval()
    return head
