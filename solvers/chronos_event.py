"""Chronos-based event-detection solver for the TSFM benchmark.

Supports:
  - event_detection : frozen Chronos T5 encoder + trainable EventHead

Overview
--------
Event detection is a structured-prediction task. Each input series x (T, C)
is mapped to a fixed-size set of N=10 span predictions:

    output : (N=10, 2+k)
        col 0   : event start,  normalised to [0, 1] over T=512
        col 1   : event length, normalised to [0, 1] over T=512
        col 2.. : k binary class probability columns

y_train / y_test format (padded)
---------------------------------
Each element is a float32 array of shape (N=10, 2+k). Empty/no-event slots
are represented as all-zero rows. Real event slots satisfy
    row[2:].sum() >= 1
The solver assumes this padded format and reads n_classes from meta
(key "n_classes") and T from meta (key "T", default 512).

Meta keys expected from the dataset
-------------------------------------
  n_classes : int   number of binary class columns k
  T         : int   series length (default 512)

These must be returned by the dataset's get_data() via the ``extra`` dict
that gets forwarded to objective.set_data() and then to get_objective().

Architecture (frozen encoder, trained head)
--------------------------------------------
Chronos is purely univariate. To handle C channels we embed each channel
independently via pipeline.embed() and **mean-pool the (T_tok, D) tensors**
across channels. This is parameter-free and works for any C at inference time.
Alternatives (concat, attention aggregation) are noted in event_detection.py.

Because Chronos is frozen, all training embeddings are pre-computed once in
set_objective() (untimed) and cached in CPU RAM. The timed run() block only
trains the small EventHead — very fast on H100.

Hyperparameters (H100 defaults)
---------------------------------
  model_size       : "small"  (D=512, 46 M params)
  model_path       : ""  empty = download from HuggingFace Hub
                         set to a local directory path to load offline
                         e.g. "/path/to/models/chronos_t5_small"
  batch_size       : 32
  num_epochs       : 100
  lr               : 3e-4
  weight_decay     : 1e-4
  warmup_epochs    : 5   (linear warmup, cosine decay thereafter)
  num_queries      : 10  (= N, fixed)
  num_dec_layers   : 2
  lambda_cls       : 1.0 (weight of class loss vs. position loss)

Benchopt timing contract
--------------------------
  set_objective()  — model loading + embedding pre-computation (UNTIMED)
  run()            — EventHead training (TIMED)
  get_result()     — returns {"model": ChronosEventAdapter}
"""

import math

import numpy as np
import torch
from benchopt import BaseSolver

from benchmark_utils.adapters.event_detection import ChronosEventAdapter, EventHead


SUPPORTED_TASKS = {"event_detection"}

# Chronos encoder output dimensions by model size
_CHRONOS_D = {
    "tiny": 64,
    "mini": 128,
    "small": 512,
    "base": 768,
    "large": 1024,
}


def _get_linear_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    """Linear warmup + cosine annealing LR scheduler (epoch-level)."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Solver(BaseSolver):
    """Chronos encoder (frozen) + EventHead event-detection solver.

    See module docstring for full details on the architecture, data format,
    and hyperparameter choices.
    """

    name = "Chronos-EventDetection"

    requirements = [
        "pip::chronos-forecasting>=1.4",
        "pip::torch",
    ]

    sampling_strategy = "run_once"

    parameters = {
        "model_size": ["small"],
        "model_path": [""],           # empty = load from HuggingFace Hub
        "batch_size": [32],
        "num_epochs": [0, 100],
        "lr": [3e-4],
        "weight_decay": [1e-4],
        "warmup_epochs": [5],
        "num_dec_layers": [2],
        "lambda_cls": [1.0],
    }

    def skip(self, task, **kwargs):
        if task not in SUPPORTED_TASKS:
            return True, f"Chronos-EventDetection does not support task={task!r}"
        return False, None

    # ------------------------------------------------------------------
    # set_objective — UNTIMED
    # ------------------------------------------------------------------

    def set_objective(self, X_train, y_train, task, **meta):
        """Load Chronos, freeze it, and pre-compute training embeddings.

        Parameters
        ----------
        X_train   : List[np.ndarray (T, C)]
        y_train   : List[np.ndarray (N=10, 2+k)]  padded event targets
        task      : str  must be "event_detection"
        **meta    : must include "n_classes" (int) and optionally "T" (int)
        """
        import torch as _torch
        from chronos import ChronosPipeline

        self.task = task
        self.X_train = X_train
        self.y_train = y_train
        self.meta = meta

        # --- Infer task dimensions ---
        self.n_classes = int(meta["n_classes"])
        self.T = int(meta.get("T", 512))
        self.k = self.n_classes  # alias

        # --- Device ---
        self.device = "cuda" if _torch.cuda.is_available() else "cpu"

        # --- Resolve model identifier ---
        # model_path="" (default) → load from HuggingFace Hub
        # model_path="/some/local/dir" → load from that local directory
        model_id = (
            self.model_path.strip()
            if self.model_path.strip()
            else f"amazon/chronos-t5-{self.model_size}"
        )

        # --- Load Chronos pipeline (once; cached across dataset configs) ---
        should_reload = (
            not hasattr(self, "_pipeline")
            or not hasattr(self, "_loaded_model_id")
            or self._loaded_model_id != model_id
        )
        if should_reload:
            self._pipeline = ChronosPipeline.from_pretrained(
                model_id,
                device_map="auto",
                dtype=_torch.bfloat16,
            )
            self._loaded_model_id = model_id
            print(f"Loaded Chronos checkpoint: {model_id}")

        # --- Freeze encoder ---
        for param in self._pipeline.model.parameters():
            param.requires_grad = False
        self._pipeline.model.eval()

        # --- Encoder output dimension ---
        self.d_model = _CHRONOS_D.get(self.model_size, 512)

        # --- Pre-compute training embeddings (frozen encoder → one-time) ---
        print(
            f"Pre-computing embeddings for {len(X_train)} training series "
            f"(C={X_train[0].shape[1]}, T={self.T}) ..."
        )
        self._Z_train = []  # List of (T_tok, D) CPU tensors
        with _torch.no_grad():
            for x in X_train:
                emb = self._embed_series(x)   # (T_tok, D) float32 on CPU
                self._Z_train.append(emb)

        print("Embedding pre-computation complete.")

    # ------------------------------------------------------------------
    # run — TIMED
    # ------------------------------------------------------------------

    def run(self, _):
        """Train the EventHead on top of cached Chronos embeddings."""
        import torch as _torch

        head = EventHead(
            d_model=self.d_model,
            n_classes=self.n_classes,
            num_queries=10,
            num_decoder_layers=self.num_dec_layers,
            nhead=8,
        ).to(self.device)
        head.train()

        optimizer = _torch.optim.AdamW(
            head.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = _get_linear_cosine_scheduler(
            optimizer, self.warmup_epochs, self.num_epochs
        )

        N_train = len(self._Z_train)
        use_amp = self.device == "cuda"
        scaler = _torch.amp.GradScaler("cuda", enabled=use_amp)

        for epoch in range(self.num_epochs):
            indices = np.random.permutation(N_train)
            epoch_loss = 0.0
            num_batches = 0

            for batch_start in range(0, N_train, self.batch_size):
                batch_idx = indices[batch_start: batch_start + self.batch_size]

                # --- Collate: pad to same T_tok (all same T → same T_tok) ---
                embs = [self._Z_train[i] for i in batch_idx]
                max_ttok = max(e.shape[0] for e in embs)
                D = embs[0].shape[1]
                B = len(embs)

                memory = _torch.zeros(B, max_ttok, D, dtype=_torch.float32)
                for bi, e in enumerate(embs):
                    memory[bi, : e.shape[0]] = e
                memory = memory.to(self.device)

                # --- Targets ---
                y_batch = _torch.tensor(
                    np.stack([self.y_train[i] for i in batch_idx]),
                    dtype=_torch.float32,
                    device=self.device,
                )  # (B, N, 2+k)

                # --- Forward + loss ---
                optimizer.zero_grad()
                with _torch.amp.autocast("cuda", dtype=_torch.bfloat16,
                                         enabled=use_amp):
                    pos_logits, cls_logits = head(memory)
                    loss = head.compute_loss(
                        pos_logits, cls_logits, y_batch,
                        lambda_cls=self.lambda_cls,
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                _torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                num_batches += 1

            scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg = epoch_loss / max(num_batches, 1)
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch + 1:3d}/{self.num_epochs} | "
                    f"loss={avg:.4f} | lr={lr_now:.2e}"
                )

        head.eval()
        self._adapter = ChronosEventAdapter(
            pipeline=self._pipeline,
            head=head,
            device=self.device,
            n_classes=self.n_classes,
            T=self.T,
        )

    # ------------------------------------------------------------------
    # get_result
    # ------------------------------------------------------------------

    def get_result(self):
        return {"model": self._adapter}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_series(self, x: np.ndarray) -> torch.Tensor:
        """Embed a (T, C) array via frozen Chronos, mean-pool across C.

        Returns
        -------
        Tensor (T_tok, D) float32 on CPU
        """
        C = x.shape[1]
        channel_embs = []
        for c in range(C):
            ctx = torch.tensor(x[:, c], dtype=torch.float32)
            emb, _ = self._pipeline.embed(ctx.unsqueeze(0))  # (1, T_tok, D)
            channel_embs.append(emb.squeeze(0).float().cpu())
        stacked = torch.stack(channel_embs, dim=0)  # (C, T_tok, D)
        return stacked.mean(dim=0)                  # (T_tok, D)
