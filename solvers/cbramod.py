"""CBraMod solver — frozen foundation backbone + linear probe.

CBraMod (Wang et al., ICLR 2025) is pretrained on TUEG with 1-second
patches at 200 Hz. Inputs at any other rate are resampled via a
polyphase (anti-aliased) filter. Channel-agnostic montage thanks to
its Asymmetric Conditional Positional Encoding.

References:
    https://arxiv.org/abs/2412.07236
    https://braindecode.org/stable/generated/braindecode.models.CBraMod.html
"""

from math import gcd

import numpy as np
import torch
from benchopt import BaseSolver
from scipy.signal import resample_poly

from benchmark_utils.adapters.linear_probe import LinearProbeAdapter

SUPPORTED_TASKS = {"classification"}
CBRAMOD_SFREQ = 200
CBRAMOD_PATCH_SIZE = 200  # 1-second patches at 200 Hz; T must be a multiple


class Solver(BaseSolver):
    """CBraMod foundation model + linear probe."""

    name = "CBraMod"

    requirements = [
        "pip::braindecode",
        "pip::torch",
        "pip::scipy",
    ]

    parameters = {
        "checkpoint": ["braindecode/cbramod-pretrained"],
        "batch_size": [16],
        "n_estimators": [100],
        "classifier": ["log_reg"],
    }

    def skip(self, task, **kwargs):
        if task not in SUPPORTED_TASKS:
            return True, f"CBraMod solver does not support task={task!r}"
        return False, None

    def set_objective(self, task, X_train, y_train, **meta):
        self.task = task
        self.X_train = X_train
        self.y_train = y_train
        self.meta = meta

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        freq = meta.get("freq") or CBRAMOD_SFREQ
        self._src_sfreq = float(freq)

        should_reload = (
            not hasattr(self, "_network")
            or getattr(self, "_loaded_checkpoint", None) != self.checkpoint
        )
        if should_reload:
            try:
                from braindecode.models import CBraMod

                # return_encoder_output=True bypasses the randomly-init
                # classification head and exposes encoder features.
                network = CBraMod.from_pretrained(
                    self.checkpoint,
                    return_encoder_output=True,
                )
                self._network = network.to(device).eval()
                self._loaded_checkpoint = self.checkpoint
                print(
                    f"✓ CBraMod checkpoint loaded: {self.checkpoint} "
                    f"on device: {device}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load CBraMod checkpoint "
                    f"'{self.checkpoint}': {e}"
                )

        self._device = device

    def run(self, _):
        self._adapter = LinearProbeAdapter(
            encoder=self,
            task=self.task,
            classifier=self.classifier,
            n_estimators=self.n_estimators,
        )
        self._adapter.fit(self.X_train, self.y_train)

    def encode(self, X):
        """LinearProbeAdapter entry point — returns (N, embed_dim)."""
        return self._extract_embeddings(X)

    def _extract_embeddings(self, X):
        batch_size = self.batch_size
        n_samples = len(X)
        all_embeddings = []

        for batch_idx in range(0, n_samples, batch_size):
            batch_end = min(batch_idx + batch_size, n_samples)
            X_batch = np.asarray(X[batch_idx:batch_end], dtype=np.float32)
            X_batch_processed = self._prepare_inputs(X_batch)

            x_t = torch.tensor(
                X_batch_processed, dtype=torch.float32, device=self._device
            )

            with torch.no_grad():
                emb = self._network(x_t)

            # Encoder returns (B, n_chans, n_patches, D) — flatten to
            # one vector per sample.
            if emb.ndim > 2:
                emb = emb.flatten(start_dim=1)

            all_embeddings.append(emb.cpu().numpy())

        return np.vstack(all_embeddings)

    def _prepare_inputs(self, X_batch):
        """Reshape (N, T, C) → (N, C, T), resample to 200 Hz, truncate.

        Polyphase resampling is anti-aliased so it works for both up-
        and downsampling. CBraMod patches at 200 samples per token, so
        ``T`` must be a multiple of ``CBRAMOD_PATCH_SIZE`` — we truncate
        any trailing partial patch rather than zero-pad (cleaner than
        injecting fake samples).
        """
        X_in = X_batch.transpose(0, 2, 1)

        if int(self._src_sfreq) != CBRAMOD_SFREQ:
            src = int(self._src_sfreq)
            g = gcd(src, CBRAMOD_SFREQ)
            X_in = resample_poly(
                X_in, up=CBRAMOD_SFREQ // g, down=src // g, axis=-1
            )

        T = X_in.shape[-1]
        remainder = T % CBRAMOD_PATCH_SIZE
        if remainder:
            X_in = X_in[..., : T - remainder]

        return np.ascontiguousarray(X_in, dtype=np.float32)

    def get_result(self):
        return {"model": self._adapter}
