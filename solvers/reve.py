"""REVE solver — frozen foundation backbone + linear probe.

REVE expects EEG sampled at 200 Hz with monopolar 10-20 channel
names. Inputs at other rates are resampled via polyphase filtering;
datasets with bipolar or prefixed channel names are skipped (the
position bank cannot resolve them to single 3D coordinates).

References:
    https://huggingface.co/brain-bzh/reve-large
    https://huggingface.co/brain-bzh/reve-positions
"""

from math import gcd

import numpy as np
import torch
from benchopt import BaseSolver
from scipy.signal import resample_poly

from benchmark_utils.adapters.linear_probe import LinearProbeAdapter

SUPPORTED_TASKS = {"classification"}
REVE_SFREQ = 200


class Solver(BaseSolver):
    """REVE foundation model + linear probe."""

    name = "REVE"

    requirements = [
        "pip::braindecode",
        "pip::transformers",
        "pip::torch",
        "pip::scipy",
    ]

    parameters = {
        "checkpoint": ["brain-bzh/reve-large"],
        "pos_bank_checkpoint": ["brain-bzh/reve-positions"],
        "batch_size": [16],
        "n_estimators": [100],
        "max_iter": [1000],
        "classifier": ["logistic_regression"],
    }

    def skip(self, task, **kwargs):
        if task not in SUPPORTED_TASKS:
            return True, f"REVE solver does not support task={task!r}"

        # REVE was trained on monopolar 10-20 montages — bipolar
        # derivations ("EEG Fpz-Cz") have no single 3D position.
        ch_names = kwargs.get("ch_names")
        if ch_names is None:
            return True, "REVE requires `ch_names` in dataset meta."
        bad = [
            n for n in ch_names
            if "-" in n or any(
                n.startswith(p) for p in ("EEG ", "EOG ", "EMG ")
            )
        ]
        if bad:
            return True, (
                f"REVE expects monopolar 10-20 names; got {bad[0]!r}."
            )

        # Gated repo: skip cleanly when access/auth is missing.
        try:
            from huggingface_hub import HfApi
            HfApi().model_info(self.checkpoint)
        except Exception as e:
            return True, (
                f"REVE checkpoint '{self.checkpoint}' gated or unreachable. "
                f"Request access and run `huggingface-cli login`. "
                f"({type(e).__name__})"
            )
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

        freq = meta.get("freq") or REVE_SFREQ
        self._src_sfreq = float(freq)
        self._ch_names = meta["ch_names"]

        should_reload = (
            not hasattr(self, "_network")
            or getattr(self, "_loaded_checkpoint", None) != self.checkpoint
        )
        if should_reload:
            try:
                from transformers import AutoModel

                # trust_remote_code: REVE ships custom modeling code on
                # the Hub. Mandatory to avoid an interactive prompt that
                # would break batch/CI runs.
                network = AutoModel.from_pretrained(
                    self.checkpoint, trust_remote_code=True
                )
                pos_bank = AutoModel.from_pretrained(
                    self.pos_bank_checkpoint, trust_remote_code=True
                )

                self._network = network.to(device).eval()
                self._pos_bank = pos_bank.to(device).eval()
                self._loaded_checkpoint = self.checkpoint
                print(
                    f"✓ REVE checkpoint loaded: {self.checkpoint} "
                    f"on device: {device}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load REVE checkpoint '{self.checkpoint}': {e}"
                )

        # Positions (C, 3) recomputed per dataset since ch_names varies.
        with torch.no_grad():
            self._positions = self._pos_bank(self._ch_names)

        self._device = device

    def run(self, _):
        self._adapter = LinearProbeAdapter(
            encoder=self,
            task=self.task,
            classifier=self.classifier,
            max_iter=self.max_iter,
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
            pos = self._positions.unsqueeze(0).expand(x_t.size(0), -1, -1)

            with torch.no_grad():
                emb = self._network(x_t, pos)

            if emb.ndim > 2:
                emb = emb.flatten(start_dim=1)

            all_embeddings.append(emb.cpu().numpy())

        return np.vstack(all_embeddings)

    def _prepare_inputs(self, X_batch):
        """Reshape (N, T, C) → (N, C, T) and resample to 200 Hz.

        Polyphase (anti-aliased) — REVE is frequency-aware so naive
        ``F.interpolate`` would corrupt the bands it was trained on.
        """
        X_in = X_batch.transpose(0, 2, 1)

        if int(self._src_sfreq) != REVE_SFREQ:
            src = int(self._src_sfreq)
            g = gcd(src, REVE_SFREQ)
            X_in = resample_poly(
                X_in, up=REVE_SFREQ // g, down=src // g, axis=-1
            )

        return np.ascontiguousarray(X_in, dtype=np.float32)

    def get_result(self):
        return {"model": self._adapter}
