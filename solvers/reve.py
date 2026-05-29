"""REVE solver for EEG time series classification.

Uses the REVE EEG foundation model (HuggingFace
``brain-bzh/reve-*``) to extract embeddings, then trains a Random
Forest classifier on top — no backbone fine-tuning, the foundation
model is kept frozen.

REVE expects EEG sampled at **200 Hz**; inputs sampled at any other
rate are resampled with a polyphase (anti-aliased) filter before the
forward pass.

References:
    https://huggingface.co/brain-bzh/reve-large
    https://huggingface.co/brain-bzh/reve-positions
"""

import numpy as np
import torch
from benchopt import BaseSolver
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer

SUPPORTED_TASKS = {"classification"}
REVE_SFREQ = 200  # REVE is trained at 200 Hz


class Solver(BaseSolver):
    """REVE foundation model + Random Forest classifier."""

    name = "REVE-RandomForest"

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
    }

    def skip(self, task, **kwargs):
        if task not in SUPPORTED_TASKS:
            return True, f"REVE solver does not support task={task!r}"

        # REVE expects monopolar 10-20 channel names ("Fpz", "C3", …).
        # Datasets that ship bipolar derivations ("EEG Fpz-Cz") or
        # prefixed names cannot be aligned with REVE's position bank —
        # skip them rather than feed bogus electrode positions.
        ch_names = kwargs.get("ch_names")
        if ch_names is None:
            return True, (
                "REVE requires `ch_names` in dataset meta to build "
                "channel positions; dataset does not provide it."
            )
        bad = [
            n for n in ch_names
            if "-" in n or any(
                n.startswith(prefix) for prefix in ("EEG ", "EOG ", "EMG ")
            )
        ]
        if bad:
            return True, (
                f"REVE expects monopolar 10-20 channel names; dataset "
                f"provides bipolar/prefixed names (e.g. {bad[0]!r}). "
                f"Skipping to avoid silently producing biased embeddings."
            )

        # REVE checkpoints are gated on HuggingFace Hub. Skip cleanly
        # (rather than crashing mid-benchmark) when the user has no
        # access — either because they haven't requested it or because
        # their machine isn't authenticated (`huggingface-cli login`).
        try:
            from huggingface_hub import HfApi
            HfApi().model_info(self.checkpoint)
        except Exception as e:
            return True, (
                f"REVE checkpoint '{self.checkpoint}' is gated or "
                f"unreachable. Request access at "
                f"https://huggingface.co/{self.checkpoint} and run "
                f"`huggingface-cli login`. ({type(e).__name__}: {e})"
            )
        return False, None

    def set_objective(self, task, X_train, y_train, **meta):
        """Prepare the solver for a given dataset configuration.

        The foundation model and position bank are loaded here (not in
        ``run``) so the download/load time is excluded from timing.
        """
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

        # Source sampling rate and electrode names must come from the
        # dataset's meta — REVE cannot infer them.
        self._src_sfreq = float(meta.get("freq", REVE_SFREQ))
        self._ch_names = meta.get("ch_names", None)
        if self._ch_names is None:
            raise ValueError(
                "REVE requires `ch_names` in dataset meta to build "
                "channel positions."
            )

        should_reload = (
            not hasattr(self, "_network")
            or getattr(self, "_loaded_checkpoint", None) != self.checkpoint
        )
        if should_reload:
            try:
                from transformers import AutoModel

                # REVE ships custom modeling code on the Hub
                # (`modeling_reve.py`), so `trust_remote_code=True` is
                # mandatory to avoid an interactive prompt that would
                # break batch / CI runs of the benchmark.
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

        # Pre-compute channel positions (C, 3) once per dataset.
        with torch.no_grad():
            self._positions = self._pos_bank(self._ch_names)

        self._device = device

        self.model = make_pipeline(
            FunctionTransformer(self._extract_embeddings),
            RandomForestClassifier(
                n_estimators=self.n_estimators,
                n_jobs=-1,
                random_state=42,
            ),
        )

    def run(self, _):
        """Fit the Random Forest on REVE embeddings."""
        self.model.fit(self.X_train, self.y_train)

    def _extract_embeddings(self, X):
        """Forward batches through the frozen REVE backbone.

        Returns
        -------
        np.ndarray of shape (N, embedding_dim)
        """
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
            # Broadcast positions (C, 3) → (B, C, 3) for this batch.
            pos = self._positions.unsqueeze(0).expand(x_t.size(0), -1, -1)

            # The HF custom `Reve` class loaded via AutoModel is tagged
            # "Feature Extraction" on the Hub: its forward already
            # returns embeddings directly, no flag needed.
            with torch.no_grad():
                emb = self._network(x_t, pos)
            # If REVE returns a sequence/spatial map (n_chans × n_patches
            # × D), flatten the trailing axes into one vector per sample.
            if emb.ndim > 2:
                emb = emb.flatten(start_dim=1)

            all_embeddings.append(emb.cpu().numpy())

        return np.vstack(all_embeddings)

    def _prepare_inputs(self, X_batch):
        """Reshape (N, T, C) → (N, C, T) and resample to 200 Hz.

        REVE is frequency-aware. A naive ``F.interpolate`` would alias when
        downsampling and add spectral artefacts when upsampling, so we
        use a polyphase resampler (``scipy.signal.resample_poly``) that
        applies a anti-aliasing filter.
        """
        X_in = X_batch.transpose(0, 2, 1)  # (N, C, T)

        if int(self._src_sfreq) != REVE_SFREQ:
            from math import gcd
            from scipy.signal import resample_poly

            src = int(self._src_sfreq)
            g = gcd(src, REVE_SFREQ)
            up = REVE_SFREQ // g
            down = src // g
            X_in = resample_poly(X_in, up=up, down=down, axis=-1)

        return np.ascontiguousarray(X_in, dtype=np.float32)

    def get_result(self):
        """Return the fitted pipeline."""
        return {"model": self.model}
