"""REVE, an EEG foundation model, as a frozen encoder for classification.

REVE embeds each patch of each channel with the 3D position of its electrode,
so it takes any montage. It needs the electrode names, which the EEG datasets
give as ``ch_names``, and data sampled at 200 Hz, like its pretraining data.
The embedding of a window is the mean of its tokens over time, one per
channel, and a linear probe classifies it.

References:
    https://huggingface.co/brain-bzh/reve-base
    https://github.com/braindecode/OpenEEGBench
"""

import numpy as np
import torch
from braindecode.models import REVE

from benchmark_utils.base_solver import BaseTSFMSolver

REVE_SFREQ = 200.0


class Solver(BaseTSFMSolver):
    """REVE frozen encoder with a linear probe."""

    name = "REVE"

    requirements = ["pip::braindecode[hub]"]

    parameters = {
        "checkpoint": ["brain-bzh/reve-base"],
        "batch_size": [64],
    }

    @property
    def supported_tasks(self):
        return {"classification"}

    def skip(self):
        reason = super().skip()
        modality = getattr(self.dataset_, "meta", {}).get("modality")
        if reason is None and modality != "eeg":
            reason = f"{self.name} needs EEG electrode names"
        return reason

    @property
    def model_id(self):
        return (self.checkpoint, tuple(self.meta["ch_names"]), len(self.X_train[0]))

    def load_model(self, device, dtype):
        if self.meta["sfreq"] != REVE_SFREQ:
            raise ValueError(f"REVE expects data at 200 Hz, got {self.meta['sfreq']}")
        model = REVE.from_pretrained(
            self.checkpoint,
            n_outputs=self.meta["n_classes"],
            n_chans=len(self.meta["ch_names"]),
            n_times=len(self.X_train[0]),
            sfreq=REVE_SFREQ,
            chs_info=[{"ch_name": name} for name in self.meta["ch_names"]],
        )
        return model.to(device).eval()

    def embed_batch(self, inputs):
        results = []
        for i in range(0, len(inputs), self.batch_size):
            x = torch.stack(inputs[i : i + self.batch_size]).float()
            x = x.transpose(1, 2).to(self.device)  # (B, T, C) -> (B, C, T)
            # Per-window z-score clipped at 15 sigma, as in REVE's pretraining.
            mean = x.mean(dim=(1, 2), keepdim=True)
            std = x.std(dim=(1, 2), keepdim=True).clamp_min(1e-10)
            x = ((x - mean) / std).clamp(-15, 15)
            with torch.no_grad():
                tokens = self.model(x, return_features=True)["features"]
            # (B, C, n_patches, D) -> (B, C * D)
            emb = tokens.mean(dim=2).flatten(1).cpu().numpy().astype(np.float32)
            results.extend(torch.from_numpy(row) for row in emb)
        return results
