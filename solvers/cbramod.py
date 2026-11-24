"""CBraMod, an EEG foundation model, as a frozen encoder for classification.

CBraMod (Wang et al., ICLR 2025) is pretrained on TUEG with 1 s patches at
200 Hz, and its positional encoding takes any montage. Windows are resampled
to 200 Hz and scaled to units of 100 µV, as in its pretraining. The embedding
of a window is the mean of its tokens over time, one per channel, and a
linear probe classifies it.

References:
    https://arxiv.org/abs/2412.07236
    https://braindecode.org/stable/generated/braindecode.models.CBraMod.html
"""

import numpy as np
import torch
from braindecode.models import CBraMod

from benchmark_utils.base_solver import BaseTSFMSolver
from benchmark_utils.eeg import resample

CBRAMOD_SFREQ = 200


class Solver(BaseTSFMSolver):
    """CBraMod frozen encoder with a linear probe."""

    name = "CBraMod"

    requirements = ["pip::braindecode[hub]"]

    parameters = {
        "checkpoint": ["braindecode/cbramod-pretrained"],
        "batch_size": [64],
    }

    @property
    def supported_tasks(self):
        return {"classification"}

    def skip(self):
        reason = super().skip()
        modality = getattr(self.dataset_, "meta", {}).get("modality")
        if reason is None and modality != "eeg":
            reason = f"{self.name} needs EEG windows in µV"
        return reason

    @property
    def model_id(self):
        return self.checkpoint

    def load_model(self, device, dtype):
        model = CBraMod.from_pretrained(self.checkpoint, return_encoder_output=True)
        return model.to(device).eval()

    def embed_batch(self, inputs):
        results = []
        for i in range(0, len(inputs), self.batch_size):
            x = torch.stack(inputs[i : i + self.batch_size]).float().numpy()
            x = resample(x.transpose(0, 2, 1), self.meta["sfreq"], CBRAMOD_SFREQ)
            x = torch.from_numpy(np.ascontiguousarray(x) / 100).float().to(self.device)
            with torch.no_grad():
                tokens = self.model(x)  # (B, C, n_patches, D)
            emb = tokens.mean(dim=2).flatten(1).cpu().numpy().astype(np.float32)
            results.extend(torch.from_numpy(row) for row in emb)
        return results
