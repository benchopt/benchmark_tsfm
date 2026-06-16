"""Mantis solver for time series classification on UCR datasets.

Additionally provides:
  - ``embed_batch``: embeddings via ``MantisTrainer.transform`` for
    classification via the default linear-probe adaptation strategy.

References:
    https://huggingface.co/paris-noah/Mantis-8M
    https://github.com/vfeofanov/mantis
"""

import numpy as np
import torch
from mantis.architecture import MantisV1, MantisV2
from mantis.trainer import MantisTrainer

from benchmark_utils.base_solver import BaseTSFMSolver


class Solver(BaseTSFMSolver):
    """Mantis time series classification solver."""

    name = "Mantis"

    requirements = [
        "pip::mantis-tsfm>=1.0.0",
    ]

    parameters = {
        "checkpoint": ["paris-noah/Mantis-8M"],
        "batch_size": [32],
        "interpolate_to": [512],
        "classifier": ["random_forest"],
        "penalty": ["l2"],
        "C": [1.0],
        "alpha": [1.0],
        "n_estimators": [100],
    }

    @property
    def supported_tasks(self):
        return {"classification"}

    @property
    def model_id(self):
        return self.checkpoint

    def load_model(self, device, dtype):
        MantisBackbone = MantisV2 if "MantisV2" in self.checkpoint else MantisV1
        network = MantisBackbone(device=device)
        network = network.from_pretrained(self.checkpoint)
        return MantisTrainer(device=device, network=network)

    def _prepare_inputs(self, X_batch):
        """Interpolate to ``interpolate_to`` and transpose to (N, C, T)."""
        X_in = X_batch.transpose(0, 2, 1)       # (N, T, C) → (N, C, T)
        target_len = int(self.interpolate_to)
        if X_in.shape[-1] != target_len:
            tensor = torch.tensor(X_in, dtype=torch.float32)
            tensor = torch.nn.functional.interpolate(
                tensor, size=target_len, mode="linear", align_corners=False
            )
            X_in = tensor.numpy()
        if X_in.shape[-1] % 32 != 0:
            raise ValueError(
                "Sequence length must be divisible by 32 for Mantis, "
                f"got {X_in.shape[-1]}"
            )
        return X_in

    def embed_batch(self, inputs):
        # self.model is a MantisTrainer
        results = []
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i : i + self.batch_size]
            X_batch = np.stack(
                [inp.float().cpu().numpy() for inp in batch]
            )                                    # (B, T, C)
            X_prepared = self._prepare_inputs(X_batch)   # (B, C, T_interp)
            with torch.no_grad():
                emb = np.asarray(
                    self.model.transform(X_prepared), dtype=np.float32
                )                                # (B, D)
            for row in emb:
                results.append(torch.from_numpy(row))
        return results
