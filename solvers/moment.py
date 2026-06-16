"""Moment solver for the TSFM benchmark.

Directly supports:
  - forecasting : zero-shot point forecast via ``MOMENTPipeline.forecast``

Additionally provides:
  - ``embed_batch``: pooled patch embeddings for classification via the
    default adaptation strategies.

Model loading is done in ``set_objective`` (untimed).

References:
    https://huggingface.co/AutonLab/MOMENT-1-large
"""

import numpy as np
import torch
from momentfm import MOMENTPipeline

from benchmark_utils.adapters import POOLERS
from benchmark_utils.base_solver import BaseTSFMSolver


class Solver(BaseTSFMSolver):
    """Moment foundation model solver.

    Supports forecasting (zero-shot) and classification (linear probe).
    """

    name = "Moment"

    requirements = [
        "pip::momentfm @ git+https://github.com/moment-timeseries-foundation-model/moment.git",
    ]

    sampling_strategy = "run_once"

    parameters = {
        "checkpoint": ["AutonLab/MOMENT-1-large"],
        "pooler": ["mean"],
        "classifier": ["log_reg"],
        "penalty": ["l2"],
        "C": [1.0],
        "alpha": [1.0],
        "n_estimators": [100],
    }

    test_config = {
        "checkpoint": "AutonLab/MOMENT-1-small",
    }

    @property
    def supported_tasks(self):
        return {"forecasting", "classification"}

    @property
    def model_id(self):
        return self.checkpoint

    def load_model(self, device, dtype):
        pipeline = MOMENTPipeline.from_pretrained(
            self.checkpoint,
            torch_dtype=torch.float32,
        )
        return pipeline.to(device)

    @property
    def quantile_levels(self):
        return (0.5,)  # Moment outputs point forecasts only

    def forecast_batch(self, inputs, covariates, prediction_length):
        device = next(self.model.parameters()).device
        results = []
        for inp in inputs:
            # inp: (T, C) — Moment expects (1, C, T)
            x_enc = inp.float().T.unsqueeze(0).to(device)   # (1, C, T)
            input_mask = torch.ones(
                1, inp.shape[0], dtype=torch.float32, device=device
            )

            with torch.no_grad():
                outputs = self.model.forecast(
                    x_enc=x_enc,
                    input_mask=input_mask,
                    prediction_length=prediction_length,
                )

            forecast = outputs.forecast if hasattr(outputs, "forecast") else outputs
            if isinstance(forecast, tuple):
                forecast = forecast[0]

            arr = forecast.squeeze(0).float().cpu()          # (C, H) or (H,)
            if arr.ndim == 1:
                arr = arr.unsqueeze(-1)                      # (H, 1)
            elif arr.shape[0] != prediction_length:
                arr = arr.T                                  # (H, C)
            arr = arr[:prediction_length]

            results.append(arr.unsqueeze(-1))                # (H, C, 1)
        return results

    def embed_batch(self, inputs):
        pooler = POOLERS[self.pooler]()
        device = next(self.model.parameters()).device
        results = []
        for inp in inputs:
            # inp: (T, C) — Moment expects (1, C, T)
            x_enc = inp.float().T.unsqueeze(0).to(device)   # (1, C, T)

            with torch.no_grad():
                outputs = self.model.embed(x_enc=x_enc, reduction="none")
                emb = outputs.embeddings                     # (1, C, n_patches, D)

            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()

            # (1, C, n_patches, D) → (1, n_patches, C, D) to match pooler convention
            emb_4d = emb.transpose(0, 2, 1, 3)
            pooled = pooler.pool(emb_4d)                     # (1, C, D)
            results.append(
                torch.from_numpy(pooled[0].reshape(-1).astype(np.float32))
            )
        return results
