"""Toto-2.0 solver for the TSFM benchmark (local inference).

Directly supports:
  - forecasting : zero-shot via ``Toto2Model.forecast``

Additionally provides:
  - ``embed_batch``: pooled transformer patch embeddings for classification
    and anomaly detection via the default adaptation strategies.

References:
    https://github.com/datadog/toto
"""

import numpy as np
import torch
from toto2 import Toto2Model

from benchmark_utils.adapters import POOLERS
from benchmark_utils.base_solver import BaseTSFMSolver


class Solver(BaseTSFMSolver):
    """Datadog Toto-2.0 zero-shot solver."""

    name = "Toto-2.0"

    requirements = [
        "pip::torch>=2.5",
        "pip::toto-2 @ git+https://github.com/DataDog/toto.git#subdirectory=toto2",
    ]

    sampling_strategy = "run_once"

    parameters = {
        "checkpoint": ["Datadog/Toto-2.0-2.5B"],
        "context_length": [512],
        "decode_block_size": [None],
        "patch_size": [32],
        "layer": [None],
        "pooler": ["mean"],
        "classifier": ["log_reg"],
        "penalty": ["l2"],
        "C": [1.0],
        "alpha": [1.0],
        "n_estimators": [100],
    }

    test_config = {
        "checkpoint": "Datadog/Toto-2.0-4m",
    }

    @property
    def supported_tasks(self):
        return {"forecasting", "classification", "anomaly_detection"}

    @property
    def model_id(self):
        return self.checkpoint

    def load_model(self, device, dtype):
        model = Toto2Model.from_pretrained(self.checkpoint)
        return model.to(device).eval()

    def get_quantile_levels(self) -> tuple[float, ...]:
        return tuple(float(q) / 10 for q in range(1, 10))

    def forecast_batch(self, inputs, covariates, prediction_length):
        device = next(self.model.parameters()).device
        results = []
        for inp in inputs:
            x = inp.float().cpu().numpy()                # (T, C)
            if self.context_length is not None:
                x = x[-self.context_length:]

            # Toto expects (B, C, T)
            target_np = x.T[None]                        # (1, C, T)
            finite_mask_np = np.isfinite(target_np)
            target_np = np.nan_to_num(
                target_np, nan=0.0, posinf=0.0, neginf=0.0
            )

            pad_len = (-target_np.shape[-1]) % self.patch_size
            if pad_len:
                target_np = np.pad(
                    target_np, ((0, 0), (0, 0), (pad_len, 0)), constant_values=0.0
                )
                finite_mask_np = np.pad(
                    finite_mask_np,
                    ((0, 0), (0, 0), (pad_len, 0)),
                    constant_values=False,
                )

            target = torch.from_numpy(target_np).to(device)
            target_mask = torch.from_numpy(finite_mask_np).to(device, dtype=torch.bool)
            series_ids = torch.zeros(
                1, target.shape[1], dtype=torch.long, device=device
            )

            with torch.inference_mode():
                quantiles = self.model.forecast(
                    {
                        "target": target,
                        "target_mask": target_mask,
                        "series_ids": series_ids,
                    },
                    horizon=prediction_length,
                    decode_block_size=self.decode_block_size,
                    has_missing_values=not bool(finite_mask_np.all()),
                )
            # quantiles: (Q, B=1, C, H) → (H, C, Q)
            results.append(quantiles[:, 0].float().cpu().permute(2, 1, 0))
        return results

    def embed_batch(self, inputs):
        pooler = POOLERS[self.pooler]()
        device = next(self.model.parameters()).device
        results = []
        for inp in inputs:
            x = inp.float().cpu().numpy()                # (T, C)
            if self.context_length is not None:
                x = x[-self.context_length:]

            # Toto expects (B, C, T)
            batch = x.T[None]                            # (1, C, T)
            mask = np.isfinite(batch)
            batch = np.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0)

            pad_len = (-batch.shape[-1]) % self.patch_size
            if pad_len:
                batch = np.pad(
                    batch, ((0, 0), (0, 0), (pad_len, 0)), constant_values=0.0
                )
                mask = np.pad(
                    mask, ((0, 0), (0, 0), (pad_len, 0)), constant_values=False
                )

            target = torch.from_numpy(batch).to(device)
            target_mask = torch.from_numpy(mask).to(device, dtype=torch.bool)
            cpm_mask = torch.ones_like(target_mask)
            series_ids = torch.zeros(
                1, target.shape[1], dtype=torch.long, device=device
            )

            captured = {}
            if self.layer is None:
                hook_module = self.model.transformer
            else:
                n = len(self.model.transformer.layers)
                hook_module = self.model.transformer.layers[self.layer % n]

            def _hook(_, __, out):
                captured["h"] = out.detach()

            handle = hook_module.register_forward_hook(_hook)
            with torch.inference_mode():
                try:
                    self.model(
                        target=target,
                        target_mask=target_mask,
                        cpm_mask=cpm_mask,
                        series_ids=series_ids,
                    )
                finally:
                    handle.remove()

            # (1, C, T_patch, D) → (1, T_patch, C, D) to match pooler convention
            emb_np = captured["h"].transpose(1, 2).float().cpu().numpy()
            pooled = pooler.pool(emb_np)                 # (1, C, D)
            results.append(
                torch.from_numpy(pooled[0].reshape(-1).astype(np.float32))
            )
        return results
