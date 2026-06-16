"""Chronos solver for the TSFM benchmark (local inference).

Directly supports:
  - forecasting : zero-shot via ``ChronosPipeline.predict`` (Monte-Carlo sampling)

Additionally provides:
  - ``embed_batch``: pooled encoder embeddings for classification and
    anomaly detection via the default adaptation strategies.

Model loading is done in ``set_objective`` (untimed). Inference batches
every (series, cutoff, channel) tuple into chunked ``ChronosPipeline.predict``
calls — Chronos v1 is univariate so each channel is processed independently.

References
----------
    https://github.com/amazon-science/chronos-forecasting
"""

import numpy as np
import torch
from chronos import ChronosPipeline

from benchmark_utils.adapters import POOLERS
from benchmark_utils.base_solver import BaseTSFMSolver


class Solver(BaseTSFMSolver):
    """Chronos v1 zero-shot solver.

    Parameters
    ----------
    model_size : str
        Chronos model variant: "tiny", "mini", "small", "base", "large".
    pooler : {"mean", "max", "last"}
        Pooling strategy over the time-token axis for embed_batch.
    """

    name = "Chronos"

    requirements = ["pip::chronos-forecasting>=2.2", "pip::torch"]

    # Cap how many univariate histories go through one pipeline.predict call.
    # Chronos v1 left-pads the whole list into a single T5 forward whose
    # attention is O(L^2) in the padded length, so batching everything at once
    # blows up memory on datasets with many long series (e.g. m4_weekly).
    PREDICT_BATCH_SIZE = 16

    parameters = {
        "model_size": ["small"],
        "pooler": ["mean"],
        "classifier": ["log_reg"],
        "penalty": ["l2"],
        "C": [1.0],
        "alpha": [1.0],
        "n_estimators": [100],
    }

    @property
    def supported_tasks(self) -> set:
        return {"forecasting", "classification", "anomaly_detection"}

    @property
    def model_id(self) -> str:
        return f"amazon/chronos-t5-{self.model_size}"

    def load_model(self, device, dtype):
        return ChronosPipeline.from_pretrained(
            self.model_id,
            device_map=device,
            dtype=dtype,
        )

    @property
    def quantile_levels(self) -> tuple[float, ...]:
        return (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    def forecast_batch(self, inputs, covariates, prediction_length):
        # Chronos v1 is univariate — split each (T, C) input into C 1-D tensors.
        univariate: list[torch.Tensor] = []
        layout: list[tuple[int, int]] = []  # (inp_idx, channel_idx)

        for inp_idx, inp in enumerate(inputs):
            x = inp.float().cpu()                    # (T, C)
            if x.ndim == 1:
                x = x.unsqueeze(-1)
            for c in range(x.shape[1]):
                univariate.append(x[:, c])           # (T,)
                layout.append((inp_idx, c))

        # Seed before sampling: Chronos v1 is non-deterministic by default.
        # A fixed seed ensures identical histories produce identical draws —
        # required by the leakage probe which compares two predict() calls.
        torch.manual_seed(0)
        all_samples: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(univariate), self.PREDICT_BATCH_SIZE):
                batch = univariate[start : start + self.PREDICT_BATCH_SIZE]
                all_samples.append(
                    self.model.predict(batch, prediction_length=prediction_length)
                )
        samples = torch.cat(all_samples, dim=0).float().cpu().numpy()
        # samples: (total_univariate, num_samples, H)

        q_levels = self.quantile_levels
        q_arr = np.quantile(samples, q=list(q_levels), axis=1).transpose(1, 0, 2)
        # q_arr: (total_univariate, Q, H)

        # Reassemble per-input (H, C, Q) tensors.
        by_input: dict[int, dict[int, np.ndarray]] = {}
        for k, (inp_idx, c) in enumerate(layout):
            by_input.setdefault(inp_idx, {})[c] = q_arr[k]  # (Q, H)

        results = []
        for inp_idx in range(len(inputs)):
            channels = by_input[inp_idx]
            # q_arr[k] is (Q, H); .T gives (H, Q); stack over C → (H, C, Q)
            result = np.stack([channels[c].T for c in range(len(channels))], axis=1)
            results.append(torch.from_numpy(result))
        return results

    def embed_batch(self, inputs):
        pooler = POOLERS[self.pooler]()
        results = []
        for inp in inputs:
            x = inp.float().cpu().numpy()            # (T, C)
            if x.ndim == 1:
                x = x[:, None]
            # Flatten channels into batch axis (pipeline is univariate)
            flat = torch.from_numpy(x.T.copy())     # (C, T)
            with torch.no_grad():
                emb, _ = self.model.embed(flat)      # (C, T_tok, D)
            emb_np = emb.float().cpu().numpy()
            # (C, T_tok, D) → (1, T_tok, C, D) to match pooler convention
            emb_4d = emb_np.transpose(1, 0, 2)[None]
            pooled = pooler.pool(emb_4d)             # (1, C, D)
            results.append(
                torch.from_numpy(pooled[0].reshape(-1).astype(np.float32))
            )
        return results
