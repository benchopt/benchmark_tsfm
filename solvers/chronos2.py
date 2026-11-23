"""Chronos-2 solver for the TSFM benchmark (local inference).

Directly supports:
  - forecasting : zero-shot via ``Chronos2Pipeline.predict_quantiles``

Additionally provides:
  - ``embed_batch``: pooled encoder embeddings for classification and
    anomaly detection via the default adaptation strategies.

Model loading is done in ``set_objective`` (untimed). Inference batches
every (series, cutoff) pair into a single
``Chronos2Pipeline.predict_quantiles`` call: the pipeline accepts a list
of variable-length tensors and applies left-padding internally, so all
the per-cutoff work happens in one forward pass.

References
----------
    https://github.com/amazon-science/chronos-forecasting
"""

import numpy as np
import torch
from chronos.chronos2 import Chronos2Pipeline
from einops import rearrange

from benchmark_utils.adapters import POOLERS
from benchmark_utils.base_solver import BaseTSFMSolver
from benchmark_utils.forecasting_constants import CANONICAL_QUANTILE_LEVELS


class Solver(BaseTSFMSolver):
    """Chronos-2 zero-shot solver.

    Parameters
    ----------
    model_size : str
        Chronos-2 model variant: "tiny", "mini", "small", "base", "large".
    pooler : {"mean", "max", "last"}
        Pooling strategy over the time-token axis for embed_batch.
    """

    name = "Chronos2"

    requirements = ["pip::chronos-forecasting>=2.2,<3", "pip::einops"]

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
    def supported_tasks(self):
        return {"forecasting", "classification", "anomaly_detection"}

    @property
    def model_id(self):
        if self.model_size == "base":
            # Leaderboard checkpoint; no autogluon/chronos-2-base on HF.
            return "amazon/chronos-2"
        return f"autogluon/chronos-2-{self.model_size}"

    def load_model(self, device, dtype):
        return Chronos2Pipeline.from_pretrained(
            self.model_id,
            device_map=device,
            dtype=dtype,
        )

    @property
    def quantile_levels(self):
        # predict_quantiles slices the native grid, no interpolation.
        return CANONICAL_QUANTILE_LEVELS

    def forecast_batch(self, inputs, covariates, prediction_length):
        # inputs: list of (T, C) tensors; Chronos-2 expects (C, T)
        chrono_inputs = [inp.T for inp in inputs]
        with torch.no_grad():
            output, _ = self.model.predict_quantiles(
                chrono_inputs,
                prediction_length=prediction_length,
                quantile_levels=list(self.quantile_levels),
            )
        return [rearrange(pred, "c h q -> h c q") for pred in output]

    def embed_batch(self, inputs):
        pooler = POOLERS[self.pooler]()
        results = []
        for inp in inputs:
            context = inp.T.unsqueeze(0)  # (1, C, T), Chronos-2 input format
            with torch.no_grad():
                embeddings, _ = self.model.embed(context)  # list[(C, T_tok, D)]
            emb_np = torch.stack(list(embeddings))[0].float().cpu().numpy()
            # Reshape to (1, T_tok, C, D) to match pooler convention (..., T, V, D)
            emb_4d = emb_np.transpose(1, 0, 2)[None]  # (1, T_tok, C, D)
            pooled = pooler.pool(emb_4d)  # (1, C, D)
            results.append(
                torch.from_numpy(pooled[0].reshape(-1).astype(np.float32))  # (C*D,)
            )
        return results
