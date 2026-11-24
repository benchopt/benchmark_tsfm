"""EEGNet, a compact EEG network trained from scratch for classification.

The baseline for the EEG foundation models: braindecode's EEGNet, trained
with Adam on the training windows, each standardized per channel.

References:
    https://arxiv.org/abs/1611.08024
    https://braindecode.org/
"""

import numpy as np
import torch
from benchopt import BaseSolver
from braindecode.models import EEGNet

from benchmark_utils.adapters.base import BaseTSFMAdapter


def _to_tensor(X, device):
    """(N, T, C) windows to an (N, C, T) tensor, standardized per channel."""
    x = torch.from_numpy(np.stack(X).astype(np.float32)).transpose(1, 2)
    x = (x - x.mean(dim=2, keepdim=True)) / x.std(dim=2, keepdim=True).clamp_min(1e-6)
    return x.to(device)


class _EEGNetClassifier(BaseTSFMAdapter):
    """Predict the class of each window with a trained EEGNet."""

    def __init__(self, network, device, batch_size):
        self.network = network
        self.device = device
        self.batch_size = batch_size

    def predict(self, x):
        X = _to_tensor(x, self.device)
        self.network.eval()
        with torch.no_grad():
            logits = [
                self.network(X[i : i + self.batch_size])
                for i in range(0, len(X), self.batch_size)
            ]
        return torch.cat(logits).argmax(dim=1).cpu().numpy()


class Solver(BaseSolver):
    """EEGNet trained from scratch."""

    name = "EEGNet"

    requirements = ["pip::braindecode"]

    sampling_strategy = "run_once"

    parameters = {
        "n_epochs": [50],
        "batch_size": [32],
        "lr": [1e-3],
    }

    def skip(self):
        reason = None
        if self.dataset_.task != "classification":
            reason = f"{self.name} does not support task={self.dataset_.task!r}"
        elif getattr(self.dataset_, "meta", {}).get("modality") != "eeg":
            reason = f"{self.name} is an EEG model"
        return reason

    def set_objective(self, X_train, y_train, task, **meta):
        self.X_train = X_train
        self.y_train = y_train
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        n_times, n_chans = np.asarray(X_train[0]).shape
        self.network = EEGNet(
            n_chans=n_chans, n_outputs=meta["n_classes"], n_times=n_times
        ).to(self.device)

    def run(self, _):
        X = _to_tensor(self.X_train, self.device)
        y = torch.as_tensor(np.asarray(self.y_train), device=self.device)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, y),
            batch_size=self.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.network.train()
        for _ in range(self.n_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(self.network(xb), yb)
                loss.backward()
                optimizer.step()

    def get_result(self):
        return {
            "model": _EEGNetClassifier(self.network, self.device, self.batch_size)
        }
