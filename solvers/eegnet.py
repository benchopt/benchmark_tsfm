"""EEGNet solver for time series classification.

Uses the ``braindecode`` implementation of EEGNetv4 as a classifier on
multivariate time series of shape (N, T, C).

References:
    https://braindecode.org/
    https://arxiv.org/abs/1611.08024
"""

import numpy as np
import torch
from benchopt import BaseSolver

SUPPORTED_TASKS = {"classification"}


class Solver(BaseSolver):
    """EEGNet time series classification solver.

    The model is built once in ``set_objective`` (not timed). During
    ``run`` the network is trained on the training set.
    """

    name = "EEGNet"

    requirements = [
        "pip::braindecode",
        "pip::torch",
    ]

    parameters = {
        "n_epochs": [50],
        "batch_size": [32],
        "lr": [1e-3],
    }

    def skip(self, task, **kwargs):
        if task not in SUPPORTED_TASKS:
            return True, f"EEGNet solver does not support task={task!r}"
        return False, None

    def set_objective(self, task, X_train, y_train, **meta):
        """Prepare the solver for a given dataset configuration.

        Model construction is done here (not inside ``run``) so that
        the build time is excluded from the benchmark timing.
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

        # Infer input dimensions directly from the training data.
        # Within a dataset all series share the same (T, C) shape.
        X0 = np.asarray(X_train[0])
        n_times = X0.shape[0]
        n_channels = X0.shape[1] if X0.ndim == 2 else 1
        n_classes = int(meta.get("n_classes", len(np.unique(y_train))))

        # Build the network

        try:
            from braindecode.models import EEGNet

            network = EEGNet(
                n_chans=n_channels,
                n_outputs=n_classes,
                n_times=n_times,
            )
            self._network = network.to(device)
            print(
                f"✓ EEGNet built: C={n_channels}, T={n_times}, "
                f"n_classes={n_classes} on device: {device}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to build EEGNet: {e}. Make sure braindecode "
                "is installed."
            )

        self._device = device
        self._optimizer = torch.optim.Adam(
            self._network.parameters(), lr=self.lr
        )
        self._criterion = torch.nn.CrossEntropyLoss()

    def run(self, _):
        """Fit the model on the training data."""
        X = self._prepare_inputs(np.asarray(self.X_train, dtype=np.float32))
        y = np.asarray(self.y_train, dtype=np.int64)

        X_t = torch.tensor(X, dtype=torch.float32, device=self._device)
        y_t = torch.tensor(y, dtype=torch.long, device=self._device)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self._network.train()
        for _ in range(self.n_epochs):
            for xb, yb in loader:
                self._optimizer.zero_grad()
                logits = self._network(xb)
                loss = self._criterion(logits, yb)
                loss.backward()
                self._optimizer.step()

    def _prepare_inputs(self, X_batch):
        """Reshape inputs to EEGNet's expected layout.

        EEGNet expects arrays of shape (N, C, T). Inputs arrive as
        (N, T, C); within a dataset all series share the same length,
        so no interpolate needed.
        """
        return X_batch.transpose(0, 2, 1)

    def get_result(self):
        """Return a predictor exposing ``predict(X_test) -> (N,)``."""
        return {
            "model": EEGNetAdapter(
                self._network,
                self._device,
                self._prepare_inputs,
                self.batch_size,
            )
        }


class EEGNetAdapter:
    """Wraps a trained EEGNet to expose ``predict(X) -> (N,)`` class indices.

    ``X`` arrives as a list / array of (T, C) series, matching the
    classification data contract in ``objective.py``.
    """

    def __init__(self, network, device, prepare_inputs, batch_size):
        self._network = network
        self._device = device
        self._prepare_inputs = prepare_inputs
        self._batch_size = batch_size

    def predict(self, X):
        X = self._prepare_inputs(np.asarray(X, dtype=np.float32))
        X_t = torch.tensor(X, dtype=torch.float32, device=self._device)

        self._network.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_t), self._batch_size):
                logits = self._network(X_t[i:i + self._batch_size])
                preds.append(logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(preds)
