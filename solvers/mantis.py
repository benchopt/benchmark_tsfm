"""Mantis solver for time series classification on UCR datasets.

Uses the official ``mantis-tsfm`` API to load a pretrained Mantis checkpoint,
extract embeddings with ``MantisTrainer.transform``, and train a Random Forest
classifier on top.

References:
    https://huggingface.co/paris-noah/Mantis-8M
    https://github.com/vfeofanov/mantis
"""

import numpy as np
import torch
from benchopt import BaseSolver
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer

from mantis.trainer import MantisTrainer
from mantis.architecture import MantisV1, MantisV2

SUPPORTED_TASKS = {"classification"}


class Solver(BaseSolver):
    """Mantis time series classification solver with Random Forest.

    The model is loaded once in ``set_objective`` (not timed). Training
    embeddings are extracted and a Random Forest classifier is trained.
    During ``run`` the predictions are generated for the test set.
    """

    name = "Mantis-RandomForest"

    # mantis-tsfm and torch are required to load the model and run inference.
    requirements = [
        "pip::mantis-tsfm>=1.0.0",
    ]

    parameters = {
        "checkpoint": ["paris-noah/Mantis-8M"],
        "batch_size": [32],
        "n_estimators": [100],
        "interpolate_to": [512],
    }

    def skip(self, task, **kwargs):
        if task not in SUPPORTED_TASKS:
            return True, f"Chronos solver does not support task={task!r}"
        return False, None

    def set_objective(self, task, X_train, y_train, **meta):
        """Prepare the solver for a given dataset configuration.

        Model loading is done here (not inside ``run``) so that the
        checkpoint download/loading time is excluded from the benchmark
        timing.
        """
        self.task = task
        self.X_train = X_train
        self.y_train = y_train
        self.meta = meta

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the model only on the first call for this checkpoint.
        should_reload = (
            not hasattr(self, "_network")
            or not hasattr(self, "_loaded_checkpoint")
            or self._loaded_checkpoint != self.checkpoint
        )
        if should_reload:
            try:
                MantisBackbone = (
                    MantisV2 if "MantisV2" in self.checkpoint else MantisV1
                )
                network = MantisBackbone(device=device)
                network = network.from_pretrained(self.checkpoint)

                self._network = network
                self._trainer = MantisTrainer(
                    device=device, network=self._network
                )
                self._loaded_checkpoint = self.checkpoint
                print(
                    f"✓ Mantis checkpoint loaded: {self.checkpoint} "
                    f"on device: {device}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Mantis checkpoint '{self.checkpoint}' "
                    f"from Hugging Face: {e}. Make sure you have internet "
                    "access and the model is available."
                )

        self.model = make_pipeline(
            FunctionTransformer(self._extract_embeddings),
            RandomForestClassifier(
                n_estimators=self.n_estimators,
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
        )

        self._device = device

    def run(self, _):
        """Fit the model on the training data."""
        self.model.fit(self.X_train, self.y_train)

    def _extract_embeddings(self, X):
        """Extract embeddings for a batch of time series.

        Parameters
        ----------
        X : np.ndarray
            Input time series of shape (N, T, C) where N is the number
            of series, T is the sequence length, and C the number of channels.
            Note that Mantis expects (N, C, T) internally.
        batch_size : int
            Batch size for processing

        Returns
        -------
        np.ndarray
            Embeddings of shape (n_samples, embedding_dim)
        """
        batch_size = self.batch_size
        n_samples = len(X)
        all_embeddings = []

        for batch_idx in range(0, n_samples, batch_size):
            batch_end = min(batch_idx + batch_size, n_samples)
            X_batch = np.asarray(X[batch_idx:batch_end], dtype=np.float32)
            X_batch_processed = self._prepare_inputs(X_batch)

            try:
                with torch.no_grad():
                    embeddings_np = self._trainer.transform(X_batch_processed)
                all_embeddings.append(np.asarray(embeddings_np))

            except Exception as e:
                print(f"  Warning: Failed to process batch {batch_idx}: {e}")
                if all_embeddings:
                    embedding_dim = all_embeddings[0].shape[1]
                else:
                    embedding_dim = 128
                all_embeddings.append(np.zeros(
                    (batch_end - batch_idx, embedding_dim), dtype=np.float32)
                )

        # Concatenate all embeddings
        if all_embeddings:
            embeddings_all = np.vstack(all_embeddings)
        else:
            embeddings_all = np.zeros((n_samples, 768))

        return embeddings_all

    def _prepare_inputs(self, X_batch):
        """Ensure Mantis-compatible shape and sequence length.

        Mantis expects arrays of shape (N, C, T), and the sequence length
        should be divisible by 32. Following official guidance,
        we interpolate to ``interpolate_to`` (default 512).
        """
        X_in = X_batch.transpose(0, 2, 1)

        current_len = X_in.shape[-1]
        target_len = int(self.interpolate_to)

        if current_len != target_len:
            tensor = torch.tensor(X_in, dtype=torch.float32)
            tensor = torch.nn.functional.interpolate(
                tensor,
                size=target_len,
                mode="linear",
                align_corners=False,
            )
            X_in = tensor.numpy()

        if X_in.shape[-1] % 32 != 0:
            raise ValueError(
                "Sequence length must be divisible by 32 for Mantis, "
                f"got {X_in.shape[-1]}"
            )

        return X_in

    def get_result(self):
        """Return the classification predictions and probabilities."""
        return {
            "model": self.model
        }
