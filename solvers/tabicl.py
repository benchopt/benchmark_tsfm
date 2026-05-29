"""TabICL-v2 solver for UCR classification.

TabICL is a tabular foundation model that uses in-context learning (ICL) to
classify new samples in a single forward pass. For the current UCR scope,
datasets are univariate and fixed-length, so each ``(T, 1)`` sample is
reshaped into a flat ``(T,)`` feature row and the full training matrix is
passed to ``TabICLClassifier``.

References:
    https://github.com/soda-inria/tabicl
    https://huggingface.co/tabicl
"""

import numpy as np
import torch
from benchopt import BaseSolver

from tabicl import TabICLClassifier

SUPPORTED_TASKS = {"classification"}


class _TabICLAdapter:
    """Wrap a fitted TabICLClassifier behind the benchmark adapter API.

    ``predict`` receives the entire test list in one call because TabICL is
    an in-context learner — all test rows are scored jointly against the
    stored training context.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(_to_tabular(X))


def _to_tabular(X):
    """Convert a list of benchmark samples into a 2-D feature matrix.

    Parameters
    ----------
    X : list of np.ndarray, each shape (T, C)
        UCR samples.  ``C`` is 1 for the current scope.

    Returns
    -------
    np.ndarray, shape (N, T * C)
        Flat tabular matrix ready for TabICL.
    """
    arr = np.asarray(X, dtype=np.float32)  # (N, T) or (N, T, C)
    if arr.ndim == 3:
        arr = arr.reshape(arr.shape[0], -1)  # flatten C into features
    return arr


class Solver(BaseSolver):
    """TabICL-v2 in-context learning solver for UCR classification.

    The classifier is instantiated once in ``set_objective`` (not timed) so
    that any checkpoint download is excluded from benchmark timing. The
    actual ``fit`` — which stores the training context — runs in ``run``.
    """

    name = "TabICL-v2"

    requirements = ["pip::tabicl"]

    sampling_strategy = "run_once"

    parameters = {
        "checkpoint_version": ["tabicl-classifier-v2-20260212.ckpt"],
        "n_estimators": [8],
    }

    def skip(self, task, **kwargs):
        if task not in SUPPORTED_TASKS:
            return True, f"TabICL solver does not support task={task!r}"
        return False, None

    def set_objective(self, task, X_train, y_train, **meta):
        """Prepare the solver for a given dataset configuration.

        Classifier instantiation is done here (not inside ``run``) so that
        the checkpoint download/init time is excluded from benchmark timing.
        """
        self.task = task
        self.X_train = X_train
        self.y_train = y_train
        self.meta = meta

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Reinstantiate when the checkpoint version or n_estimators changes.
        current_key = (self.checkpoint_version, self.n_estimators)
        should_reload = (
            not hasattr(self, "_classifier")
            or not hasattr(self, "_loaded_key")
            or self._loaded_key != current_key
        )
        if should_reload:
            try:
                self._classifier = TabICLClassifier(
                    checkpoint_version=self.checkpoint_version,
                    n_estimators=self.n_estimators,
                    device=device,
                    random_state=42,
                    verbose=False,
                )
                self._loaded_key = current_key
                print(
                    f"\u2713 TabICL checkpoint ready: {self.checkpoint_version} "
                    f"on device: {device}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialise TabICL checkpoint "
                    f"'{self.checkpoint_version}': {e}. "
                    "Make sure you have internet access and tabicl is installed."
                ) from e

        self._device = device
        self._adapter = _TabICLAdapter(self._classifier)

    def run(self, _):
        """Fit TabICL on the training data (timed)."""
        X_fit = _to_tabular(self.X_train)
        y_fit = np.asarray(self.y_train)
        self._classifier.fit(X_fit, y_fit)

    def get_result(self):
        """Return the fitted adapter wrapping the TabICL classifier."""
        return {"model": self._adapter}
