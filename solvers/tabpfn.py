"""TabPFN-v2 solver for UCR/UEA classification.

TabPFN is a Prior-fitted Network (PFN) — a Transformer pre-trained over
synthetic tabular datasets via meta-learning. For the UCR/UEA scope each
``(T, C)`` time-series sample is reshaped into a flat ``(T*C,)`` feature row
and the full training matrix is passed to ``TabPFNClassifier``.

References:
    https://github.com/PriorLabs/TabPFN
    https://doi.org/10.1038/s41586-024-08328-6
"""

import numpy as np
from benchopt import BaseSolver

from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

SUPPORTED_TASKS = {"classification"}


class _TabPFNAdapter:
    """Wrap a fitted TabPFNClassifier behind the benchmark adapter API.

    ``predict`` receives the entire test list in one call — scoring all test
    rows at once is significantly faster than individual calls because TabPFN
    recomputes the training context on every call.
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
        UCR/UEA samples.  ``C`` is 1 for univariate datasets.

    Returns
    -------
    np.ndarray, shape (N, T * C)
        Flat tabular matrix ready for TabPFN.
    """
    arr = np.asarray(X, dtype=np.float32)  # (N, T) or (N, T, C)
    if arr.ndim == 3:
        arr = arr.reshape(arr.shape[0], -1)  # flatten C into features
    return arr


class Solver(BaseSolver):
    """TabPFN-v2 in-context learning solver for UCR/UEA classification.

    The classifier is instantiated once in ``set_objective`` (not timed) so
    that any checkpoint download is excluded from benchmark timing. The
    actual ``fit`` — which stores the training context — runs in ``run``.
    """

    name = "TabPFN-v2"

    requirements = ["pip::tabpfn"]

    sampling_strategy = "run_once"

    parameters = {
        "n_estimators": [8],
    }

    def skip(self, task, **kwargs):
        if task not in SUPPORTED_TASKS:
            return True, f"TabPFN solver does not support task={task!r}"
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

        # Reinstantiate only when n_estimators changes.
        should_reload = (
            not hasattr(self, "_classifier")
            or not hasattr(self, "_loaded_n_estimators")
            or self._loaded_n_estimators != self.n_estimators
        )
        if should_reload:
            try:
                self._classifier = TabPFNClassifier.create_default_for_version(
                    ModelVersion.V2,
                    n_estimators=self.n_estimators,
                    device="auto",
                    random_state=42,
                    ignore_pretraining_limits=True,
                )
                self._loaded_n_estimators = self.n_estimators
                print(
                    f"\u2713 TabPFN v2 ready "
                    f"(n_estimators={self.n_estimators}, device=auto)"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialise TabPFN v2: {e}. "
                    "Make sure tabpfn is installed and the v2 checkpoint "
                    "is available in the local cache (~/.cache/tabpfn/)."
                ) from e

        self._adapter = _TabPFNAdapter(self._classifier)

    def run(self, _):
        """Fit TabPFN on the training data (timed)."""
        X_fit = _to_tabular(self.X_train)
        y_fit = np.asarray(self.y_train)
        self._classifier.fit(X_fit, y_fit)

    def get_result(self):
        """Return the fitted adapter wrapping the TabPFN classifier."""
        return {"model": self._adapter}
