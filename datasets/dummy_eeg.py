"""Synthetic EEG classification dataset for fast, offline testing.

Windows of 1 s at 200 Hz over 4 electrodes of the 10-20 system, in µV. The
class is whether a 10 Hz alpha rhythm runs on the occipital electrodes, on top
of white noise everywhere. It gives the EEG solvers a dataset to run on in CI,
where the real EEG datasets are too long to download. It stays small, as the
tabular models flatten each window into channels x times features.
"""

import numpy as np
from benchopt import BaseDataset

CH_NAMES = ["Fz", "Cz", "O1", "O2"]
SFREQ = 200.0


class Dataset(BaseDataset):
    name = "Dummy-EEG"
    # Read by the solvers' skip without loading the data.
    task = "classification"

    requirements = []

    # ``debug`` is accepted and ignored, the dataset is already tiny.
    parameters = {
        "random_state": [0],
        "debug": [False],
    }

    n_train = 32
    n_test = 16

    @property
    def meta(self):
        """Attributes read by the solvers' skip without loading the data."""
        return {"n_classes": 2, "multivariate": True, "modality": "eeg"}

    def get_data(self):
        rng = np.random.default_rng(self.random_state)
        t = np.arange(int(SFREQ)) / SFREQ
        occipital = np.isin(CH_NAMES, ["O1", "O2"])

        def make(n):
            y = np.arange(n) % 2
            X = rng.normal(0, 10, (n, len(t), len(CH_NAMES)))
            alpha = 20 * np.sin(
                2 * np.pi * 10 * t[:, None] + rng.uniform(0, 6, (n, 1, 1))
            )
            X[:, :, occipital] += y[:, None, None] * alpha
            return list(X.astype(np.float32)), y.astype(np.int64)

        X_train, y_train = make(self.n_train)
        X_test, y_test = make(self.n_test)
        return dict(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            task=self.task,
            metrics=["accuracy", "balanced_accuracy", "f1_weighted"],
            n_classes=2,
            ch_names=CH_NAMES,
            sfreq=SFREQ,
        )
