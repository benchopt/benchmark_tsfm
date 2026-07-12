"""Synthetic multivariate classification dataset for fast, offline testing.

Generated in-memory (no download), this dataset exists mainly so the
benchmark's classification task — and classification-only solvers such as
Mantis — always have something that runs in CI, independent of any external
download host.

Each sample is a multivariate series with ``C = 3`` channels: two "targets"
and one "covariate". The class label is encoded in the *trend* of the second
target channel — rising, falling, or flat for the three classes — which also
shifts that channel's mean, so even simple time-pooled features can separate
the classes. The remaining target and the covariate are pure noise.
"""

import numpy as np
from benchopt import BaseDataset


class Dataset(BaseDataset):
    name = "Dummy-Classification"

    # Pure-numpy synthetic data — no external requirements, no network.
    requirements = []

    # ``debug`` is accepted (and ignored) so the benchmark-wide test_config
    # ``debug=True`` override applies cleanly; the dataset is already tiny.
    parameters = {
        "random_state": [0],
        "debug": [False],
    }

    # Fixed structure: 3 classes (rising / falling / same trend) over 3 channels
    # (target_0 = noise, target_1 = class-encoding trend, covariate = noise).
    n_train = 64
    n_test = 32
    length = 128
    n_classes = 3
    _amplitude = 2.0
    _noise = 0.3

    def get_data(self):
        rng = np.random.default_rng(self.random_state)

        # class -> trend slope of the signal channel (rising / falling / same)
        slopes = np.array([1.0, -1.0, 0.0])

        def _make(n):
            # balanced, deterministic class assignment
            y = np.arange(n) % self.n_classes
            X = []
            for c in y:
                ramp = np.linspace(0.0, slopes[c] * self._amplitude, self.length)
                target_signal = ramp + rng.normal(0, self._noise, self.length)
                target_noise = rng.normal(0, self._noise, self.length)
                covariate = rng.normal(0, self._noise, self.length)
                # (T, C=3): [target_noise, target_signal (class trend), covariate]
                series = np.stack(
                    [target_noise, target_signal, covariate], axis=1
                ).astype(np.float32)
                X.append(series)
            return X, y.astype(np.int64)

        X_train, y_train = _make(self.n_train)
        X_test, y_test = _make(self.n_test)

        return dict(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            task="classification",
            metrics=["accuracy", "balanced_accuracy", "f1_weighted"],
            n_classes=self.n_classes,
        )
