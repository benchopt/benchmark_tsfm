"""P2S (Production Press Sensor Data) time series classification dataset.

Wraps the ``AIML-TUDA/P2S`` Hugging Face dataset (Normal variant): binary
classification of whether a deep-drawing press run produced a normal (``0``) or
defective (``1``) part from its 4096-step force-sensor series.

Data contract output
--------------------
X_train : List[np.ndarray (4096, 1)]   one series per training sample
y_train : np.ndarray (N,) int          class labels (0/1)
X_test  : List[np.ndarray (4096, 1)]
y_test  : np.ndarray (M,) int
task    : "classification"
metrics : ["accuracy", "balanced_accuracy", "f1_weighted"]
n_classes : 2
"""

import numpy as np
import pandas as pd
from benchopt import BaseDataset

# Requirement checks
import fsspec  # noqa: F401
from huggingface_hub import hf_hub_download  # noqa: F401


class Dataset(BaseDataset):
    """P2S press-sensor binary classification dataset (Normal variant).

    Parameters
    ----------
    variant : str
        Which Hub variant to load. Only ``"Normal"`` is shipped.
    debug : bool
        If True, keep only the first 20 samples of each split for fast testing.
    """

    name = "P2S"

    requirements = ["pip::huggingface_hub", "fsspec"]

    parameters = {
        "variant": ["Normal"],
        "debug": [False],
    }

    test_parameters = {
        "variant": ["Normal"],
        "debug": [True],
    }

    def _load_split(self, split):
        """Read one split into a list of ``(4096, 1)`` series and int labels."""
        url = (
            f"hf://datasets/AIML-TUDA/P2S/{self.variant}/{split}-00000-of-00001.parquet"
        )
        df = pd.read_parquet(url, columns=["dowel_deep_drawing_ow", "label"])

        if self.debug:
            df = df.head(20)

        X = [
            np.asarray(cell, dtype=np.float32).reshape(-1, 1)
            for cell in df["dowel_deep_drawing_ow"]
        ]
        y = df["label"].to_numpy(dtype=np.int64)
        return X, y

    def get_data(self):
        X_train, y_train = self._load_split("train")
        X_test, y_test = self._load_split("test")

        return dict(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            task="classification",
            metrics=["accuracy", "balanced_accuracy", "f1_weighted"],
            n_classes=2,
        )
