"""UCR Archive time series classification dataset.

Uses tslearn to download and load any UCR/UEA dataset by name.
Each time series is returned as a (T, C) numpy array; labels are integers.

Data contract output
--------------------
X_train : List[np.ndarray (T, C)]   one array per training sample
y_train : np.ndarray (N,) int       class labels
X_test  : List[np.ndarray (T, C)]
y_test  : np.ndarray (M,) int
task    : "classification"
metrics : ["accuracy", "balanced_accuracy", "f1_weighted"]
n_classes : int
"""
from benchopt import BaseDataset

import numpy as np
from sklearn.preprocessing import LabelEncoder
from tslearn.datasets import UCR_UEA_datasets


class Dataset(BaseDataset):
    """UCR Archive classification dataset.

    Parameters
    ----------
    dataset_name : str
        Name of a UCR/UEA dataset (e.g. "ECG5000", "GunPoint").
    debug : bool
        If True, keep only the first 20 training samples for fast testing.
    """

    name = "UCR"

    requirements = ["pip::tslearn"]

    parameters = {
        "dataset_name": ["ECG5000"],
        "debug": [False],
    }

    def get_data(self):

        loader = UCR_UEA_datasets()
        X_tr, y_tr, X_te, y_te = loader.load_dataset(self.dataset_name)

        # tslearn returns (N, T, C) — already the right layout.
        X_tr = np.asarray(X_tr, dtype=np.float32)
        X_te = np.asarray(X_te, dtype=np.float32)

        # Encode string labels to consecutive integers.
        le = LabelEncoder()
        y_tr_enc = le.fit_transform(y_tr).astype(np.int64)
        y_te_enc = le.transform(y_te).astype(np.int64)

        if self.debug:
            X_tr = X_tr[:20]
            y_tr_enc = y_tr_enc[:20]

        # Convert to list of (T, C) arrays so variable-length datasets work too
        X_train, X_test = list(X_tr), list(X_te)

        return dict(
            X_train=X_train,
            y_train=y_tr_enc,
            X_test=X_test,
            y_test=y_te_enc,
            task="classification",
            metrics=["accuracy", "balanced_accuracy", "f1_weighted"],
            n_classes=int(len(le.classes_)),
        )
