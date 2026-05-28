"""ECG anomaly detection dataset from TSB-UAD.

Wraps the ECG recordings from the MIT-BIH / TSB-UAD benchmark.
Each recording is split into a training portion (first 10 %) and a test
portion.  Labels are point-level binary anomaly indicators.

Data contract output
--------------------
X_train : List[np.ndarray (T_i, C)]   training portions  (C == 1)
y_train : None                         unsupervised task
X_test  : List[np.ndarray (T_j, C)]   test portions
y_test  : List[np.ndarray (T_j,)]     point-level binary labels
task    : "anomaly_detection"
metrics : ["auc_roc", "auc_pr", "f1_pa"]
"""

import numpy as np
import pandas as pd
from pathlib import Path

from benchopt import BaseDataset
from benchopt.config import get_data_path
from benchmark_utils.download import fetch_tsb_uad


def _load_records(db_path, record_ids, number):
    db_path = Path(db_path)
    if record_ids in (None, "all", ["all"]):
        record_ids = [f.stem for f in db_path.glob("*.out")
                      if f.stem != "MBA_ECG14046_data"]
    if number > 0:
        record_ids = record_ids[:number]

    X_list, y_list = [], []
    for rid in record_ids:
        path = db_path / f"{rid}.out"
        if not path.exists():
            continue
        data = pd.read_csv(path, header=None).dropna().to_numpy()
        if data.shape[1] < 2:
            continue
        X_list.append(data[:, 0].astype(np.float32))
        y_list.append(data[:, 1].astype(np.int32))
    return X_list, y_list


class Dataset(BaseDataset):
    """ECG anomaly detection dataset (TSB-UAD).

    Parameters
    ----------
    record_ids : list of str or "all"
        Which ECG recordings to include.
    debug : bool
        If True, truncate each recording to 5000 timesteps for fast iteration.
    number : int
        Maximum number of recordings to load (-1 = all).
    train_ratio : float
        Fraction of each recording used as the training (normal) portion.
    """

    name = "ECG"

    requirements = ["pip::pooch", "pandas"]

    parameters = {
        "record_ids": [
            ["MBA_ECG14046_data_1", "MBA_ECG14046_data_2"],
        ],
        "debug": [False],
        "number": [-1],
        "train_ratio": [0.1],
    }

    def get_data(self):

        # Allow reuse of the download helper from benchmark_ad if present,
        # otherwise fall back to the data path directly.
        try:
            path = fetch_tsb_uad("ECG")
        except ImportError:
            path = get_data_path("ECG")

        record_ids = self.record_ids
        X_raw, y_raw = _load_records(path, record_ids, self.number)

        if not X_raw:
            raise ValueError("No valid ECG records found.")

        X_train, X_test, y_test = [], [], []
        for x, y in zip(X_raw, y_raw):
            if self.debug:
                x = x[:5000]
                y = y[:5000]

            split = max(1, int(len(x) * self.train_ratio))

            # Reshape to (T, C=1)
            X_train.append(x[:split].reshape(-1, 1))
            X_test.append(x[split:].reshape(-1, 1))
            y_test.append(y[split:])

        return dict(
            X_train=X_train,
            y_train=None,
            X_test=X_test,
            y_test=y_test,
            task="anomaly_detection",
            metrics=["auc_roc", "auc_pr", "f1_pa"],
        )
