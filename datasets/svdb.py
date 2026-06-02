import numpy as np
import pandas as pd
from pathlib import Path

from benchopt import BaseDataset
from benchopt.config import get_data_path
from benchmark_utils.download import fetch_tsb_uad, load_data_tsb_uad
from benchmark_utils.metrics import AD_METRICS


class Dataset(BaseDataset):
    name = "SVDB"

    requirements = ["pip:pooch"]

    parameters = {
        "record_ids": [["all"]],
        "debug": [False],
        "train_ratio": [0.1],
        "number": [5],
    }

    def get_data(self):
        """Load the SVDB dataset."""

        try:
            path = fetch_tsb_uad("SVDB")
        except ImportError:
            path = get_data_path("SVDB")

        X_train, X_test, y_test = load_data_tsb_uad(
            path=path,
            records_ids=self.record_ids,
            train_ratio=self.train_ratio,
            number=self.number,
        )

        if len(X_test) == 0:
            raise ValueError("No valid SVDB records")

        return dict(
            X_train=X_train,
            y_train=None,
            y_test=y_test,
            X_test=X_test,
            task="anomaly_detection",
            metrics=AD_METRICS.keys(),
        )
