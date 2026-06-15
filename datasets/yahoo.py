from benchopt import BaseDataset

from benchmark_utils.download import fetch_tsb_uad, load_data_tsb_uad
from benchmark_utils.metrics import AD_METRICS


class Dataset(BaseDataset):
    name = "YAHOO"

    requirements = ["pip::pooch", "pip::tqdm"]

    parameters = {
        "record_ids": [["all"]],
        "debug": [False],
        "train_ratio": [0.1],
        "number": [5],
    }

    def get_data(self):
        """Load the YAHOO dataset."""

        path = fetch_tsb_uad("YAHOO")
        X_train, X_test, y_test = load_data_tsb_uad(
            path=path,
            records_ids=self.record_ids,
            train_ratio=self.train_ratio,
            number=self.number,
        )

        if len(X_test) == 0:
            raise ValueError("No valid YAHOO records")

        if self.debug:
            X_train = [x[:100] for x in X_train]
            X_test = [x[:100] for x in X_test]
            y_test = [y[:100] for y in y_test]

        return dict(
            X_train=X_train,
            y_train=None,
            y_test=y_test,
            X_test=X_test,
            task="anomaly_detection",
            metrics=AD_METRICS.keys(),
        )
