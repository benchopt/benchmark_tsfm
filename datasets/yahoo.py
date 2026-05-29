import numpy as np
import pandas as pd
from pathlib import Path

from benchopt import BaseDataset
from benchopt.config import get_data_path
from benchmark_utils.download import fetch_tsb_uad
from benchmark_utils.metrics import AD_METRICS


def load_data(db_path, record_ids=None, verbose=False):
    """
    Load data from the database path for specified record IDs.

    Args:
        db_path: Path to the database directory
        record_ids: List of record IDs to load.
        If None, loads all available records.

    Returns:
        tuple: (X, y_true) where:
            - X: numpy array of shape (num_records, num_samples)
            - y_true: numpy array of shape (num_records, num_samples)
    """
    db_path = Path(db_path)

    if record_ids is None:
        record_files = list(db_path.glob("*.data.out"))
        record_ids = [f.name for f in record_files]

    data_list = []
    labels_list = []
    for record_id in record_ids:
        # Handle case where record_id already includes the pattern
        if record_id.endswith('.data.out'):
            pattern = record_id
        else:
            # Create pattern based on the A{record_id} format
            patterns = [
                f"Yahoo_A{record_id}real_*_data.out",
                f"Yahoo_A{record_id}synthetic_*_data.out",
                f"YahooA{record_id}Benchmark-TS*_data.out"
            ]

        # Find all matching files for this record_id
        matching_files = []
        if record_id.endswith('.data.out'):
            matching_files = list(db_path.glob(pattern))
        else:
            for pattern in patterns:
                matching_files.extend(list(db_path.glob(pattern)))

        if not matching_files:
            if verbose:
                print(f"No files found for record {record_id}")
            continue

        for record_file in matching_files:
            if record_file.exists():
                record_data = pd.read_csv(
                    record_file, header=None).dropna().to_numpy()
                # First column is the data, second column is labels
                if record_data.shape[1] >= 2:
                    data_list.append(record_data[:, 0].astype(float))
                    labels_list.append(record_data[:, 1].astype(int))
                else:
                    if verbose:
                        print(f"Insufficient columns for file {record_file}")
            else:
                if verbose:
                    print(f"Record file not found: {record_file}")

    if not data_list:
        raise ValueError("No valid data found")

    max_length = max(len(data) for data in data_list)

    padded_data = []
    padded_labels = []
    for data, labels in zip(data_list, labels_list):
        if len(data) < max_length:
            # Padding with last value for data and 0 for labels
            padded_data.append(
                np.pad(
                    data,
                    (0, max_length - len(data)),
                    mode="constant",
                    constant_values=data[-1],
                )
            )
            padded_labels.append(
                np.pad(
                    labels,
                    (0, max_length - len(labels)),
                    mode="constant",
                    constant_values=0,
                )
            )
        else:
            padded_data.append(data[:max_length])
            padded_labels.append(labels[:max_length])

    return np.array(padded_data), np.array(padded_labels)


class Dataset(BaseDataset):
    name = "YAHOO"

    requirements = ["pip:pooch"]

    parameters = {
        "recordings_id": [["1"]],
        "debug": [False],
    }

    def get_data(self):
        """Load the YAHOO dataset."""

        try:
            path = fetch_tsb_uad("YAHOO")
        except ImportError:
            path = get_data_path("YAHOO")

        # X shape (n_recordings, n_samples)
        # y shape (n_recordings, n_samples)
        X_raw, y_raw = load_data(path, self.recordings_id)

        if X_raw.size == 0:
            raise ValueError("No valid YAHOO records")        

        # Reshaping data to (n_recordings, n_features, n_samples)
        n_recordings = X_raw.shape[0]
        X_test = X_raw.reshape(n_recordings, -1, 1)
        y_test = y_raw.reshape(n_recordings, -1)

        return dict(
            X_train=None,
            y_train=None,
            y_test=y_test,
            X_test=X_test, 
            task="anomaly_detection",
            metrics=AD_METRICS.keys()
        )