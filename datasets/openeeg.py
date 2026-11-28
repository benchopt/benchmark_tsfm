"""EEG classification datasets of Open EEG Bench.

The datasets are pre-windowed by braindecode and hosted on the Hugging Face
Hub. Each window is returned as a (T, C) array in µV, and the split keeps the
subjects of the test set out of the training set, as in Open EEG Bench. Its
validation subjects join the training set.

Data contract output
--------------------
X_train  : List[np.ndarray (T, C)]   one array per window
y_train  : np.ndarray (N,) int
X_test   : List[np.ndarray (T, C)]
y_test   : np.ndarray (M,) int
task     : "classification"
n_classes : int
ch_names : List[str]   electrode names, in the channel order of the arrays
sfreq    : float       sampling frequency in Hz

References:
    https://github.com/braindecode/OpenEEGBench
"""

import numpy as np

# The hub extra is imported here, so benchopt sees it missing and installs it
import huggingface_hub  # noqa: F401
import zarr  # noqa: F401
from benchopt import BaseDataset
from braindecode.datasets import BaseConcatDataset

# Pinned revision, classes and subject split of each dataset, as in
# ``open_eeg_bench.default_configs.datasets`` (v0.6.0).
DATASETS = {
    # Mental arithmetic vs rest, 36 subjects.
    "arithmetic_zyma2019": dict(
        revision="8359206e7e1b3d1f068033348e291d6677be1920",
        n_classes=2,
        train=[f"{i:02d}" for i in range(32)],
        test=[f"{i:02d}" for i in range(32, 36)],
    ),
    # BCI Competition IV 2a, 4-class motor imagery, 9 subjects.
    "bcic2a": dict(
        revision="3585206f74463e787688a0f4afd12605c9be700c",
        n_classes=4,
        train=[1, 2, 3, 4, 5, 6],
        test=[7, 8, 9],
    ),
}


class Dataset(BaseDataset):
    """Open EEG Bench classification dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset on the braindecode Hub, a key of ``DATASETS``.
    debug : bool
        If True, keep only 20 training and 20 test windows.
    """

    name = "OpenEEG"
    # Read by the solvers' skip without loading the data.
    task = "classification"

    requirements = ["pip::braindecode[hub]"]

    parameters = {
        "dataset_name": list(DATASETS),
        "debug": [False],
    }

    prepare_cache_ignore = ("debug",)

    @property
    def meta(self):
        """Attributes read by the solvers' skip without loading the data."""
        return {
            "n_classes": DATASETS[self.dataset_name]["n_classes"],
            "multivariate": True,
            "modality": "eeg",
        }

    def prepare(self):
        self._load()

    def _load(self):
        return BaseConcatDataset.pull_from_hub(
            f"braindecode/{self.dataset_name}",
            preload=True,
            revision=DATASETS[self.dataset_name]["revision"],
        )

    def get_data(self):
        config = DATASETS[self.dataset_name]
        windows = self._load()
        metadata = windows.get_metadata().reset_index(drop=True)

        X = [np.asarray(windows[i][0], dtype=np.float32).T for i in range(len(windows))]
        y = metadata["target"].to_numpy().astype(np.int64)
        train = np.flatnonzero(metadata["subject"].isin(config["train"]))
        test = np.flatnonzero(metadata["subject"].isin(config["test"]))
        if self.debug:
            rng = np.random.default_rng(0)
            train = rng.choice(train, size=20, replace=False)
            test = rng.choice(test, size=20, replace=False)

        recording = windows.datasets[0]
        raw = getattr(recording, "raw", None)
        info = (raw if raw is not None else recording.windows).info
        return dict(
            X_train=[X[i] for i in train],
            y_train=y[train],
            X_test=[X[i] for i in test],
            y_test=y[test],
            task=self.task,
            metrics=["accuracy", "balanced_accuracy", "f1_weighted"],
            n_classes=config["n_classes"],
            ch_names=list(info["ch_names"]),
            sfreq=float(info["sfreq"]),
        )
