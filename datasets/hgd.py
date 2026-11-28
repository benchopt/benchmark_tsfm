"""High-Gamma Dataset (HGD), executed movements, from braindecode.

Schirrmeister et al. (2017), "Deep learning with convolutional neural
networks for EEG decoding and visualization", Human Brain Mapping,
https://doi.org/10.1002/hbm.23730.

The motor-cortex channels (names with a C, 44 of the 128) of the 14
subjects are resampled to 250 Hz, low-pass filtered at 40 Hz and cut into
4 s windows. The labels are left hand, right hand, feet and rest. Subjects
are drawn at random between train and test, pooling the dataset's own train
and test runs.

Data contract output: see :mod:`benchmark_utils.eeg`, plus ``n_classes``.
"""

import numpy as np
from benchopt import BaseDataset
from braindecode.datasets import HGD
from braindecode.preprocessing import Preprocessor, preprocess
from braindecode.preprocessing.windowers import create_windows_from_events

from benchmark_utils.eeg import split_subjects

MAPPING = {"left_hand": 0, "right_hand": 1, "feet": 2, "rest": 3}
SFREQ = 250


class Dataset(BaseDataset):
    """High-Gamma Dataset, 4 classes.

    Parameters
    ----------
    train_ratio : float
        Fraction of the subjects in the training set.
    seed : int
        Seed of the subject split.
    debug : bool
        If True, load only the first 2 subjects.
    """

    name = "HGD"
    # Read by the solvers' skip without loading the data.
    task = "classification"

    requirements = ["pip::braindecode", "pip::moabb"]

    parameters = {
        "train_ratio": [0.8],
        "seed": [42],
        "debug": [False],
    }

    @property
    def meta(self):
        """Attributes read by the solvers' skip without loading the data."""
        return {"n_classes": 4, "multivariate": True, "modality": "eeg"}

    def get_data(self):
        X, y = [], []
        for subject in [1, 2] if self.debug else range(1, 15):
            dataset = HGD(subject_ids=[subject])
            preprocess(
                dataset,
                [
                    Preprocessor(
                        lambda raw: raw.pick([c for c in raw.ch_names if "C" in c]),
                        apply_on_array=False,
                    ),
                    Preprocessor(lambda data: np.multiply(data, 1e6)),  # V to µV
                    Preprocessor("resample", sfreq=SFREQ),
                    Preprocessor("filter", l_freq=None, h_freq=40.0),
                ],
            )
            windows = create_windows_from_events(
                dataset,
                trial_start_offset_samples=0,
                trial_stop_offset_samples=0,
                window_size_samples=4 * SFREQ,
                window_stride_samples=4 * SFREQ,
                preload=True,
                mapping=MAPPING,
            )
            X.append([x.T.astype(np.float32) for x, _, _ in windows])
            y.append([label for _, label, _ in windows])
        raw = dataset.datasets[0].raw
        return dict(
            **split_subjects(X, y, self.train_ratio, self.seed),
            task=self.task,
            metrics=["accuracy", "balanced_accuracy", "f1_weighted"],
            n_classes=4,
            ch_names=list(raw.ch_names),
            sfreq=float(raw.info["sfreq"]),
        )
