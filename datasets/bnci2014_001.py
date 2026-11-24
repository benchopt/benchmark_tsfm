"""BNCI 2014-001 (BCI Competition IV 2a) motor imagery, from MOABB.

The raw recordings of the 9 subjects are band-pass filtered between 4 and
40 Hz and cut into trials starting 0.5 s before each cue. Subjects are drawn
at random between train and test. The same recordings, pre-windowed by Open
EEG Bench with its own split, are the ``bcic2a`` dataset of ``OpenEEG``.

Data contract output: see :mod:`benchmark_utils.eeg`, plus ``n_classes``.
"""

import numpy as np
from benchopt import BaseDataset
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import Preprocessor, preprocess
from braindecode.preprocessing.windowers import create_windows_from_events

from benchmark_utils.eeg import split_subjects


class Dataset(BaseDataset):
    """BNCI 2014-001 motor imagery, 4 classes.

    Parameters
    ----------
    train_ratio : float
        Fraction of the subjects in the training set.
    seed : int
        Seed of the subject split.
    debug : bool
        If True, load only subjects 1 and 2.
    """

    name = "BNCI2014_001"
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
        subjects = [1, 2] if self.debug else list(range(1, 10))
        dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=subjects)
        preprocess(
            dataset,
            [
                Preprocessor("pick_types", eeg=True, meg=False, stim=False),
                Preprocessor(lambda data: np.multiply(data, 1e6)),  # V to µV
                Preprocessor("filter", l_freq=4.0, h_freq=40.0),
            ],
        )
        raw = dataset.datasets[0].raw
        sfreq = raw.info["sfreq"]
        windows = create_windows_from_events(
            dataset,
            trial_start_offset_samples=int(-0.5 * sfreq),
            trial_stop_offset_samples=0,
            preload=False,
        )
        X, y = [], []
        for subject in windows.split("subject").values():
            X.append([x.T.astype(np.float32) for x, _, _ in subject])
            y.append([label for _, label, _ in subject])
        return dict(
            **split_subjects(X, y, self.train_ratio, self.seed),
            task=self.task,
            metrics=["accuracy", "balanced_accuracy", "f1_weighted"],
            n_classes=4,
            ch_names=list(raw.ch_names),
            sfreq=float(sfreq),
        )
