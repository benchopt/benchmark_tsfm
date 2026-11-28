"""Sleep staging on Sleep Physionet (Sleep-EDF), from braindecode.

Each recording is low-pass filtered at 40 Hz and cut into 30 s windows, with
the wake periods cropped to 30 min around sleep. The labels are the 5 stages
W, N1, N2, N3 (stages 3 and 4 merged, as AASM) and REM. Subjects are drawn
at random between train and test. The two EEG channels are bipolar
derivations (Fpz-Cz, Pz-Oz).

Data contract output: see :mod:`benchmark_utils.eeg`, plus ``n_classes``.
"""

import numpy as np
from benchopt import BaseDataset
from braindecode.datasets import SleepPhysionet
from braindecode.preprocessing import Preprocessor, preprocess
from braindecode.preprocessing.windowers import create_windows_from_events

from benchmark_utils.eeg import split_subjects

MAPPING = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}
# Subjects 1 to 82, without the ones missing from Sleep-EDF.
SUBJECTS = [s for s in range(1, 83) if s not in (39, 68, 69, 78, 79)]


class Dataset(BaseDataset):
    """Sleep Physionet sleep staging, 5 classes.

    Parameters
    ----------
    train_ratio : float
        Fraction of the subjects in the training set.
    seed : int
        Seed of the subject split.
    debug : bool
        If True, load only the first 2 subjects.
    """

    name = "SleepPhysionet"
    # Read by the solvers' skip without loading the data.
    task = "classification"

    requirements = ["pip::braindecode"]

    parameters = {
        "train_ratio": [0.8],
        "seed": [42],
        "debug": [False],
    }

    @property
    def meta(self):
        """Attributes read by the solvers' skip without loading the data."""
        return {
            "n_classes": 5,
            "multivariate": True,
            "modality": "eeg",
            "bipolar": True,
        }

    def get_data(self):
        X, y = [], []
        for subject in SUBJECTS[:2] if self.debug else SUBJECTS:
            dataset = SleepPhysionet(subject_ids=[subject], crop_wake_mins=30)
            preprocess(
                dataset,
                [
                    Preprocessor(lambda data: np.multiply(data, 1e6)),  # V to µV
                    Preprocessor("filter", l_freq=None, h_freq=40.0),
                ],
            )
            windows = create_windows_from_events(
                dataset,
                trial_start_offset_samples=0,
                trial_stop_offset_samples=0,
                window_size_samples=3000,
                window_stride_samples=3000,
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
            n_classes=5,
            ch_names=list(raw.ch_names),
            sfreq=float(raw.info["sfreq"]),
        )
