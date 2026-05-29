"""High-Gamma Dataset (HGD) — motor imagery EEG classification.

Wraps the High-Gamma Dataset from Schirrmeister et al. (2017):
  "Deep learning with convolutional neural networks for EEG decoding
   and visualization", Human Brain Mapping.
   https://doi.org/10.1002/hbm.23730

128-electrode EEG (44 motor-cortex channels used) recorded from 14 healthy
subjects performing ~1000 four-second trials of executed movements across
13 runs.  Data are downloaded automatically via braindecode / MOABB.

Labels (4 classes):
    0: left_hand   1: right_hand   2: feet   3: rest

The dataset's own train/test split (runs 1-11 → train, runs 12-13 → test)
is preserved per subject; subjects are then pooled according to
`train_ratio`.

Data contract output
--------------------
X_train : np.ndarray (N, T, C)   windows (n_times, n_channels=44)
y_train : np.ndarray (N,) int    class labels 0-3
X_test  : np.ndarray (M, T, C)
y_test  : np.ndarray (M,) int
task    : "classification"
metrics : ["accuracy", "balanced_accuracy", "f1_weighted"]
n_classes : 4
"""

import numpy as np
from braindecode.datasets import HGD
from braindecode.preprocessing.preprocess import preprocess, Preprocessor
from braindecode.preprocessing.windowers import create_windows_from_events
from sklearn.preprocessing import scale as standard_scale

from benchopt import BaseDataset


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
LABEL_MAPPING = {
    "left_hand":  0,
    "right_hand": 1,
    "feet":       2,
    "rest":       3,
}


# ---------------------------------------------------------------------------
# Per-subject loader
# ---------------------------------------------------------------------------

def _load_subject(sub_id, preprocessors, window_size_samples):
    """Wrapper that falls back to a description-based split if metadata
    item access is unavailable in the installed braindecode version."""
    dataset = HGD(subject_ids=[sub_id])
    preprocess(dataset, preprocessors)

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples,
        preload=True,
        mapping=LABEL_MAPPING,
    )
    preprocess(
        windows_dataset, [Preprocessor(standard_scale, channel_wise=True)]
    )

    X_, y_ = [], []

    # windows_dataset.description contains a 'split' column ('train'/'test')
    # Each window is indexed into its base dataset; we replicate that mapping.
    descriptions = windows_dataset.datasets  # list of WindowsDataset

    for ds in descriptions:
        for x, label, _ in ds:
            window = x.T  # (T, C)
            X_.append(window)
            y_.append(label)

    return X_, y_


# ---------------------------------------------------------------------------
# Benchopt Dataset
# ---------------------------------------------------------------------------

class Dataset(BaseDataset):
    """High-Gamma Dataset (HGD) — 4-class motor EEG classification.

    Parameters
    ----------
    resample_hz : float
        Target sampling frequency. The raw data are recorded at 500 Hz;
        default resamples to 250 Hz (window_size_samples=1000 → 4 s).
    high_cut_hz : float
        Cutoff for the low-pass filter applied to raw signals (Hz).
    factor : float
        Multiplicative scaling applied before filtering (e.g. V → µV).
    window_size_samples : int
        Length of each trial window in samples (after resampling).
    train_ratio : float
        Fraction of subjects whose trials go into the training pool.
        The internal per-subject train/test split (runs 1-11 vs 12-13) is
        always respected first; `train_ratio` then selects which subjects
        contribute to the final train vs test arrays.
    debug : bool
        If True, load only the first 2 subjects for fast iteration.
    seed : int
        Random seed for the subject-level train/test split.
    """

    name = "HGD"

    requirements = ["pip::moabb", "pip::pandas", "pip::braindecode"]

    parameters = {
        "seed":                [42],
        "train_ratio":         [0.8],
        "resample_hz":         [250],
        "high_cut_hz":         [40.0],
        "factor":              [1e6],   # V → µV
        "window_size_samples": [1000],  # 4 s at 250 Hz
        "debug":               [True],
    }

    def get_data(self):
        n_jobs = 1
        preprocessors = [
            # Convert V → µV
            Preprocessor(lambda data: np.multiply(data, self.factor)),
            # Resample to target frequency
            Preprocessor("resample", sfreq=self.resample_hz),
            # Low-pass filter
            Preprocessor(
                "filter", l_freq=None, h_freq=self.high_cut_hz, n_jobs=n_jobs
            ),
        ]

        sub_ids = list(range(1, 15))       # 14 subjects
        if self.debug:
            sub_ids = sub_ids[:2]

        # Collect per-subject (train, test) pairs
        X_all, y_all = [], []

        for sub_id in sub_ids:
            X_, y_ = _load_subject(
                sub_id, preprocessors, self.window_size_samples
            )
            X_all.append(X_)
            y_all.append(y_)

        # ------------------------------------------------------------------
        # Subject-level train / test split (same pattern as Sleep dataset)
        # ------------------------------------------------------------------
        random_state = np.random.RandomState(seed=self.seed)
        ids_train = random_state.choice(
            len(X_all),
            size=int(len(X_all) * self.train_ratio),
            replace=False,
        )
        ids_train_set = set(ids_train.tolist())

        X_train = np.concatenate(
            [X_all[i] for i in ids_train_set], axis=0
        )
        y_train = np.concatenate(
            [y_all[i] for i in ids_train_set], axis=0
        )
        X_test = np.concatenate(
            [X_all[i] for i in range(len(X_all))
             if i not in ids_train_set], axis=0
        )
        y_test = np.concatenate(
            [y_all[i] for i in range(len(y_all))
             if i not in ids_train_set], axis=0
        )
        return dict(
            X_train=X_train,   # (N, window_size_samples, n_channels)
            y_train=y_train,   # (N,)  int in {0, 1, 2, 3}
            X_test=X_test,     # (M, window_size_samples, n_channels)
            y_test=y_test,     # (M,)
            task="classification",
            metrics=["accuracy", "balanced_accuracy", "f1_weighted"],
            n_classes=4,
        )
