"""Sleep classification dataset from Sleep Physionet.

Wraps the sleep recordings from the Sleep Physionet.
Each recording is split into a training portion (first 10 %) and a test
portion.
Labels are from 0 to 4, corresponding to the sleep stages W, N1, N2, N3, REM.

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

import numpy as np
from braindecode.datasets import SleepPhysionet
from braindecode.preprocessing.preprocess import preprocess, Preprocessor

from braindecode.preprocessing.windowers import create_windows_from_events
from sklearn.preprocessing import scale as standard_scale

from benchopt import BaseDataset


def _load_subject(
    sub_id, preprocessors, mapping=None, window_size_samples=3000
):
    dataset = SleepPhysionet(subject_ids=[sub_id], crop_wake_mins=30)

    preprocess(dataset, preprocessors)

    # Extract the frequency and channels names
    raw = dataset.datasets[0].raw
    sfreq = raw.info["sfreq"]
    ch_names = raw.ch_names

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples,
        preload=True,
        mapping=mapping,
    )

    preprocess(
        windows_dataset, [Preprocessor(standard_scale, channel_wise=True)]
    )
    all_labels = []
    all_data = []
    for i, x in enumerate(windows_dataset):
        label = x[1]
        all_labels.append(label)

        data = x[0].T
        all_data.append(data)
    return all_data, all_labels, sfreq, ch_names


class Dataset(BaseDataset):
    """Sleep classification dataset (TSB-UAD).

    Parameters
    ----------
    window_size_samples : int
        Length of the windows to split the recordings into.
    sub_ids : List[int]
        Subject IDs to include (from 1 to 82, excluding 39, 68, 69, 78, 79).
    mapping : dict
        Mapping from the original sleep stage labels to integers.
        We merge stages 3 and 4 following AASM standards.
    n_jobs : int
        Number of parallel jobs to use for preprocessing.
    debug : bool
        If True, keep only the first 5000 samples
        of each recording for fast testing.
    high_cut_hz : float
        If not None, apply a low-pass filter with this cutoff frequency (in Hz)
        to the raw signals.
    factor : float
        Factor to multiply the raw signals by (e.g. to convert from V to uV
    train_ratio : float
        Fraction of each recording used as the training (normal) portion.
    """

    name = "Sleep"

    requirements = ["pip::pooch", "pandas", 'braindecode==1.5.1']

    parameters = {
        "seed": [42],
        "train_ratio": [0.8],
        "debug": [False],
    }

    def prepare(self):
        # Allow reuse of the download helper from benchmark_ad if present,
        # otherwise fall back to the data path directly.
        sub_ids = range(1, 83)
        window_size_samples = 3000
        mapping = {  # We merge stages 3 and 4 following AASM standards.
            "Sleep stage W": 0,
            "Sleep stage 1": 1,
            "Sleep stage 2": 2,
            "Sleep stage 3": 3,
            "Sleep stage 4": 3,
            "Sleep stage R": 4,
        }
        n_jobs = 1
        high_cut_hz = 40.0
        factor = 1e6  # Factor to convert from V to uV

        preprocessors = [
            Preprocessor(lambda data: np.multiply(data, factor)),
            Preprocessor(
                "filter", l_freq=None,
                h_freq=high_cut_hz, n_jobs=n_jobs
            ),
        ]

        X_all, y_all = [], []
        sub_ids = sub_ids[:2] if self.debug else sub_ids
        sfreq_ref, ch_names_ref = None, None
        for sub_id in sub_ids:
            if sub_id in [39, 68, 69, 78, 79]:
                continue
            X_, y_, sfreq, ch_names = _load_subject(
                sub_id, preprocessors, mapping, window_size_samples
            )
            if sfreq_ref is None:
                sfreq_ref, ch_names_ref = sfreq, ch_names
            else:
                assert sfreq == sfreq_ref and ch_names == ch_names_ref, f"Inconsistent meta for sub {sub_id}"
            if self.debug:
                X_ = X_[:5000]
                y_ = y_[:5000]
            X_all.append(X_)
            y_all.append(y_)

        random_state = np.random.RandomState(seed=self.seed)
        ids_train = random_state.choice(
            len(X_all), size=int(len(X_all) * self.train_ratio),
            replace=False
        )

        X_train = np.concatenate([X_all[i] for i in ids_train])
        y_train = np.concatenate([y_all[i] for i in ids_train])
        X_test = np.concatenate(
            [X_all[i] for i in range(len(X_all)) if i not in ids_train]
        )
        y_test = np.concatenate(
            [y_all[i] for i in range(len(y_all)) if i not in ids_train]
        )

        return X_train, y_train, X_test, y_test, sfreq_ref, ch_names_ref

    def get_data(self):

        X_train, y_train, X_test, y_test, sfreq, ch_names = self.prepare()

        return dict(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            task="classification",
            metrics=["accuracy", "balanced_accuracy", "f1_weighted"],
            n_classes=5,
            freq=sfreq,
            ch_names=ch_names
        )
