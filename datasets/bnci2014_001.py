import numpy as np

from braindecode.preprocessing.windowers import create_windows_from_events

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    exponential_moving_standardize,
    preprocess,
    Preprocessor,
)

from benchopt import BaseDataset


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "BNCI2014_001"

    requirements = [
        'braindecode==1.5.1', 'moabb==1.5.0',
    ]

    parameters = {
        'train_ratio': [0.8],
        'debug': [False],
        'seed': [42],
    }

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        subjects = [1, 2] if self.debug else [1, 2, 3, 4, 5, 6, 7, 8, 9]
        dataset = MOABBDataset(
            dataset_name="BNCI2014_001", subject_ids=subjects,
        )
        low_cut_hz = 4.0  # low cut frequency for filtering
        high_cut_hz = 40.0  # high cut frequency for filtering
        # Parameters for exponential moving standardization
        factor_new = 1e-3
        init_block_size = 1000
        # Factor to convert from V to uV
        factor = 1e6

        preprocessors = [
            Preprocessor("pick_types", eeg=True, meg=False, stim=False),
            Preprocessor(lambda data: np.multiply(data, factor)),
            Preprocessor(
                "filter", l_freq=low_cut_hz, h_freq=high_cut_hz
            ),
            Preprocessor(
                exponential_moving_standardize,
                factor_new=factor_new,
                init_block_size=init_block_size,
            ),
        ]

        # Transform the data
        preprocess(dataset, preprocessors)

        trial_start_offset_seconds = -0.5
        # Extract sampling frequency, check that they are same in all datasets
        sfreq = dataset.datasets[0].raw.info["sfreq"]

        assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])
        # Calculate the trial start offset in samples.
        trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

        window_size_samples = None
        window_stride_samples = None

        windows_dataset = create_windows_from_events(
            dataset,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=0,
            preload=False,
            window_size_samples=window_size_samples,
            window_stride_samples=window_stride_samples,
        )

        splitted = windows_dataset.split("subject")
        subjects = list(splitted.keys())

        X_all = []
        y_all = []
        for sub in subjects:
            n_runs = len(splitted[sub].datasets)
            x = []
            y = []
            for run in range(n_runs):
                x += [sample[0].T for sample in splitted[sub].datasets[run]]
                y += [sample[1] for sample in splitted[sub].datasets[run]]
            X_all.append(np.array(x))
            y_all.append(np.array(y))
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
        np.unique(y_train), np.unique(y_test)  # sanity check
        return dict(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            task="classification",
            metrics=["accuracy", "balanced_accuracy", "f1_weighted"],
            n_classes=len(np.unique(y_train)),
        )
