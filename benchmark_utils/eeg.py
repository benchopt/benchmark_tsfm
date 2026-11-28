"""Helpers shared by the EEG datasets and solvers.

EEG datasets return windows in µV, filtered but not standardized, with the
electrode names (``ch_names``) and the sampling frequency (``sfreq``). Each
solver normalizes the windows as its model was pretrained.
"""

from math import gcd

import numpy as np
from scipy.signal import resample_poly


def resample(x: np.ndarray, sfreq: float, target_sfreq: float) -> np.ndarray:
    """Resample windows along their last axis with a polyphase filter.

    Parameters
    ----------
    x : np.ndarray (..., T)
        Windows, time last.
    sfreq, target_sfreq : float
        Sampling frequencies of ``x`` and of the output, in Hz.

    Returns
    -------
    np.ndarray (..., T')
        ``x`` itself when the frequencies match, else the anti-aliased
        resampled windows.
    """
    if sfreq == target_sfreq:
        return x
    g = gcd(int(sfreq), int(target_sfreq))
    return resample_poly(x, up=int(target_sfreq) // g, down=int(sfreq) // g, axis=-1)


def split_subjects(
    X_per_subject: list, y_per_subject: list, train_ratio: float, seed: int
) -> dict:
    """Draw the training subjects at random and pool the windows.

    Parameters
    ----------
    X_per_subject : list of list of np.ndarray (T, C)
        The windows of each subject.
    y_per_subject : list of list of int
        Their labels.
    train_ratio : float
        Fraction of the subjects in the training set.
    seed : int

    Returns
    -------
    dict
        ``X_train``, ``y_train``, ``X_test`` and ``y_test``.
    """
    n_subjects = len(X_per_subject)
    rng = np.random.RandomState(seed)
    n_train = int(n_subjects * train_ratio)
    train = sorted(rng.choice(n_subjects, size=n_train, replace=False))
    test = [i for i in range(n_subjects) if i not in train]
    return dict(
        X_train=[x for i in train for x in X_per_subject[i]],
        y_train=np.array([y for i in train for y in y_per_subject[i]], dtype=np.int64),
        X_test=[x for i in test for x in X_per_subject[i]],
        y_test=np.array([y for i in test for y in y_per_subject[i]], dtype=np.int64),
    )
