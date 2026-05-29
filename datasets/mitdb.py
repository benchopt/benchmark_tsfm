"""MIT-BIH Arrhythmia Database — 1-D event detection.

Each record is a 2-channel ECG sampled at 360 Hz.  Beat annotations are
converted to an object-detection style target using the 5-class AAMI grouping:

    class 0  N  — Normal / bundle-branch-block / paced
    class 1  S  — Supraventricular ectopic
    class 2  V  — Ventricular ectopic
    class 3  F  — Fusion
    class 4  Q  — Unknown / pacemaker artefact

Data contract output
--------------------
X_train : List[np.ndarray (T_i, 2)]      training portions  (C == 2)
y_train : List[np.ndarray (N_i, 2+K)]   one row per beat event:
                                            col 0    start  (normalised, 0–1)
                                            col 1    width  (normalised, 0–1)
                                            cols 2…  one-hot class vector (K)
X_test  : List[np.ndarray (T_j, 2)]      test portions
y_test  : List[np.ndarray (N_j, 2+K)]   same format
task    : "event_detection"
metrics : ["map_iou"]
extra   : n_classes (int)  K above
"""

import numpy as np
from benchopt import BaseDataset
from benchmark_utils.download import fetch_mitdb


# AAMI beat-type grouping (MIT-BIH annotation symbol → class index)
BEAT_CLASS = {
    # N group
    "N": 1, "L": 1, "R": 1, "e": 1, "j": 1,
    # S group
    "A": 2, "a": 2, "J": 2, "S": 2,
    # V group
    "V": 3, "E": 3,
    # F group
    "F": 4,
    # Q group
    "P": 5, "f": 5, "u": 5,
}

# All 48 standard MIT-BIH record IDs
MITDB_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107",
    "108", "109", "111", "112", "113", "114", "115", "116",
    "117", "118", "119", "121", "122", "123", "124",
    "200", "201", "202", "203", "205", "207", "208", "209",
    "210", "212", "213", "214", "215", "217", "219", "220",
    "221", "222", "223", "228", "230", "231", "232", "233", "234",
]


def _load_record(record_id, data_dir):
    """Load one WFDB record and return (signal, labels) as numpy arrays.

    Parameters
    ----------
    record_id : str  e.g. "100"
    data_dir  : str or Path  local directory holding .hea / .dat / .atr files

    Returns
    -------
    signal     : np.ndarray (T, 2)   float32
    ann_samples: np.ndarray (A,)     int32   R-peak sample indices
    ann_symbols: list of str         length A annotation symbols
    """
    import wfdb

    path = str(data_dir / record_id)
    record = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")

    signal = record.p_signal.astype(np.float32)
    return signal, ann.sample, ann.symbol


def _annotations_to_events(n_samples, ann_samples, ann_symbols, beat_window,
                           n_classes):
    """Convert beat annotations to an object-detection target array.

    Parameters
    ----------
    n_samples    : int                   total length of the series
    ann_samples  : np.ndarray (A,) int   sample indices of each annotation
    ann_symbols  : list of str           annotation symbols (len A)
    beat_window  : int                   half-width of each event in samples
    n_classes    : int                   K — number of AAMI classes

    Returns
    -------
    events : np.ndarray (N, 2+K)  float32
        Each row: [start_norm, width_norm, *one_hot_class]
        Only beats whose symbol appears in BEAT_CLASS are included.
    """
    rows = []
    for sample, symbol in zip(ann_samples, ann_symbols):
        aami_class = BEAT_CLASS.get(symbol)
        if aami_class is None:
            continue
        # Collapse to single class when n_classes == 1
        class_idx = 0 if n_classes == 1 else aami_class - 1
        if class_idx >= n_classes:
            continue

        start = max(0, sample - beat_window)
        end = min(n_samples, sample + beat_window)
        one_hot = np.zeros(n_classes, dtype=np.float32)
        one_hot[class_idx] = 1.0
        rows.append([start / n_samples, (end - start) / n_samples, *one_hot])

    if not rows:
        return np.zeros((0, 2 + n_classes), dtype=np.float32)
    return np.array(rows, dtype=np.float32)


class Dataset(BaseDataset):
    """MIT-BIH Arrhythmia Database for 1-D event detection.

    Parameters
    ----------
    record_ids : list of str or "all"
        Which records to include. Defaults to the full 48-record set.
    debug : bool
        If True, use only the first 2 records and truncate to 5 000 samples.
    train_ratio : float
        Fraction of each record used as training data.
    beat_window : int
        Half-width (in samples) of each event box around the R-peak.
        Default 36 ≈ ±100 ms at 360 Hz (covers the QRS complex).
    n_classes : int
        K — number of AAMI beat classes to distinguish (1–5).
        Classes are ordered N, S, V, F, Q; setting n_classes=1 collapses all
        annotated beats into a single "beat" class.
    """

    name = "MITDB"

    requirements = ["wfdb"]

    parameters = {
        "record_ids": ["all"],
        "debug": [False],
        "train_ratio": [0.7],
        "beat_window": [36],
        "n_classes": [5],
    }

    def get_data(self):
        data_dir = fetch_mitdb()

        record_ids = MITDB_RECORDS if self.record_ids == "all" else self.record_ids
        if self.debug:
            record_ids = record_ids[:2]

        X_train, y_train, X_test, y_test = [], [], [], []
        for rid in record_ids:
            signal, ann_samples, ann_symbols = _load_record(rid, data_dir)

            if self.debug:
                mask = ann_samples < 5000
                ann_samples = ann_samples[mask]
                ann_symbols = [s for s, m in zip(ann_symbols, mask) if m]
                signal = signal[:5000]

            split = max(1, int(len(signal) * self.train_ratio))

            for seg_signal, start, end, Xl, yl in [
                (signal[:split],  0,     split,       X_train, y_train),
                (signal[split:],  split, len(signal), X_test,  y_test),
            ]:
                seg_ann = ann_samples[(ann_samples >= start) & (ann_samples < end)] - start
                seg_sym = [s for s, idx in zip(ann_symbols, ann_samples)
                           if start <= idx < end]
                Xl.append(seg_signal)
                yl.append(_annotations_to_events(
                    len(seg_signal), seg_ann, seg_sym,
                    self.beat_window, self.n_classes,
                ))

        return dict(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            task="event_detection",
            metrics=["map_iou"],
            n_classes=self.n_classes,
        )
