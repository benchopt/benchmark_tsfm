"""Shared download helper for the TSB-UAD public dataset bundle.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from benchopt import config


_BUNDLE_URL = "https://www.thedatum.org/datasets/TSB-UAD-Public.zip"
_BUNDLE_SHA256 = (
    "ff4aa83a5a111835d410d962152e8dbebcda1039b778bae45b6b9c3f46dd49a1"
)
_BUNDLE_FILENAME = "TSB-UAD-Public.zip"
_BUNDLE_ROOT = "TSB-UAD-Public"

# Map benchmark dataset name -> subdirectory inside the TSB-UAD bundle.
_SUBDIR = {
    "DAPHNET": "Daphnet",
    "DODGERS": "Dodgers",
    "ECG": "ECG",
    "GENESIS": "Genesis",
    "GHL": "GHL",
    "IOPS": "IOPS",
    "KDD21": "KDD21",
    "MGAB": "MGAB",
    "MITDB": "MITDB",
    "NAB": "NAB",
    "OCCUPANCY": "Occupancy",
    "OPPORTUNITY": "OPPORTUNITY",
    "SENSORSCOPE": "SensorScope",
    "SMD": "SMD",
    "SVDB": "SVDB",
    "YAHOO": "YAHOO",
}


_BASE_NAMES = {
    "YAHOO": 'Yahoo_', 
    "ECG" : 'MBA_ECG',
    "SVDB": '8',
}

_FILES_EXT = {
    "YAHOO": '.out',
    "ECG": '.out',
    "SVDB": '.out'
}


def fetch_tsb_uad(name: str) -> Path:
    """Return the local directory holding TSB-UAD's ``.out`` files for *name*.

    The bundle is downloaded once into
    ``benchopt.config.get_data_path("TSB-UAD-Public")`` and extracted;
    subsequent calls are cache hits.
    """
    if name not in _SUBDIR:
        raise KeyError(
            f"{name!r} is not a TSB-UAD dataset name. "
            f"Known names: {sorted(_SUBDIR)}"
        )

    import pooch  # local import: only required when downloading

    try:
        import tqdm  # noqa: F401
        progressbar = True
    except ImportError:
        progressbar = False

    cache_root = Path(config.get_data_path(key=_BUNDLE_ROOT))
    cache_root.mkdir(parents=True, exist_ok=True)

    registry = pooch.create(
        path=cache_root,
        base_url="https://www.thedatum.org/datasets/",
        registry={_BUNDLE_FILENAME: f"sha256:{_BUNDLE_SHA256}"},
        urls={_BUNDLE_FILENAME: _BUNDLE_URL},
    )
    registry.fetch(
        _BUNDLE_FILENAME,
        processor=pooch.Unzip(extract_dir="."),
        progressbar=progressbar,
    )

    subdir = cache_root / _BUNDLE_ROOT / _SUBDIR[name]
    if not subdir.exists():
        raise FileNotFoundError(
            f"Expected {subdir} after extracting the TSB-UAD bundle."
        )
    return subdir


def load_data_tsb_uad(path, records_ids, train_ratio, number):
    """
    Load series from a dataset given the path, the record ids
    to get and a training ratio. 
    """
    # files names
    path = Path(path)
    base_name = _BASE_NAMES.get(path.name)
    extension = _FILES_EXT.get(path.name)

    # get ids of records
    if records_ids in (None, "all", ["all"]):
        records_ids = [
            f.stem for f in path.glob('*'+extension)
            if f.stem.startswith(base_name)
        ]

    if number in (None, -1):
        number = len(records_ids)

    X_train, X_test, y_test = [], [], []
    for i, id in enumerate(records_ids):

        if i >= number:
            break

        file_path = path / f"{id}{extension}"
        data = pd.read_csv(file_path, header=None).dropna().to_numpy()
        if data.shape[1] < 2:
            continue

        # compute split
        split = max(1, int(data.shape[0] * train_ratio))

        # split in train/test
        X_train.append(data[:split, 0].astype(np.float32))
        X_test.append(data[split:, 0].astype(np.float32))
        y_test.append(data[split:, 1].astype(np.int32))

    return X_train, X_test, y_test       
