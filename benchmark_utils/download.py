"""Shared download helper for the TSB-UAD public dataset bundle.
"""
from pathlib import Path
import pooch

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


def fetch_mitdb() -> Path:
    """Return the local directory holding MIT-BIH Arrhythmia Database files.

    Downloads the database via ``wfdb.dl_database`` on first call; subsequent
    calls are cache hits if the header files are already present.

    Returns
    -------
    Path  directory containing ``<record_id>.hea / .dat / .atr`` files
    """
    import wfdb
    _MITDB_DIR = Path(__file__).parent.parent / "data" / "mitdb"

    _MITDB_DIR.mkdir(parents=True, exist_ok=True)
    if not (_MITDB_DIR / "100.hea").exists():
        wfdb.dl_database("mitdb", dl_dir=str(_MITDB_DIR))
    return _MITDB_DIR


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
