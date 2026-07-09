"""Hugging Face Hub download helper shared by the HF-backed datasets."""

from pathlib import Path

from huggingface_hub import snapshot_download


def snapshot_hf_files(repo_id: str, subdir: str, pattern: str) -> "list[str]":
    """Download ``<subdir>/<pattern>`` files from a HF dataset repo and
    return their sorted local paths.

    Tries the local HF cache first (``local_files_only``) so cached runs
    skip the Hub round-trip and work offline; falls back to a network
    snapshot when the cache misses or lacks the requested files.
    """
    kwargs = dict(repo_type="dataset", allow_patterns=f"{subdir}/{pattern}")
    try:
        root = snapshot_download(repo_id, local_files_only=True, **kwargs)
        if not any((Path(root) / subdir).glob(pattern)):
            # Cached snapshot lacks these files — handled below.
            raise FileNotFoundError
    except FileNotFoundError:
        root = snapshot_download(repo_id, **kwargs)
    return sorted(str(p) for p in (Path(root) / subdir).glob(pattern))
