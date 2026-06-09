"""Import FEV-bench upstream results and convert to benchopt parquet schema.

Downloads per-model CSV result files from
https://github.com/autogluon/fev/tree/main/benchmarks/fev_bench/results
and rewrites them as a single benchopt-compatible parquet whose schema
matches the rest of this benchmark's outputs (see ``objective.py``).

Schema mapping (upstream CSV column → benchopt column):

    model_name              → solver_name "Model[framework_version=...]"
    task_name               → dataset_name "FEV[task=...,horizon=...,...]"
    training_time_s
        + inference_time_s  → time   (total wall-clock per cell)
    SQL                     → objective_sql        (primary FEV metric)
    MASE                    → objective_mase
    WAPE                    → objective_wape
    WQL                     → objective_wql
    test_error              → objective_test_error
    training_time_s         → objective_training_time_s
    inference_time_s        → objective_inference_time_s
    horizon, num_windows,
    seasonality, ...        → p_dataset_* columns
    framework_version,
    fev_version             → p_solver_* columns
    platform / numpy /...   → environment placeholders (Linux x86_64, FEV upstream)

Usage::

    python scripts/import_fev_bench.py                       # default paths
    python scripts/import_fev_bench.py --out outputs/fev.parquet
    python scripts/import_fev_bench.py --cache-dir /tmp/fev  # reuse downloads
    python scripts/import_fev_bench.py --no-download         # skip refresh

The script is idempotent — re-running with the same ``--cache-dir`` only
re-fetches CSVs that are missing or older than 1 day on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REPO_RAW_BASE = (
    "https://raw.githubusercontent.com/autogluon/fev/main/benchmarks/fev_bench/results"
)
REPO_API_LISTING = (
    "https://api.github.com/repos/autogluon/fev/contents/benchmarks/fev_bench/results"
)
DEFAULT_CACHE_DIR = "/tmp/fev_bench_results"
DEFAULT_OUT = "outputs/fev_bench_results.parquet"

# Fallback list used when the GitHub API listing is unavailable (rate-limited
# without a token). Kept in sync with upstream as of 2026-05.
FALLBACK_MODELS = [
    "autoarima",
    "autoets",
    "autotheta",
    "catboost",
    "chronos-2",
    "chronos-bolt",
    "deepar",
    "drift",
    "flowstate",
    "lightgbm",
    "moirai-2_0",
    "naive",
    "patchtst",
    "seasonal_naive",
    "stat_ensemble",
    "sundial-base",
    "tabpfn-ts",
    "tft",
    "timesfm-2_5",
    "tirex",
    "toto-1_0",
]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def list_models() -> list[str]:
    """Discover available model CSVs via the GitHub contents API.

    Falls back to a hardcoded list if the API is unreachable or rate-limited.
    """
    req = urllib.request.Request(
        REPO_API_LISTING,
        headers={"Accept": "application/vnd.github+json"},
    )
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            items = json.load(resp)
        names = [
            it["name"].removesuffix(".csv")
            for it in items
            if it.get("type") == "file" and it.get("name", "").endswith(".csv")
        ]
        if names:
            return sorted(names)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        print(
            f"[warn] GitHub listing failed: {exc}; using fallback list", file=sys.stderr
        )
    return FALLBACK_MODELS


def _stale(path: Path, max_age_days: int = 1) -> bool:
    if not path.exists() or path.stat().st_size < 200:
        return True
    age_days = (time.time() - path.stat().st_mtime) / 86400
    return age_days > max_age_days


def fetch_one(model: str, cache_dir: Path) -> Path:
    out = cache_dir / f"{model}.csv"
    url = f"{REPO_RAW_BASE}/{urllib.parse.quote(model)}.csv"
    req = urllib.request.Request(url, headers={"User-Agent": "benchopt-fev-import"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        out.write_bytes(resp.read())
    return out


def download_all(
    cache_dir: Path, force: bool = False, max_workers: int = 8
) -> list[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    models = list_models()
    print(f"[info] {len(models)} models to import")

    targets: list[tuple[str, Path]] = []
    for m in models:
        path = cache_dir / f"{m}.csv"
        if force or _stale(path):
            targets.append((m, path))
        else:
            print(f"[skip] {m}.csv (cached)")

    if not targets:
        print("[info] all CSVs cached; nothing to fetch")
        return sorted(cache_dir.glob("*.csv"))

    print(f"[info] fetching {len(targets)} CSVs from {REPO_RAW_BASE}")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_one, m, cache_dir): m for m, _ in targets}
        for fut in as_completed(futures):
            m = futures[fut]
            try:
                fut.result()
                print(f"  ✓ {m}.csv")
            except Exception as exc:
                print(f"  ✗ {m}.csv: {exc}", file=sys.stderr)

    return sorted(cache_dir.glob("*.csv"))


# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------


def load_all(csv_paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in csv_paths:
        if p.stat().st_size < 200:
            print(f"[skip] {p.name}: empty / 404 placeholder", file=sys.stderr)
            continue
        frames.append(pd.read_csv(p))
    if not frames:
        raise RuntimeError("no FEV CSVs loaded")
    return pd.concat(frames, ignore_index=True)


def to_benchopt_parquet(fev: pd.DataFrame) -> pd.DataFrame:
    """Map upstream CSV columns onto the benchopt result schema."""
    n = len(fev)
    out = pd.DataFrame()

    out["base_seed"] = np.zeros(n, dtype=np.int64)
    out["objective_name"] = "Forecasting"
    out["obj_description"] = (
        "FEV-bench forecasting benchmark (autogluon/fev) — probabilistic + point "
        "forecasting across heterogeneous time-series datasets."
    )

    framework_version = fev["framework_version"].astype("string").fillna("na")
    out["solver_name"] = (
        fev["model_name"].astype(str) + "[framework_version=" + framework_version + "]"
    )
    out["solver_description"] = fev["model_name"].astype(str) + " (FEV-bench result)"

    out["dataset_name"] = (
        "FEV[task="
        + fev["task_name"].astype(str)
        + ",horizon="
        + fev["horizon"].astype(str)
        + ",n_windows="
        + fev["num_windows"].astype(str)
        + ",seasonality="
        + fev["seasonality"].astype(str)
        + ",eval_metric="
        + fev["eval_metric"].astype(str)
        + "]"
    )

    out["idx_rep"] = np.zeros(n, dtype=np.int64)
    out["sampling_strategy"] = "Run_once"
    out["file_objective"] = "objective.py"
    out["file_solver"] = "solvers/_fev_imported.py"
    out["file_dataset"] = "datasets/_fev_imported.py"

    out["p_solver_framework_version"] = framework_version
    out["p_solver_fev_version"] = fev["fev_version"].astype("string")

    out["p_dataset_task_name"] = fev["task_name"].astype("string")
    out["p_dataset_dataset_config"] = fev["dataset_config"].astype("string")
    out["p_dataset_horizon"] = fev["horizon"].astype(np.int64)
    out["p_dataset_num_windows"] = fev["num_windows"].astype(np.int64)
    out["p_dataset_seasonality"] = fev["seasonality"].astype(np.int64)
    out["p_dataset_eval_metric"] = fev["eval_metric"].astype("string")
    out["p_dataset_min_context_length"] = fev["min_context_length"].astype(np.int64)
    out["p_dataset_num_forecasts"] = fev["num_forecasts"].astype(np.int64)
    out["p_dataset_trained_on_this_dataset"] = fev["trained_on_this_dataset"].astype(
        bool
    )
    out["p_dataset_dataset_fingerprint"] = fev["dataset_fingerprint"].astype("string")

    out["stop_val"] = np.ones(n, dtype=np.int64)

    # `time` is the benchopt-canonical wall-clock for a cell. We sum
    # training + inference so the column reflects end-to-end runtime.
    out["time"] = (
        fev["training_time_s"].fillna(0).to_numpy()
        + fev["inference_time_s"].fillna(0).to_numpy()
    ).astype(np.float64)

    # Metric columns (every key prefixed `objective_` to match benchopt).
    out["objective_sql"] = fev["SQL"].astype(np.float64)
    out["objective_mase"] = fev["MASE"].astype(np.float64)
    out["objective_wape"] = fev["WAPE"].astype(np.float64)
    out["objective_wql"] = fev["WQL"].astype(np.float64)
    out["objective_training_time_s"] = fev["training_time_s"].astype(np.float64)
    out["objective_inference_time_s"] = fev["inference_time_s"].astype(np.float64)
    out["objective_test_error"] = fev["test_error"].astype(np.float64)

    # Environment placeholders — upstream FEV doesn't report per-cell sysinfo.
    out["env-OMP_NUM_THREADS"] = pd.Series([None] * n, dtype="object")
    out["platform"] = "Linux"
    out["platform-architecture"] = "x86_64"
    out["platform-release"] = ""
    out["platform-version"] = "FEV upstream results (autogluon/fev)"
    out["system-cpus"] = np.zeros(n, dtype=np.int64)
    out["system-processor"] = ""
    out["system-ram (GB)"] = np.zeros(n, dtype=np.int64)
    out["version-cuda"] = pd.Series([None] * n, dtype="object")
    out["version-numpy"] = ""
    out["version-numpy-libs"] = ""
    out["version-scipy"] = ""
    out["benchmark-git-tag"] = pd.Series([None] * n, dtype="object")
    out["run_date"] = datetime.now(timezone.utc).isoformat()

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(DEFAULT_OUT),
        help=f"Output parquet path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(DEFAULT_CACHE_DIR),
        help=f"CSV cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Reuse cached CSVs; do not refresh from GitHub",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download CSVs even if cached"
    )
    args = parser.parse_args()

    if args.no_download:
        csvs = sorted(args.cache_dir.glob("*.csv"))
        if not csvs:
            print(
                f"[error] --no-download given but no CSVs in {args.cache_dir}",
                file=sys.stderr,
            )
            return 2
    else:
        csvs = download_all(args.cache_dir, force=args.force)

    print(f"[info] loading {len(csvs)} CSVs")
    fev = load_all(csvs)
    print(
        f"[info] {len(fev)} rows from {fev['model_name'].nunique()} models, "
        f"{fev['task_name'].nunique()} tasks"
    )

    out_df = to_benchopt_parquet(fev)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    metrics = sorted(
        c
        for c in out_df.columns
        if c.startswith("objective_") and c != "objective_name"
    )
    print(f"\nWrote {args.out}")
    print(f"  rows    : {len(out_df)}")
    print(f"  solvers : {out_df['solver_name'].str.split('[').str[0].nunique()}")
    print(f"  datasets: {out_df['dataset_name'].nunique()}")
    print(f"  metrics : {metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
