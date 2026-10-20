"""Shared paths for the GIFT-Eval validation workflow."""

from __future__ import annotations

import os
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent

RESULTS_DIR = REPO_ROOT / "results"
VALIDATION_DIR = RESULTS_DIR / "gifteval_validation"

ARTIFACT_DIR = VALIDATION_DIR / "artifacts"
FIGURE_DIR = VALIDATION_DIR / "figures"
HTML_DIR = VALIDATION_DIR / "html"
OUTPUT_DIR = VALIDATION_DIR / "outputs"

BENCH_FILE = RESULTS_DIR / "benchmark_tsfm_results_only.parquet"

# The official GIFT-Eval repository can be provided explicitly:
#
#   export GIFT_EVAL_REPO=/path/to/gift-eval
#
# Otherwise, use the current project layout where the reference
# repository is stored next to benchmark_tsfm under references/.
GIFT_REPO = Path(
    os.environ.get(
        "GIFT_EVAL_REPO",
        REPO_ROOT.parent / "references" / "gift-eval",
    )
).expanduser().resolve()
