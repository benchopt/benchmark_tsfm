#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

PYTHON="${PYTHON:-python}"

CODE="${REPO_ROOT}/results_pipeline"
RESULTS="${REPO_ROOT}/results"
OUTPUTS="${REPO_ROOT}/outputs"
LOGS="${RESULTS}/logs"

STANDARD="${OUTPUTS}/benchmark_tsfm_standard.parquet"
GIFTEVAL="${OUTPUTS}/gifteval_all.parquet"
MERGED="${RESULTS}/benchmark_tsfm_complete.parquet"

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "ERROR: required file not found: $1" >&2
        exit 1
    fi
}

run_step() {
    local name="$1"
    shift

    echo
    echo "======================================================================"
    echo "[${name}]"
    echo "======================================================================"

    "$@" 2>&1 | tee "${LOGS}/${name}.log"
}

prepare_scientific_results() {
    mkdir -p "${RESULTS}" "${LOGS}"

    require_file "${STANDARD}"
    require_file "${GIFTEVAL}"

    require_file "${CODE}/merge_results.py"
    require_file "${CODE}/analysis.py"
    require_file "${CODE}/leaderboard.py"
    require_file "${CODE}/validate_leaderboard.py"
    require_file "${CODE}/scientific_leaderboard.py"

    cd "${REPO_ROOT}"

    run_step "00_merge_results" \
        "${PYTHON}" "${CODE}/merge_results.py" \
        "${STANDARD}" "${GIFTEVAL}" \
        --output "${MERGED}"

    run_step "01_analysis" \
        "${PYTHON}" "${CODE}/analysis.py" \
        "${MERGED}" \
        --output-dir "${RESULTS}"

    run_step "02_leaderboard" \
        "${PYTHON}" "${CODE}/leaderboard.py" \
        "${RESULTS}/benchmark_tsfm_results_only.parquet" \
        --output-dir "${RESULTS}"

    run_step "03_validation" \
        "${PYTHON}" "${CODE}/validate_leaderboard.py" \
        "${RESULTS}/benchmark_tsfm_results_only.parquet" \
        --output-dir "${RESULTS}"

    run_step "04_scientific_leaderboard" \
        "${PYTHON}" "${CODE}/scientific_leaderboard.py" \
        "${RESULTS}/csv/dataset_ranks_by_configuration.csv" \
        --output-dir "${RESULTS}"
}
