#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

prepare_scientific_results

require_file "${CODE}/dataset_reports.py"
require_file "${CODE}/global_statistics.py"

run_step "04b_dataset_global_statistics" \
    "${PYTHON}" "${CODE}/global_statistics.py" \
    "${RESULTS}/csv/scientific_dataset_ranks.csv" \
    --output "${RESULTS}/csv/dataset_global_statistics.csv"

run_step "05_dataset_reports" \
    "${PYTHON}" "${CODE}/dataset_reports.py" \
    "${RESULTS}/csv/scientific_dataset_ranks.csv" \
    --global-summary "${RESULTS}/csv/dataset_global_statistics.csv" \
    --output-dir "${RESULTS}/reports_by_dataset"

echo
echo "======================================================================"
echo "DATASET REPORTS COMPLETED SUCCESSFULLY"
echo "======================================================================"
echo "Portal : ${RESULTS}/reports_by_dataset/index.html"
echo "Logs   : ${LOGS}"
