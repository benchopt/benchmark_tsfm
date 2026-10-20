#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

PYTHON="${PYTHON:-python}"

CODE_DIR="${REPO_ROOT}/gifteval_validation"
VALIDATION_DIR="${REPO_ROOT}/results/gifteval_validation"
OUTPUT_DIR="${VALIDATION_DIR}/outputs"

mkdir -p \
    "${VALIDATION_DIR}/artifacts" \
    "${VALIDATION_DIR}/figures" \
    "${VALIDATION_DIR}/html" \
    "${OUTPUT_DIR}"

run_step() {
    local number="$1"
    local script="$2"
    local name="$3"

    local output="${OUTPUT_DIR}/${number}_${name}.out"

    echo
    echo "============================================================"
    echo "STEP ${number} — ${name}"
    echo "============================================================"

    (
        cd "${REPO_ROOT}"
        "${PYTHON}" "${CODE_DIR}/${script}"
    ) 2>&1 | tee "${output}"

    echo
    echo "[OK] Output saved to:"
    echo "${output}"
}

echo "============================================================"
echo "       GIFT-EVAL VALIDATION — BENCHOPT TSFM"
echo "============================================================"
echo "Date       : $(date)"
echo "Host       : $(hostname)"
echo "Repository : ${REPO_ROOT}"
echo

run_step "01" "01_model_mapping.py"         "model_mapping"
run_step "02" "02_reference_schema.py"      "reference_schema"
run_step "03" "03_dataset_alignment.py"     "dataset_alignment"
run_step "04" "04_metric_source_audit.py"   "metric_source_audit"
run_step "05" "05_metric_equivalence.py"    "metric_equivalence"
run_step "06" "06_comparability_report.py"  "comparability_report"
run_step "06b" "global_statistics.py"       "global_statistics"
run_step "07" "07_comparability_figures.py" "comparability_figures"
run_step "08" "08_comparability_html.py"    "comparability_html"

echo
echo "============================================================"
echo "              VALIDATION FINISHED"
echo "============================================================"

echo
echo "Artifacts:"
find "${VALIDATION_DIR}/artifacts" \
    -maxdepth 1 -type f -print | sort

echo
echo "Figures:"
find "${VALIDATION_DIR}/figures" \
    -maxdepth 1 -type f -print | sort

echo
echo "HTML:"
find "${VALIDATION_DIR}/html" \
    -maxdepth 1 -type f -print | sort

echo
echo "Logs:"
find "${OUTPUT_DIR}" \
    -maxdepth 1 -type f -print | sort
