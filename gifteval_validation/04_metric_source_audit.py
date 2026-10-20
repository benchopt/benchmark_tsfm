#!/usr/bin/env python3

from pathlib import Path
from _paths import ARTIFACT_DIR, REPO_ROOT
import csv
import re


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = REPO_ROOT
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = ARTIFACT_DIR / "metric_source_locations.csv"

SEARCH_ROOTS = [
    PROJECT_ROOT / "objective.py",
    PROJECT_ROOT / "benchmark_utils",
    PROJECT_ROOT / "datasets",
    PROJECT_ROOT / "solvers",
]

KEYWORDS = [
    "mae",
    "mse",
    "rmse",
    "mase",
    "smape",
    "mape",
    "crps",
    "wql",
    "weighted_sum_quantile",
    "mcis",
    "msis",
    "pinball",
    "skill_score",
    "quantile",
    "forecast",
    "median",
    "mean",
]

CONTEXT = 7


# ============================================================
# HELPERS
# ============================================================

def iter_python_files():
    for root in SEARCH_ROOTS:
        if root.is_file():
            yield root
        elif root.is_dir():
            yield from sorted(root.rglob("*.py"))


def matching_keywords(line: str):
    lower = line.lower()

    return [
        keyword
        for keyword in KEYWORDS
        if keyword in lower
    ]


# ============================================================
# SCAN
# ============================================================

records = []

print("=" * 100)
print("METRIC SOURCE AUDIT — BENCHOPT TSFM")
print("=" * 100)

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("Search roots:")
for root in SEARCH_ROOTS:
    print(" -", root)

print()
print("=" * 100)
print("MATCHES")
print("=" * 100)

for path in iter_python_files():

    try:
        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        print(f"[ERROR] Cannot read {path}: {exc}")
        continue

    lines = text.splitlines()

    matches = []

    for idx, line in enumerate(lines, start=1):
        keywords = matching_keywords(line)

        if keywords:
            matches.append(
                (
                    idx,
                    line,
                    keywords,
                )
            )

    if not matches:
        continue

    relative_path = path.relative_to(PROJECT_ROOT)

    print()
    print("#" * 100)
    print(relative_path)
    print("#" * 100)

    shown_ranges = set()

    for line_number, line, keywords in matches:

        start = max(
            1,
            line_number - CONTEXT,
        )

        end = min(
            len(lines),
            line_number + CONTEXT,
        )

        range_key = (
            start,
            end,
        )

        records.append(
            {
                "file": str(relative_path),
                "line_number": line_number,
                "keywords": "|".join(keywords),
                "line": line.strip(),
            }
        )

        # Avoid repeatedly printing the same overlapping
        # context block.
        if range_key in shown_ranges:
            continue

        shown_ranges.add(range_key)

        print()
        print(
            f"--- lines {start}-{end} "
            f"(match at {line_number}: "
            f"{', '.join(keywords)}) ---"
        )

        for current in range(
            start,
            end + 1,
        ):
            marker = ">>" if current == line_number else "  "

            print(
                f"{marker} {current:4d} | "
                f"{lines[current - 1]}"
            )


# ============================================================
# WRITE ARTIFACT
# ============================================================

with OUTPUT_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file",
            "line_number",
            "keywords",
            "line",
        ],
    )

    writer.writeheader()
    writer.writerows(records)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

print("Matching source lines :", len(records))
print("Artifact              :", OUTPUT_CSV)

print()
print("Files containing matches:")

files = sorted({
    record["file"]
    for record in records
})

for file in files:
    print(" -", file)

print()
print("=" * 100)
print("END METRIC SOURCE AUDIT")
print("=" * 100)
