#!/usr/bin/env python3

from pathlib import Path
from _paths import ARTIFACT_DIR, BENCH_FILE, GIFT_REPO
import pandas as pd


bench_path = BENCH_FILE

gift_ref = (
    GIFT_REPO
    / "results"
    / "Toto-2.0-2.5B"
    / "all_results.csv"
)

bench = pd.read_parquet(
    bench_path
)

bench = bench[
    bench["dataset_name"]
    .astype(str)
    .str.startswith("GiftEval")
].copy()

bench["gift_id"] = (
    bench["p_dataset_dataset_name"]
    .astype(str)
    + "/"
    + bench["p_dataset_term"]
    .astype(str)
)

bench_ids = set(
    bench["gift_id"]
    .dropna()
    .unique()
)

ref = pd.read_csv(
    gift_ref
)

ref_ids = set(
    ref["dataset"]
    .dropna()
    .astype(str)
    .unique()
)

common = sorted(
    bench_ids & ref_ids
)

missing_bench = sorted(
    ref_ids - bench_ids
)

extra_bench = sorted(
    bench_ids - ref_ids
)

summary = pd.DataFrame(
    [
        {
            "gifteval_total": len(ref_ids),
            "benchopt_total": len(bench_ids),
            "common": len(common),
            "missing_from_benchopt": len(
                missing_bench
            ),
            "extra_in_benchopt": len(
                extra_bench
            ),
        }
    ]
)

artifact_dir = ARTIFACT_DIR

artifact_dir.mkdir(
    parents=True,
    exist_ok=True
)

summary.to_csv(
    artifact_dir / "dataset_alignment.csv",
    index=False
)

(
    artifact_dir
    / "missing_datasets.txt"
).write_text(
    "\n".join(missing_bench)
    + "\n"
)

(
    artifact_dir
    / "extra_datasets.txt"
).write_text(
    "\n".join(extra_bench)
    + (
        "\n"
        if extra_bench
        else ""
    )
)

(
    artifact_dir
    / "common_datasets.txt"
).write_text(
    "\n".join(common)
    + "\n"
)

print("=" * 90)
print("DATASET ALIGNMENT — BENCHOPT vs GIFTEVAL")
print("=" * 90)

print()
print(
    "GiftEval officiel :",
    len(ref_ids)
)

print(
    "BenchOpt          :",
    len(bench_ids)
)

print(
    "Communs           :",
    len(common)
)

print(
    "Absents BenchOpt  :",
    len(missing_bench)
)

print(
    "Extras BenchOpt   :",
    len(extra_bench)
)

print()
print("=" * 90)
print("ABSENTS DE BENCHOPT")
print("=" * 90)

for dataset in missing_bench:
    print(" -", dataset)

print()
print("=" * 90)
print("EXTRAS DANS BENCHOPT")
print("=" * 90)

if extra_bench:
    for dataset in extra_bench:
        print(" -", dataset)
else:
    print("Aucun.")

print()
print("=" * 90)
print("ARTIFACTS")
print("=" * 90)

for p in sorted(
    artifact_dir.iterdir()
):
    print(p.resolve())
