#!/usr/bin/env python3

from pathlib import Path
from _paths import ARTIFACT_DIR, GIFT_REPO
import pandas as pd


gift_repo = GIFT_REPO

models = [
    "Chronos_small",
    "chronos-2",
    "Toto-2.0-2.5B",
    "naive",
    "seasonal_naive",
]

rows = []

print("=" * 90)
print("GIFTEVAL REFERENCE SCHEMA")
print("=" * 90)

for model in models:

    path = (
        gift_repo
        / "results"
        / model
        / "all_results.csv"
    )

    print()
    print("-" * 90)
    print(model)
    print("-" * 90)

    if not path.is_file():
        print("ABSENT:", path)
        continue

    df = pd.read_csv(path)

    print(
        "shape =",
        df.shape
    )

    print("columns:")

    for c in df.columns:
        print("  -", c)

    rows.append(
        {
            "model": model,
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": "|".join(df.columns),
        }
    )

artifact = ARTIFACT_DIR / "reference_schema.csv"

pd.DataFrame(rows).to_csv(
    artifact,
    index=False
)

print()
print("=" * 90)
print("Artifact:")
print(artifact.resolve())
