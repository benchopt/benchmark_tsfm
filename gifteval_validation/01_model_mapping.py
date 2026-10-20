#!/usr/bin/env python3

from pathlib import Path
from _paths import ARTIFACT_DIR, BENCH_FILE, GIFT_REPO
import json
import pandas as pd


bench_path = BENCH_FILE
gift_repo = GIFT_REPO

reference_models = {
    "Chronos": "Chronos_small",
    "Chronos2": "chronos-2",
    "Toto-2.0": "Toto-2.0-2.5B",
    "Naive": "naive",
    "SeasonalNaive": "seasonal_naive",
}

df = pd.read_parquet(bench_path)

gift = df[
    df["dataset_name"]
    .astype(str)
    .str.startswith("GiftEval")
].copy()

rows = []

for solver_family, reference_name in reference_models.items():

    subset = gift[
        gift["solver_name"]
        .astype(str)
        .str.startswith(solver_family)
    ]

    if subset.empty:
        bench_solver = None
        checkpoint = None
    else:
        bench_solver = subset["solver_name"].iloc[0]

        checkpoint = None

        if "p_solver_checkpoint" in subset.columns:
            values = (
                subset["p_solver_checkpoint"]
                .dropna()
                .astype(str)
                .unique()
            )

            if len(values):
                checkpoint = values[0]

    config_path = (
        gift_repo
        / "results"
        / reference_name
        / "config.json"
    )

    config = {}

    if config_path.is_file():
        config = json.loads(
            config_path.read_text()
        )

    rows.append(
        {
            "benchopt_family": solver_family,
            "benchopt_solver": bench_solver,
            "benchopt_checkpoint": checkpoint,
            "gifteval_reference": reference_name,
            "gifteval_model": config.get("model"),
            "gifteval_model_link": config.get("model_link"),
            "testdata_leakage": config.get(
                "testdata_leakage"
            ),
            "reference_exists": config_path.is_file(),
        }
    )

# MOMENT : pas de référence trouvée
moment = gift[
    gift["solver_name"]
    .astype(str)
    .str.startswith("Moment")
]

checkpoint = None

if not moment.empty:
    values = (
        moment["p_solver_checkpoint"]
        .dropna()
        .astype(str)
        .unique()
    )

    if len(values):
        checkpoint = values[0]

rows.append(
    {
        "benchopt_family": "Moment",
        "benchopt_solver": (
            moment["solver_name"].iloc[0]
            if not moment.empty
            else None
        ),
        "benchopt_checkpoint": checkpoint,
        "gifteval_reference": None,
        "gifteval_model": None,
        "gifteval_model_link": None,
        "testdata_leakage": None,
        "reference_exists": False,
    }
)

out = pd.DataFrame(rows)

artifact = ARTIFACT_DIR / "model_mapping.csv"

artifact.parent.mkdir(
    parents=True,
    exist_ok=True
)

out.to_csv(
    artifact,
    index=False
)

print("=" * 90)
print("MODEL MAPPING — BENCHOPT vs GIFTEVAL")
print("=" * 90)

print()
print(
    out.to_string(
        index=False
    )
)

print()
print("Artifact:")
print(artifact.resolve())
