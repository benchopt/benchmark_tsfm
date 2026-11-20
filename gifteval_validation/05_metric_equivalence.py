#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from _paths import ARTIFACT_DIR, BENCH_FILE, GIFT_REPO

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

MODEL_MAPPING_FILE = (
    ARTIFACT_DIR
    / "model_mapping.csv"
)

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# METRIC MAPPING
# ============================================================
#
# IMPORTANT:
#
# This is intentionally conservative.
#
# "candidate" means:
#     the BenchOpt implementation and GiftEval column appear to
#     represent the same conceptual quantity.
#
# It does NOT yet mean numerical reproduction has been proven.
#
# Metrics without a direct GiftEval equivalent are recorded but
# not compared numerically.
# ============================================================

METRIC_MAPPING = [
    {
        "benchopt_metric": "objective_mae",
        "gifteval_metric": "eval_metrics/MAE[0.5]",
        "mapping_status": "CANDIDATE",
        "reason": (
            "BenchOpt point forecast uses quantile 0.5 when available."
        ),
    },
    {
        "benchopt_metric": "objective_mse",
        "gifteval_metric": "eval_metrics/MSE[0.5]",
        "mapping_status": "CANDIDATE",
        "reason": (
            "BenchOpt MSE uses the point forecast, which is the "
            "0.5 quantile when available."
        ),
    },
    {
        "benchopt_metric": "objective_rmse",
        "gifteval_metric": None,
        "mapping_status": "NO_DIRECT_MATCH",
        "reason": (
            "BenchOpt RMSE is sqrt(MSE) of the median/0.5 point "
            "forecast. GiftEval exposes RMSE[mean], not RMSE[0.5]."
        ),
    },
    {
        "benchopt_metric": "objective_mase",
        "gifteval_metric": "eval_metrics/MASE[0.5]",
        "mapping_status": "CANDIDATE",
        "reason": (
            "Both are median/0.5 point MASE candidates. "
            "Seasonal scaling implementation still needs numerical "
            "validation."
        ),
    },
    {
        "benchopt_metric": "objective_smape",
        "gifteval_metric": "eval_metrics/sMAPE[0.5]",
        "mapping_status": "CANDIDATE",
        "reason": (
            "BenchOpt sMAPE uses the median/0.5 point forecast."
        ),
    },
    {
        "benchopt_metric": "objective_crps",
        "gifteval_metric": None,
        "mapping_status": "NO_DIRECT_MATCH",
        "reason": (
            "BenchOpt CRPS is implemented as 2 * mean pinball loss. "
            "No CRPS column exists in the inspected GiftEval CSV."
        ),
    },
    {
        "benchopt_metric": "objective_wql",
        "gifteval_metric": (
            "eval_metrics/mean_weighted_sum_quantile_loss"
        ),
        "mapping_status": "CANDIDATE",
        "reason": (
            "BenchOpt WQL implements mean weighted quantile loss "
            "over the model quantile grid."
        ),
    },
    {
        "benchopt_metric": "objective_mcis",
        "gifteval_metric": None,
        "mapping_status": "NO_DIRECT_MATCH",
        "reason": (
            "BenchOpt MCIS is an unscaled coverage interval score. "
            "GiftEval exposes MSIS, which should not be assumed "
            "equivalent."
        ),
    },
    {
        "benchopt_metric": "objective_pinball",
        "gifteval_metric": None,
        "mapping_status": "NO_DIRECT_MATCH",
        "reason": (
            "No standalone mean pinball-loss column exists in the "
            "inspected GiftEval result schema."
        ),
    },
    {
        "benchopt_metric": "objective_skill_score_ratio",
        "gifteval_metric": None,
        "mapping_status": "NO_DIRECT_MATCH",
        "reason": (
            "BenchOpt defines this as 1 - MASE. "
            "No corresponding GiftEval column exists."
        ),
    },
]


# ============================================================
# MODEL MAPPING
# ============================================================

# We intentionally postpone SeasonalNaive to P0.5 because the
# exact seasonality-selection protocol must be validated first.

MODEL_REFERENCES = {
    "Chronos": "Chronos_small",
    "Chronos2": "chronos-2",
    "Toto-2.0": "Toto-2.0-2.5B",
    "Naive": "naive",
}


def solver_family(name: str) -> str | None:
    """
    Map a BenchOpt solver configuration string to a reference
    solver family used in this validation.
    """
    text = str(name)

    # Chronos2 must be tested before Chronos.
    if text.startswith("Chronos2["):
        return "Chronos2"

    if text.startswith("Chronos["):
        return "Chronos"

    if text.startswith("Toto-2.0["):
        return "Toto-2.0"

    if text.startswith("Naive["):
        return "Naive"

    return None


# ============================================================
# NUMERICAL HELPERS
# ============================================================

def safe_relative_difference(
    bench: np.ndarray,
    ref: np.ndarray,
) -> np.ndarray:

    denominator = np.abs(ref)

    result = np.full(
        ref.shape,
        np.nan,
        dtype=float,
    )

    valid = denominator > 1e-12

    result[valid] = (
        np.abs(bench[valid] - ref[valid])
        / denominator[valid]
    )

    return result


def correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> float:

    if len(x) < 2:
        return float("nan")

    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")

    return float(
        np.corrcoef(x, y)[0, 1]
    )


# ============================================================
# LOAD BENCHOPT
# ============================================================

if not BENCH_FILE.is_file():
    raise FileNotFoundError(
        f"BenchOpt result file not found: {BENCH_FILE}"
    )

bench = pd.read_parquet(
    BENCH_FILE
)

bench = bench[
    bench["dataset_name"]
    .astype(str)
    .str.startswith("GiftEval")
].copy()

bench["solver_family_validation"] = (
    bench["solver_name"]
    .map(solver_family)
)

bench = bench[
    bench["solver_family_validation"]
    .notna()
].copy()

bench["gift_id"] = (
    bench["p_dataset_dataset_name"]
    .astype(str)
    + "/"
    + bench["p_dataset_term"]
    .astype(str)
)


# ============================================================
# SAVE DECLARED METRIC MAPPING
# ============================================================

mapping_df = pd.DataFrame(
    METRIC_MAPPING
)

mapping_path = (
    ARTIFACT_DIR
    / "metric_mapping.csv"
)

mapping_df.to_csv(
    mapping_path,
    index=False,
)


# ============================================================
# COMPARISON
# ============================================================

comparison_rows = []
summary_rows = []

print("=" * 100)
print("METRIC EQUIVALENCE — BENCHOPT vs GIFTEVAL")
print("=" * 100)

print()
print("BenchOpt file:")
print(BENCH_FILE)

print()
print("Reference repository:")
print(GIFT_REPO)

print()
print("=" * 100)
print("DECLARED METRIC MAPPING")
print("=" * 100)

print(
    mapping_df.to_string(
        index=False
    )
)


for family, reference_name in MODEL_REFERENCES.items():

    print()
    print("=" * 100)
    print(
        f"MODEL: {family} -> {reference_name}"
    )
    print("=" * 100)

    ref_path = (
        GIFT_REPO
        / "results"
        / reference_name
        / "all_results.csv"
    )

    if not ref_path.is_file():
        print(
            "[ERROR] Missing reference:",
            ref_path,
        )
        continue

    ref = pd.read_csv(
        ref_path
    )

    if "dataset" not in ref.columns:
        raise RuntimeError(
            f"'dataset' column missing from {ref_path}"
        )

    bench_model = bench[
        bench["solver_family_validation"]
        == family
    ].copy()

    print(
        "BenchOpt rows :",
        len(bench_model),
    )

    print(
        "BenchOpt datasets :",
        bench_model["gift_id"].nunique(),
    )

    print(
        "GiftEval datasets :",
        ref["dataset"].nunique(),
    )

    for metric_info in METRIC_MAPPING:

        bench_metric = metric_info[
            "benchopt_metric"
        ]

        gift_metric = metric_info[
            "gifteval_metric"
        ]

        declared_status = metric_info[
            "mapping_status"
        ]

        if gift_metric is None:
            continue

        print()
        print("-" * 100)
        print(
            f"{bench_metric} <-> {gift_metric}"
        )
        print("-" * 100)

        if bench_metric not in bench_model.columns:
            print(
                "[SKIP] BenchOpt column absent:",
                bench_metric,
            )
            continue

        if gift_metric not in ref.columns:
            print(
                "[SKIP] GiftEval column absent:",
                gift_metric,
            )
            continue

        left = (
            bench_model[
                [
                    "gift_id",
                    "solver_name",
                    bench_metric,
                ]
            ]
            .rename(
                columns={
                    bench_metric:
                    "benchopt_value",
                }
            )
        )

        right = (
            ref[
                [
                    "dataset",
                    gift_metric,
                ]
            ]
            .rename(
                columns={
                    "dataset": "gift_id",
                    gift_metric:
                    "gifteval_value",
                }
            )
        )

        merged = left.merge(
            right,
            on="gift_id",
            how="inner",
            validate="many_to_one",
        )

        merged["benchopt_value"] = (
            pd.to_numeric(
                merged["benchopt_value"],
                errors="coerce",
            )
        )

        merged["gifteval_value"] = (
            pd.to_numeric(
                merged["gifteval_value"],
                errors="coerce",
            )
        )

        finite = (
            np.isfinite(
                merged["benchopt_value"]
            )
            & np.isfinite(
                merged["gifteval_value"]
            )
        )

        usable = merged[
            finite
        ].copy()

        if usable.empty:
            print(
                "No finite common values."
            )
            continue

        bench_values = (
            usable["benchopt_value"]
            .to_numpy(dtype=float)
        )

        ref_values = (
            usable["gifteval_value"]
            .to_numpy(dtype=float)
        )

        abs_diff = np.abs(
            bench_values - ref_values
        )

        rel_diff = safe_relative_difference(
            bench_values,
            ref_values,
        )

        usable["absolute_difference"] = (
            abs_diff
        )

        usable[
            "relative_difference"
        ] = rel_diff

        usable[
            "relative_difference_percent"
        ] = rel_diff * 100.0

        usable["solver_family"] = family
        usable[
            "gifteval_reference"
        ] = reference_name

        usable[
            "benchopt_metric"
        ] = bench_metric

        usable[
            "gifteval_metric"
        ] = gift_metric

        usable[
            "declared_mapping_status"
        ] = declared_status

        comparison_rows.append(
            usable
        )

        corr = correlation(
            bench_values,
            ref_values,
        )

        # Purely descriptive thresholds.
        # PASS/WARN/FAIL will be defined later in P0.7.
        exact_1e8 = np.isclose(
            bench_values,
            ref_values,
            rtol=1e-8,
            atol=1e-10,
        ).mean()

        close_1e4 = np.isclose(
            bench_values,
            ref_values,
            rtol=1e-4,
            atol=1e-8,
        ).mean()

        summary_rows.append(
            {
                "solver_family": family,
                "gifteval_reference":
                    reference_name,
                "benchopt_metric":
                    bench_metric,
                "gifteval_metric":
                    gift_metric,
                "n_common":
                    len(merged),
                "n_finite":
                    len(usable),
                "mean_absolute_difference":
                    float(np.mean(abs_diff)),
                "median_absolute_difference":
                    float(np.median(abs_diff)),
                "max_absolute_difference":
                    float(np.max(abs_diff)),
                "mean_relative_difference_percent":
                    float(
                        np.nanmean(rel_diff)
                        * 100.0
                    ),
                "median_relative_difference_percent":
                    float(
                        np.nanmedian(rel_diff)
                        * 100.0
                    ),
                "max_relative_difference_percent":
                    float(
                        np.nanmax(rel_diff)
                        * 100.0
                    ),
                "correlation":
                    corr,
                "fraction_exact_1e8":
                    float(exact_1e8),
                "fraction_close_1e4":
                    float(close_1e4),
                "declared_mapping_status":
                    declared_status,
            }
        )

        print(
            "Common rows       :",
            len(merged),
        )

        print(
            "Finite rows       :",
            len(usable),
        )

        print(
            "Mean abs diff     :",
            f"{np.mean(abs_diff):.10g}",
        )

        print(
            "Median abs diff   :",
            f"{np.median(abs_diff):.10g}",
        )

        print(
            "Mean rel diff (%) :",
            f"{np.nanmean(rel_diff) * 100:.6f}",
        )

        print(
            "Median rel (%)    :",
            f"{np.nanmedian(rel_diff) * 100:.6f}",
        )

        print(
            "Max rel diff (%)  :",
            f"{np.nanmax(rel_diff) * 100:.6f}",
        )

        print(
            "Correlation       :",
            (
                f"{corr:.8f}"
                if np.isfinite(corr)
                else "nan"
            ),
        )

        print(
            "Exact @ 1e-8      :",
            f"{exact_1e8 * 100:.2f}%",
        )

        print(
            "Close @ 1e-4      :",
            f"{close_1e4 * 100:.2f}%",
        )


# ============================================================
# SAVE NUMERICAL RESULTS
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_path = (
    ARTIFACT_DIR
    / "metric_equivalence_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
)


if comparison_rows:

    detailed_df = pd.concat(
        comparison_rows,
        ignore_index=True,
    )

else:

    detailed_df = pd.DataFrame()


detail_path = (
    ARTIFACT_DIR
    / "metric_equivalence_detailed.csv"
)

detailed_df.to_csv(
    detail_path,
    index=False,
)


# ============================================================
# GLOBAL SUMMARY
# ============================================================

print()
print("=" * 100)
print("GLOBAL SUMMARY")
print("=" * 100)

if summary_df.empty:

    print(
        "No comparable metric pairs found."
    )

else:

    display_cols = [
        "solver_family",
        "benchopt_metric",
        "gifteval_metric",
        "n_finite",
        "mean_relative_difference_percent",
        "median_relative_difference_percent",
        "correlation",
    ]

    print(
        summary_df[
            display_cols
        ].to_string(
            index=False
        )
    )


print()
print("=" * 100)
print("ARTIFACTS")
print("=" * 100)

print(mapping_path)
print(summary_path)
print(detail_path)

print()
print("=" * 100)
print("IMPORTANT")
print("=" * 100)

print(
    "This step does NOT assign PASS/WARN/FAIL."
)

print(
    "It measures candidate metric equivalence and numerical "
    "agreement only."
)

print(
    "PASS/WARN/FAIL thresholds will be defined in a later "
    "validation step."
)

print()
print("=" * 100)
print("END METRIC EQUIVALENCE")
print("=" * 100)
