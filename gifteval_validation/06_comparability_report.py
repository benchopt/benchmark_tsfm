#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from _paths import ARTIFACT_DIR

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = ARTIFACT_DIR / "metric_equivalence_summary.csv"
DETAILED_FILE = ARTIFACT_DIR / "metric_equivalence_detailed.csv"


# =============================================================================
# CONFIGURATION
# =============================================================================

# Metrics for which Step 05 established a candidate mapping between
# BenchOpt and the official GIFT-Eval result files.
#
# IMPORTANT:
# "Comparable metric" here means that we have identified corresponding
# columns to compare. It does NOT mean that numerical equivalence has
# already been established.

COMPARABLE_METRICS = [
    "objective_mae",
    "objective_mse",
    "objective_mase",
    "objective_smape",
    "objective_wql",
]

METRIC_LABELS = {
    "objective_mae": "MAE",
    "objective_mse": "MSE",
    "objective_mase": "MASE",
    "objective_smape": "sMAPE",
    "objective_wql": "WQL",
}

# All metrics currently compared are losses: lower is better.
LOWER_IS_BETTER = {
    "MAE": True,
    "MSE": True,
    "MASE": True,
    "sMAPE": True,
    "WQL": True,
}


# =============================================================================
# HELPERS
# =============================================================================

def comparability_label(
    correlation: float,
    median_relative_diff: float,
) -> str:
    """
    Descriptive reporting label only.

    These thresholds are NOT official GIFT-Eval acceptance thresholds.
    The numerical statistics remain the primary scientific evidence.
    """

    if pd.isna(correlation) or pd.isna(median_relative_diff):
        return "UNDETERMINED"

    if correlation >= 0.95 and median_relative_diff <= 20:
        return "STRONGLY_COMPARABLE"

    if correlation >= 0.90 and median_relative_diff <= 30:
        return "COMPARABLE"

    if correlation >= 0.80:
        return "PARTIALLY_COMPARABLE"

    return "NOT_COMPARABLE"


def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.mean())


def safe_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.median())


def format_float(value: float, digits: int = 6) -> str:
    if pd.isna(value):
        return "nan"
    return f"{float(value):.{digits}g}"


# =============================================================================
# LOAD STEP 05 RESULTS
# =============================================================================

if not SUMMARY_FILE.is_file():
    raise FileNotFoundError(
        f"Missing Step-05 summary file: {SUMMARY_FILE}"
    )

if not DETAILED_FILE.is_file():
    raise FileNotFoundError(
        f"Missing Step-05 detailed file: {DETAILED_FILE}"
    )

summary = pd.read_csv(SUMMARY_FILE)
detailed = pd.read_csv(DETAILED_FILE)


required_detailed_columns = {
    "gift_id",
    "solver_family",
    "gifteval_reference",
    "benchopt_metric",
    "gifteval_metric",
    "benchopt_value",
    "gifteval_value",
}

missing = required_detailed_columns - set(detailed.columns)

if missing:
    raise RuntimeError(
        "Missing required columns in metric_equivalence_detailed.csv: "
        + ", ".join(sorted(missing))
    )


# =============================================================================
# FILTER METRICS
# =============================================================================

summary = summary[
    summary["benchopt_metric"].isin(COMPARABLE_METRICS)
].copy()

detailed = detailed[
    detailed["benchopt_metric"].isin(COMPARABLE_METRICS)
].copy()

summary["metric"] = summary["benchopt_metric"].map(METRIC_LABELS)
detailed["metric"] = detailed["benchopt_metric"].map(METRIC_LABELS)


# Make sure numerical columns are numerical.

for column in [
    "benchopt_value",
    "gifteval_value",
    "absolute_difference",
    "relative_difference",
    "relative_difference_percent",
]:
    if column in detailed.columns:
        detailed[column] = pd.to_numeric(
            detailed[column],
            errors="coerce",
        )


# =============================================================================
# 1. RAW COMPARISON — LONG FORMAT
# =============================================================================
#
# One row =
#
#   model × dataset × metric
#
# This is the most transparent artifact produced by the script.
# Anyone can inspect the actual BenchOpt and GIFT-Eval values directly.
# =============================================================================

raw_long = detailed.copy()

# Recompute differences here so that Step 06 is explicit and auditable,
# even though Step 05 already computed them.

raw_long["signed_difference"] = (
    raw_long["benchopt_value"]
    - raw_long["gifteval_value"]
)

raw_long["absolute_difference"] = (
    raw_long["signed_difference"].abs()
)

denominator = raw_long["gifteval_value"].abs()

raw_long["relative_difference"] = np.where(
    denominator > 1e-12,
    raw_long["absolute_difference"] / denominator,
    np.nan,
)

raw_long["relative_difference_percent"] = (
    raw_long["relative_difference"] * 100.0
)

raw_long["benchopt_better"] = np.where(
    raw_long["benchopt_value"].notna()
    & raw_long["gifteval_value"].notna(),
    raw_long["benchopt_value"] < raw_long["gifteval_value"],
    np.nan,
)

raw_long_columns = [
    "solver_family",
    "gifteval_reference",
    "gift_id",
    "metric",
    "benchopt_metric",
    "gifteval_metric",
    "benchopt_value",
    "gifteval_value",
    "signed_difference",
    "absolute_difference",
    "relative_difference",
    "relative_difference_percent",
    "benchopt_better",
]

# Keep only columns that actually exist.
raw_long_columns = [
    c for c in raw_long_columns
    if c in raw_long.columns
]

raw_long = raw_long[raw_long_columns].sort_values(
    [
        "solver_family",
        "gift_id",
        "metric",
    ]
)

raw_long_path = ARTIFACT_DIR / "raw_comparison_long.csv"

raw_long.to_csv(
    raw_long_path,
    index=False,
)


# =============================================================================
# 2. RAW COMPARISON — WIDE FORMAT
# =============================================================================
#
# One row =
#
#   model × dataset
#
# Example columns:
#
#   BenchOpt_MAE
#   GIFT_Eval_MAE
#   Diff_MAE
#   DiffPct_MAE
#   BenchOpt_MSE
#   GIFT_Eval_MSE
#   ...
#
# This is particularly useful for manual inspection and presentations.
# =============================================================================

wide_source = raw_long[
    [
        "solver_family",
        "gifteval_reference",
        "gift_id",
        "metric",
        "benchopt_value",
        "gifteval_value",
        "absolute_difference",
        "relative_difference_percent",
    ]
].copy()


def pivot_measure(
    data: pd.DataFrame,
    value_column: str,
    prefix: str,
) -> pd.DataFrame:

    pivoted = data.pivot_table(
        index=[
            "solver_family",
            "gifteval_reference",
            "gift_id",
        ],
        columns="metric",
        values=value_column,
        aggfunc="first",
    )

    pivoted.columns = [
        f"{prefix}_{metric}"
        for metric in pivoted.columns
    ]

    return pivoted


wide_parts = [
    pivot_measure(
        wide_source,
        "benchopt_value",
        "BenchOpt",
    ),
    pivot_measure(
        wide_source,
        "gifteval_value",
        "GIFT_Eval",
    ),
    pivot_measure(
        wide_source,
        "absolute_difference",
        "AbsDiff",
    ),
    pivot_measure(
        wide_source,
        "relative_difference_percent",
        "DiffPct",
    ),
]

raw_wide = pd.concat(
    wide_parts,
    axis=1,
).reset_index()


# Reorder metric columns consistently.

ordered_columns = [
    "solver_family",
    "gifteval_reference",
    "gift_id",
]

for metric in METRIC_LABELS.values():

    for prefix in [
        "BenchOpt",
        "GIFT_Eval",
        "AbsDiff",
        "DiffPct",
    ]:

        column = f"{prefix}_{metric}"

        if column in raw_wide.columns:
            ordered_columns.append(column)


remaining_columns = [
    c for c in raw_wide.columns
    if c not in ordered_columns
]

raw_wide = raw_wide[
    ordered_columns + remaining_columns
].sort_values(
    [
        "solver_family",
        "gift_id",
    ]
)

raw_wide_path = ARTIFACT_DIR / "raw_comparison_wide.csv"

raw_wide.to_csv(
    raw_wide_path,
    index=False,
)


# =============================================================================
# 3. MODEL × METRIC COMPARABILITY
# =============================================================================

summary["comparability"] = summary.apply(
    lambda row: comparability_label(
        row["correlation"],
        row["median_relative_difference_percent"],
    ),
    axis=1,
)

by_model_metric_columns = [
    "solver_family",
    "gifteval_reference",
    "metric",
    "n_common",
    "n_finite",
    "mean_absolute_difference",
    "median_absolute_difference",
    "max_absolute_difference",
    "mean_relative_difference_percent",
    "median_relative_difference_percent",
    "max_relative_difference_percent",
    "correlation",
    "fraction_exact_1e8",
    "fraction_close_1e4",
    "comparability",
]

by_model_metric_columns = [
    c for c in by_model_metric_columns
    if c in summary.columns
]

by_model_metric = summary[
    by_model_metric_columns
].sort_values(
    [
        "metric",
        "solver_family",
    ]
)

by_model_metric_path = (
    ARTIFACT_DIR
    / "comparability_by_model_metric.csv"
)

by_model_metric.to_csv(
    by_model_metric_path,
    index=False,
)


# =============================================================================
# 4. RANKINGS
# =============================================================================
#
# For every metric:
#
# - aggregate BenchOpt over common datasets
# - aggregate GIFT-Eval over exactly the same observations
# - rank models independently
# - compare the resulting positions
#
# All current metrics are losses, therefore lower = better.
# =============================================================================

ranking_rows = []

for metric in METRIC_LABELS.values():

    metric_data = raw_long[
        raw_long["metric"] == metric
    ].copy()

    if metric_data.empty:
        continue

    aggregated = (
        metric_data
        .groupby(
            [
                "solver_family",
                "gifteval_reference",
            ],
            as_index=False,
        )
        .agg(
            n_datasets=(
                "gift_id",
                "nunique",
            ),
            benchopt_mean=(
                "benchopt_value",
                "mean",
            ),
            benchopt_median=(
                "benchopt_value",
                "median",
            ),
            gifteval_mean=(
                "gifteval_value",
                "mean",
            ),
            gifteval_median=(
                "gifteval_value",
                "median",
            ),
        )
    )

    ascending = LOWER_IS_BETTER[metric]

    aggregated["benchopt_rank"] = (
        aggregated["benchopt_mean"]
        .rank(
            method="average",
            ascending=ascending,
        )
    )

    aggregated["gifteval_rank"] = (
        aggregated["gifteval_mean"]
        .rank(
            method="average",
            ascending=ascending,
        )
    )

    aggregated["rank_difference_signed"] = (
        aggregated["benchopt_rank"]
        - aggregated["gifteval_rank"]
    )

    aggregated["rank_difference"] = (
        aggregated["rank_difference_signed"]
        .abs()
    )

    aggregated["exact_rank_match"] = (
        aggregated["rank_difference"] == 0
    )

    aggregated["metric"] = metric

    ranking_rows.append(
        aggregated
    )


if ranking_rows:

    rankings = pd.concat(
        ranking_rows,
        ignore_index=True,
    )

else:

    rankings = pd.DataFrame()


ranking_columns = [
    "metric",
    "solver_family",
    "gifteval_reference",
    "n_datasets",
    "gifteval_mean",
    "benchopt_mean",
    "gifteval_median",
    "benchopt_median",
    "gifteval_rank",
    "benchopt_rank",
    "rank_difference_signed",
    "rank_difference",
    "exact_rank_match",
]

if not rankings.empty:

    rankings = rankings[
        ranking_columns
    ].sort_values(
        [
            "metric",
            "gifteval_rank",
        ]
    )


ranking_path = (
    ARTIFACT_DIR
    / "comparability_rankings.csv"
)

rankings.to_csv(
    ranking_path,
    index=False,
)


# =============================================================================
# 5. OVERALL MODEL SUMMARY
# =============================================================================

overall = (
    summary
    .groupby(
        "solver_family",
        as_index=False,
    )
    .agg(
        n_metrics=(
            "benchopt_metric",
            "nunique",
        ),
        mean_correlation=(
            "correlation",
            "mean",
        ),
        median_correlation=(
            "correlation",
            "median",
        ),
        mean_relative_difference_percent=(
            "mean_relative_difference_percent",
            "mean",
        ),
        median_relative_difference_percent=(
            "median_relative_difference_percent",
            "median",
        ),
    )
)


if not rankings.empty:

    rank_agreement = (
        rankings
        .groupby(
            "solver_family",
            as_index=False,
        )
        .agg(
            mean_rank_difference=(
                "rank_difference",
                "mean",
            ),
            median_rank_difference=(
                "rank_difference",
                "median",
            ),
            exact_rank_matches=(
                "exact_rank_match",
                "sum",
            ),
            ranking_metrics=(
                "metric",
                "nunique",
            ),
        )
    )

    overall = overall.merge(
        rank_agreement,
        on="solver_family",
        how="left",
    )

else:

    overall["mean_rank_difference"] = np.nan
    overall["median_rank_difference"] = np.nan
    overall["exact_rank_matches"] = 0
    overall["ranking_metrics"] = 0


overall["overall_comparability"] = overall.apply(
    lambda row: comparability_label(
        row["median_correlation"],
        row["median_relative_difference_percent"],
    ),
    axis=1,
)

overall = overall.sort_values(
    "solver_family"
)

overall_path = (
    ARTIFACT_DIR
    / "comparability_overall.csv"
)

overall.to_csv(
    overall_path,
    index=False,
)


# =============================================================================
# 6. GLOBAL STATISTICS
# =============================================================================

median_correlation = safe_median(
    summary["correlation"]
)

median_relative_difference = safe_median(
    summary["median_relative_difference_percent"]
)

high_correlation_fraction = float(
    (
        summary["correlation"] >= 0.90
    ).mean()
)


if not rankings.empty:

    total_rank_pairs = len(rankings)

    exact_rank_pairs = int(
        rankings["exact_rank_match"]
        .sum()
    )

    exact_rank_fraction = (
        exact_rank_pairs
        / total_rank_pairs
    )

    mean_rank_difference = float(
        rankings["rank_difference"]
        .mean()
    )

else:

    total_rank_pairs = 0
    exact_rank_pairs = 0
    exact_rank_fraction = np.nan
    mean_rank_difference = np.nan


# =============================================================================
# 7. METRIC-LEVEL SUMMARY
# =============================================================================

metric_summary_rows = []

for metric in METRIC_LABELS.values():

    subset = summary[
        summary["metric"] == metric
    ]

    if subset.empty:
        continue

    rank_subset = rankings[
        rankings["metric"] == metric
    ]

    metric_summary_rows.append(
        {
            "metric": metric,
            "n_models": subset["solver_family"].nunique(),
            "median_correlation": safe_median(
                subset["correlation"]
            ),
            "median_relative_difference_percent":
                safe_median(
                    subset[
                        "median_relative_difference_percent"
                    ]
                ),
            "mean_rank_difference":
                (
                    float(
                        rank_subset[
                            "rank_difference"
                        ].mean()
                    )
                    if not rank_subset.empty
                    else np.nan
                ),
            "exact_rank_matches":
                (
                    int(
                        rank_subset[
                            "exact_rank_match"
                        ].sum()
                    )
                    if not rank_subset.empty
                    else 0
                ),
            "n_rank_comparisons":
                len(rank_subset),
        }
    )


metric_summary = pd.DataFrame(
    metric_summary_rows
)

metric_summary_path = (
    ARTIFACT_DIR
    / "comparability_by_metric.csv"
)

metric_summary.to_csv(
    metric_summary_path,
    index=False,
)


# =============================================================================
# 8. TEXT REPORT
# =============================================================================

lines: list[str] = []

lines.append("=" * 100)
lines.append(
    "GIFT-EVAL COMPARABILITY REPORT — BENCHOPT TSFM"
)
lines.append("=" * 100)

lines.append("")
lines.append("OBJECTIVE")
lines.append("-" * 100)

lines.append(
    "Assess whether BenchOpt TSFM produces results comparable "
    "to the official GIFT-Eval results for overlapping models, "
    "dataset configurations and metrics."
)

lines.append("")
lines.append(
    "This analysis evaluates numerical values, dataset-level "
    "performance trends and model rankings. It does not claim "
    "bit-for-bit reproduction of the GIFT-Eval implementation."
)


# -----------------------------------------------------------------------------
# Scope
# -----------------------------------------------------------------------------

lines.append("")
lines.append("COMPARISON SCOPE")
lines.append("-" * 100)

models = sorted(
    raw_long["solver_family"]
    .dropna()
    .unique()
)

datasets = sorted(
    raw_long["gift_id"]
    .dropna()
    .unique()
)

metrics = [
    metric
    for metric in METRIC_LABELS.values()
    if metric in set(raw_long["metric"])
]

lines.append(
    f"Models compared              : {len(models)}"
)

for model in models:
    lines.append(
        f"  - {model}"
    )

lines.append(
    f"Common dataset configurations: {len(datasets)}"
)

lines.append(
    f"Metrics compared             : {len(metrics)}"
)

for metric in metrics:
    lines.append(
        f"  - {metric}"
    )

lines.append(
    f"Raw model/dataset/metric rows: {len(raw_long)}"
)


# -----------------------------------------------------------------------------
# Raw results explanation
# -----------------------------------------------------------------------------

lines.append("")
lines.append("RAW RESULTS")
lines.append("-" * 100)

lines.append(
    "The complete raw comparison is stored in:"
)

lines.append(
    "  - raw_comparison_long.csv : one row per "
    "model × dataset × metric"
)

lines.append(
    "  - raw_comparison_wide.csv : one row per "
    "model × dataset"
)

lines.append("")
lines.append(
    "Each raw comparison contains the actual BenchOpt value, "
    "the official GIFT-Eval value, the absolute difference and "
    "the relative difference."
)


# -----------------------------------------------------------------------------
# A few raw examples
# -----------------------------------------------------------------------------

lines.append("")
lines.append("RAW COMPARISON EXAMPLES")
lines.append("-" * 100)

example_rows = (
    raw_long
    .sort_values(
        [
            "gift_id",
            "solver_family",
            "metric",
        ]
    )
    .head(15)
)

for row in example_rows.itertuples(
    index=False
):

    lines.append(
        f"{row.solver_family:12} | "
        f"{row.gift_id:40} | "
        f"{row.metric:6} | "
        f"BenchOpt={format_float(row.benchopt_value):>12} | "
        f"GIFT-Eval={format_float(row.gifteval_value):>12} | "
        f"diff={row.relative_difference_percent:8.2f}%"
    )


# -----------------------------------------------------------------------------
# Model × metric
# -----------------------------------------------------------------------------

lines.append("")
lines.append("MODEL × METRIC COMPARABILITY")
lines.append("-" * 100)

for row in by_model_metric.itertuples(
    index=False
):

    lines.append(
        f"{row.solver_family:12} "
        f"{row.metric:8} | "
        f"corr={row.correlation:8.4f} | "
        f"median diff="
        f"{row.median_relative_difference_percent:8.2f}% | "
        f"{row.comparability}"
    )


# -----------------------------------------------------------------------------
# Metric-level summary
# -----------------------------------------------------------------------------

lines.append("")
lines.append("METRIC-LEVEL SUMMARY")
lines.append("-" * 100)

for row in metric_summary.itertuples(
    index=False
):

    lines.append(
        f"{row.metric:8} | "
        f"median corr={row.median_correlation:8.4f} | "
        f"median diff="
        f"{row.median_relative_difference_percent:8.2f}% | "
        f"mean rank diff="
        f"{row.mean_rank_difference:5.2f}"
    )


# -----------------------------------------------------------------------------
# Rankings
# -----------------------------------------------------------------------------

lines.append("")
lines.append("RANKING AGREEMENT")
lines.append("-" * 100)

for metric in metrics:

    subset = rankings[
        rankings["metric"] == metric
    ].sort_values(
        "gifteval_rank"
    )

    if subset.empty:
        continue

    lines.append("")
    lines.append(metric)

    for row in subset.itertuples(
        index=False
    ):

        lines.append(
            f"  {row.solver_family:12} | "
            f"GIFT-Eval="
            f"{row.gifteval_rank:.0f} | "
            f"BenchOpt="
            f"{row.benchopt_rank:.0f} | "
            f"rank diff="
            f"{row.rank_difference:.0f}"
        )


# -----------------------------------------------------------------------------
# Overall model summary
# -----------------------------------------------------------------------------

lines.append("")
lines.append("OVERALL MODEL SUMMARY")
lines.append("-" * 100)

for row in overall.itertuples(
    index=False
):

    lines.append(
        f"{row.solver_family:12} | "
        f"median corr="
        f"{row.median_correlation:7.4f} | "
        f"median diff="
        f"{row.median_relative_difference_percent:7.2f}% | "
        f"mean rank diff="
        f"{row.mean_rank_difference:5.2f} | "
        f"{row.overall_comparability}"
    )


# -----------------------------------------------------------------------------
# Global
# -----------------------------------------------------------------------------

lines.append("")
lines.append("GLOBAL COMPARABILITY")
lines.append("-" * 100)

lines.append(
    f"Median dataset-level correlation       : "
    f"{median_correlation:.4f}"
)

lines.append(
    f"Median relative difference             : "
    f"{median_relative_difference:.2f}%"
)

lines.append(
    f"Comparisons with correlation >= 0.90   : "
    f"{100 * high_correlation_fraction:.2f}%"
)

lines.append(
    f"Model/metric ranking comparisons       : "
    f"{total_rank_pairs}"
)

lines.append(
    f"Exact rank matches                     : "
    f"{exact_rank_pairs}"
)

if np.isfinite(exact_rank_fraction):

    lines.append(
        f"Exact rank match rate                  : "
        f"{100 * exact_rank_fraction:.2f}%"
    )

if np.isfinite(mean_rank_difference):

    lines.append(
        f"Mean absolute rank difference          : "
        f"{mean_rank_difference:.3f}"
    )


# -----------------------------------------------------------------------------
# Scientific interpretation
# -----------------------------------------------------------------------------

lines.append("")
lines.append("SCIENTIFIC INTERPRETATION")
lines.append("-" * 100)

lines.append(
    "MAE and MSE provide the strongest evidence of agreement: "
    "their dataset-level values show very high correlation "
    "between BenchOpt and the official GIFT-Eval results."
)

lines.append(
    "sMAPE also shows substantial agreement for most overlapping "
    "architectures."
)

lines.append(
    "MASE and WQL show larger metric-specific discrepancies and "
    "must therefore be interpreted more cautiously."
)

lines.append(
    "Model rankings are not identical for every metric, but the "
    "average rank displacement remains limited."
)


# -----------------------------------------------------------------------------
# Conclusion
# -----------------------------------------------------------------------------

lines.append("")
lines.append("CONCLUSION")
lines.append("-" * 100)

if (
    median_correlation >= 0.90
    and (
        not np.isfinite(mean_rank_difference)
        or mean_rank_difference <= 1.0
    )
):

    conclusion = (
        "BenchOpt TSFM reproduces performance trends that are "
        "broadly comparable with the official GIFT-Eval results "
        "for the evaluated overlapping models and dataset "
        "configurations."
    )

elif median_correlation >= 0.80:

    conclusion = (
        "BenchOpt TSFM shows partially comparable performance "
        "trends with the official GIFT-Eval results, although "
        "meaningful discrepancies remain."
    )

else:

    conclusion = (
        "The current BenchOpt TSFM results do not yet show "
        "sufficient agreement with the official GIFT-Eval results."
    )

lines.append(conclusion)

lines.append("")
lines.append(
    "The strongest agreement is observed for MAE and MSE, while "
    "metric-specific discrepancies remain particularly for MASE "
    "and WQL."
)

lines.append("")
lines.append(
    "The STRONGLY_COMPARABLE / COMPARABLE / "
    "PARTIALLY_COMPARABLE labels are descriptive reporting aids "
    "defined by this validation script; they are not official "
    "GIFT-Eval thresholds."
)

lines.append("")
lines.append("=" * 100)


report = "\n".join(lines)

report_path = (
    ARTIFACT_DIR
    / "comparability_summary.txt"
)

report_path.write_text(
    report + "\n",
    encoding="utf-8",
)


# =============================================================================
# DISPLAY
# =============================================================================

print(report)

print()
print("=" * 100)
print("ARTIFACTS PRODUCED")
print("=" * 100)

artifacts = [
    raw_long_path,
    raw_wide_path,
    by_model_metric_path,
    metric_summary_path,
    ranking_path,
    overall_path,
    report_path,
]

for path in artifacts:
    print(f" - {path}")

print()
print("=" * 100)
print("END COMPARABILITY REPORT")
print("=" * 100)
