#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    "objective_accuracy",
    "objective_balanced_accuracy",
    "objective_f1_weighted",
    "objective_auc_roc",
    "objective_auc_pr",
    "objective_f1_pa",
    "objective_vus_roc",
    "objective_vus_pr",
    "objective_mae",
    "objective_mse",
    "objective_rmse",
    "objective_mase",
    "objective_smape",
    "objective_crps",
    "objective_wql",
    "objective_mcis",
    "objective_pinball",
    "objective_skill_score_ratio",
    "objective_map_iou",
]


METRIC_LABELS = {
    "objective_accuracy": "Accuracy",
    "objective_balanced_accuracy": "Balanced Accuracy",
    "objective_f1_weighted": "Weighted F1",
    "objective_auc_roc": "AUC-ROC",
    "objective_auc_pr": "AUC-PR",
    "objective_f1_pa": "F1-PA",
    "objective_vus_roc": "VUS-ROC",
    "objective_vus_pr": "VUS-PR",
    "objective_mae": "MAE",
    "objective_mse": "MSE",
    "objective_rmse": "RMSE",
    "objective_mase": "MASE",
    "objective_smape": "SMAPE",
    "objective_crps": "CRPS",
    "objective_wql": "WQL",
    "objective_mcis": "MCIS",
    "objective_pinball": "Pinball Loss",
    "objective_skill_score_ratio": "Skill Score Ratio",
    "objective_map_iou": "mAP-IoU",
}


IMPORTANT_PAIRS = [
    ("objective_crps", "objective_wql"),
    ("objective_crps", "objective_pinball"),
    ("objective_wql", "objective_pinball"),
    ("objective_vus_roc", "objective_vus_pr"),
]


def short_name(value: object) -> str:
    if pd.isna(value):
        return "unknown"

    return str(value).split("[", maxsplit=1)[0].strip()


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(
        metric,
        metric.removeprefix("objective_").upper(),
    )


def numeric_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")

    return values.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def format_float(value: object, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"

    if not math.isfinite(number):
        return "N/A"

    return f"{number:.{digits}f}"


def load_results(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    df = pd.read_parquet(path).copy()

    required = {
        "dataset_name",
        "solver_name",
        "task",
    }

    missing = sorted(required.difference(df.columns))

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    if "is_result" in df.columns:
        result_mask = df["is_result"].fillna(False).astype(bool)
        df = df.loc[result_mask].copy()

    if "dataset_short" not in df.columns:
        df["dataset_short"] = df["dataset_name"].map(short_name)

    if "solver_short" not in df.columns:
        df["solver_short"] = df["solver_name"].map(short_name)

    return df.reset_index(drop=True)


def discover_metrics(df: pd.DataFrame) -> list[str]:
    discovered = []

    for metric in METRICS:
        if metric not in df.columns:
            continue

        values = numeric_series(df[metric])

        if values.notna().any():
            discovered.append(metric)

    return discovered


def build_metric_diagnostics(
    df: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    rows = []

    for metric in metrics:
        values = numeric_series(df[metric])
        valid = values.dropna()

        n_total = len(values)
        n_valid = len(valid)
        n_missing = n_total - n_valid
        n_unique = valid.nunique(dropna=True)

        if n_valid == 0:
            minimum = np.nan
            maximum = np.nan
            mean = np.nan
            std = np.nan
            median = np.nan
            is_constant = False
            near_constant_ratio = np.nan
        else:
            minimum = valid.min()
            maximum = valid.max()
            mean = valid.mean()
            std = valid.std()
            median = valid.median()
            is_constant = n_unique == 1

            most_frequent_count = (
                valid.value_counts(dropna=True).iloc[0]
            )

            near_constant_ratio = (
                most_frequent_count / n_valid
            )

        rows.append(
            {
                "metric": metric,
                "metric_label": metric_label(metric),
                "n_total_rows": n_total,
                "n_valid": n_valid,
                "n_missing": n_missing,
                "valid_ratio": (
                    n_valid / n_total if n_total else np.nan
                ),
                "n_unique": n_unique,
                "minimum": minimum,
                "maximum": maximum,
                "mean": mean,
                "median": median,
                "std": std,
                "is_constant": is_constant,
                "most_frequent_value_ratio": near_constant_ratio,
                "is_near_constant_95pct": (
                    bool(near_constant_ratio >= 0.95)
                    if not pd.isna(near_constant_ratio)
                    else False
                ),
            }
        )

    return pd.DataFrame(rows)


def compare_metric_pair(
    df: pd.DataFrame,
    metric_a: str,
    metric_b: str,
) -> dict[str, object]:
    a = numeric_series(df[metric_a])
    b = numeric_series(df[metric_b])

    common_mask = a.notna() & b.notna()
    a_common = a.loc[common_mask]
    b_common = b.loc[common_mask]

    n_common = int(common_mask.sum())

    if n_common == 0:
        return {
            "metric_a": metric_a,
            "metric_b": metric_b,
            "label_a": metric_label(metric_a),
            "label_b": metric_label(metric_b),
            "n_common": 0,
            "n_equal_exact": 0,
            "n_equal_close": 0,
            "exact_equal_ratio": np.nan,
            "close_equal_ratio": np.nan,
            "max_abs_difference": np.nan,
            "mean_abs_difference": np.nan,
            "correlation": np.nan,
            "identical_missing_pattern": bool(
                a.isna().equals(b.isna())
            ),
            "status": "no_common_values",
        }

    exact_equal = a_common.eq(b_common)

    close_equal = np.isclose(
        a_common.to_numpy(dtype=float),
        b_common.to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-12,
        equal_nan=False,
    )

    differences = np.abs(
        a_common.to_numpy(dtype=float)
        - b_common.to_numpy(dtype=float)
    )

    if (
        a_common.nunique() > 1
        and b_common.nunique() > 1
    ):
        correlation = a_common.corr(b_common)
    else:
        correlation = np.nan

    exact_ratio = float(exact_equal.mean())
    close_ratio = float(close_equal.mean())

    if close_ratio == 1.0:
        status = "identical_within_tolerance"
    elif close_ratio >= 0.99:
        status = "almost_identical"
    elif (
        not pd.isna(correlation)
        and abs(correlation) >= 0.999
    ):
        status = "nearly_perfectly_correlated"
    else:
        status = "different"

    return {
        "metric_a": metric_a,
        "metric_b": metric_b,
        "label_a": metric_label(metric_a),
        "label_b": metric_label(metric_b),
        "n_common": n_common,
        "n_equal_exact": int(exact_equal.sum()),
        "n_equal_close": int(close_equal.sum()),
        "exact_equal_ratio": exact_ratio,
        "close_equal_ratio": close_ratio,
        "max_abs_difference": float(differences.max()),
        "mean_abs_difference": float(differences.mean()),
        "correlation": correlation,
        "identical_missing_pattern": bool(
            a.isna().equals(b.isna())
        ),
        "status": status,
    }


def build_pairwise_diagnostics(
    df: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    rows = []

    for metric_a, metric_b in itertools.combinations(
        metrics,
        2,
    ):
        rows.append(
            compare_metric_pair(
                df,
                metric_a,
                metric_b,
            )
        )

    return pd.DataFrame(rows)


def build_important_pair_diagnostics(
    df: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    available = set(metrics)
    rows = []

    for metric_a, metric_b in IMPORTANT_PAIRS:
        if metric_a not in available:
            continue

        if metric_b not in available:
            continue

        rows.append(
            compare_metric_pair(
                df,
                metric_a,
                metric_b,
            )
        )

    return pd.DataFrame(rows)


def build_solver_coverage(
    df: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    rows = []

    for metric in metrics:
        values = numeric_series(df[metric])
        valid_df = df.loc[
            values.notna(),
            [
                "task",
                "dataset_name",
                "dataset_short",
                "solver_name",
                "solver_short",
            ],
        ].copy()

        if valid_df.empty:
            continue

        totals = valid_df.groupby("task")["dataset_name"].nunique()

        grouped = (
            valid_df.groupby(
                [
                    "task",
                    "solver_short",
                ],
                as_index=False,
                dropna=False,
            )
            .agg(
                n_dataset_configs=(
                    "dataset_name",
                    "nunique",
                ),
                n_dataset_families=(
                    "dataset_short",
                    "nunique",
                ),
                n_solver_configs=(
                    "solver_name",
                    "nunique",
                ),
                n_rows=(
                    "dataset_name",
                    "size",
                ),
            )
        )

        grouped["metric"] = metric
        grouped["metric_label"] = metric_label(metric)
        grouped["metric_total_datasets"] = grouped["task"].map(totals)
        grouped["coverage"] = (
            grouped["n_dataset_configs"]
            / grouped["metric_total_datasets"]
        )

        rows.append(grouped)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(
        rows,
        ignore_index=True,
    )

    return result[
        [
            "task",
            "metric",
            "metric_label",
            "solver_short",
            "n_solver_configs",
            "n_dataset_configs",
            "n_dataset_families",
            "metric_total_datasets",
            "coverage",
            "n_rows",
        ]
    ].sort_values(
        [
            "task",
            "metric",
            "coverage",
            "solver_short",
        ],
        ascending=[
            True,
            True,
            False,
            True,
        ],
    )


def build_dataset_solver_matrix(
    df: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    rows = []

    for metric in metrics:
        values = numeric_series(df[metric])

        metric_df = df.loc[
            values.notna(),
            [
                "task",
                "dataset_name",
                "dataset_short",
                "solver_short",
            ],
        ].copy()

        if metric_df.empty:
            continue

        metric_df["metric"] = metric
        metric_df["metric_label"] = metric_label(metric)
        metric_df["has_result"] = 1

        metric_df = metric_df.drop_duplicates(
            [
                "task",
                "metric",
                "dataset_name",
                "solver_short",
            ]
        )

        rows.append(metric_df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(
        rows,
        ignore_index=True,
    ).sort_values(
        [
            "task",
            "metric",
            "dataset_name",
            "solver_short",
        ]
    )


def build_seasonal_naive_diagnostics(
    df: pd.DataFrame,
    metrics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasonal = df.loc[
        df["solver_short"].eq("SeasonalNaive")
    ].copy()

    if seasonal.empty:
        return pd.DataFrame(), pd.DataFrame()

    config_rows = []
    oracle_rows = []

    for metric in metrics:
        values = numeric_series(seasonal[metric])

        metric_df = seasonal.loc[
            values.notna(),
            [
                "task",
                "dataset_name",
                "dataset_short",
                "solver_name",
                "solver_short",
            ],
        ].copy()

        metric_df["metric_value"] = values.loc[
            values.notna()
        ].to_numpy()

        if metric_df.empty:
            continue

        config_stats = (
            metric_df.groupby(
                ["task", "solver_name"],
                as_index=False,
            )
            .agg(
                n_datasets=(
                    "dataset_name",
                    "nunique",
                ),
                mean_value=(
                    "metric_value",
                    "mean",
                ),
                median_value=(
                    "metric_value",
                    "median",
                ),
                std_value=(
                    "metric_value",
                    "std",
                ),
            )
        )

        config_stats["metric"] = metric
        config_stats["metric_label"] = metric_label(metric)
        config_rows.append(config_stats)

        per_dataset = (
            metric_df.groupby(
                ["task", "dataset_name"],
                as_index=False,
            )
            .agg(
                n_available_configurations=(
                    "solver_name",
                    "nunique",
                ),
                min_value=(
                    "metric_value",
                    "min",
                ),
                max_value=(
                    "metric_value",
                    "max",
                ),
                mean_value=(
                    "metric_value",
                    "mean",
                ),
                std_value=(
                    "metric_value",
                    "std",
                ),
            )
        )

        per_dataset["metric"] = metric
        per_dataset["metric_label"] = metric_label(metric)

        per_dataset["absolute_config_spread"] = (
            per_dataset["max_value"]
            - per_dataset["min_value"]
        )

        denominator = (
            per_dataset["mean_value"]
            .abs()
            .replace(0, np.nan)
        )

        per_dataset["relative_config_spread"] = (
            per_dataset["absolute_config_spread"]
            / denominator
        )

        oracle_rows.append(per_dataset)

    config_df = (
        pd.concat(config_rows, ignore_index=True)
        if config_rows
        else pd.DataFrame()
    )

    oracle_df = (
        pd.concat(oracle_rows, ignore_index=True)
        if oracle_rows
        else pd.DataFrame()
    )

    return config_df, oracle_df


def build_duplicate_rows_diagnostic(
    df: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    key_columns = [
        "task",
        "dataset_name",
        "solver_name",
    ]

    available_metric_columns = [
        metric
        for metric in metrics
        if metric in df.columns
    ]

    columns = key_columns + available_metric_columns

    duplicated_mask = df.duplicated(
        key_columns,
        keep=False,
    )

    duplicates = df.loc[
        duplicated_mask,
        columns,
    ].copy()

    if duplicates.empty:
        return pd.DataFrame(
            columns=[
                *key_columns,
                "duplicate_count",
            ]
        )

    counts = (
        duplicates.groupby(
            key_columns,
            as_index=False,
            dropna=False,
        )
        .size()
        .rename(
            columns={"size": "duplicate_count"}
        )
    )

    return counts.sort_values(
        "duplicate_count",
        ascending=False,
    )


def build_report(
    input_path: Path,
    df: pd.DataFrame,
    metrics: list[str],
    metric_diagnostics: pd.DataFrame,
    important_pairs: pd.DataFrame,
    coverage: pd.DataFrame,
    seasonal_spread: pd.DataFrame,
    duplicates: pd.DataFrame,
) -> str:
    lines = [
        "=" * 92,
        "                    TSFM BENCHMARK — LEADERBOARD VALIDATION",
        "=" * 92,
        "",
        f"Input file                           : {input_path}",
        f"Usable rows                          : {len(df)}",
        f"Detected metrics                     : {len(metrics)}",
        f"Dataset configurations               : {df['dataset_name'].nunique()}",
        f"Dataset families                     : {df['dataset_short'].nunique()}",
        f"Solver configurations                : {df['solver_name'].nunique()}",
        f"Solver families                      : {df['solver_short'].nunique()}",
        "",
        "1. METRIC DIAGNOSTICS",
        "-" * 92,
    ]

    for row in metric_diagnostics.itertuples(index=False):
        flags = []

        if row.is_constant:
            flags.append("CONSTANT")

        if row.is_near_constant_95pct:
            flags.append("NEAR-CONSTANT")

        flag_text = (
            ", ".join(flags)
            if flags
            else "OK"
        )

        lines.append(
            f"{row.metric_label:24} | "
            f"valid={int(row.n_valid):4d} | "
            f"unique={int(row.n_unique):4d} | "
            f"min={format_float(row.minimum):>12} | "
            f"max={format_float(row.maximum):>12} | "
            f"{flag_text}"
        )

    lines.extend(
        [
            "",
            "2. KEY METRIC COMPARISONS",
            "-" * 92,
        ]
    )

    if important_pairs.empty:
        lines.append(
            "No key metric pairs available."
        )
    else:
        for row in important_pairs.itertuples(index=False):
            lines.append(
                f"{row.label_a:18} vs {row.label_b:18} | "
                f"paired values={int(row.n_common):4d} | "
                f"equal within tolerance={format_float(100 * row.close_equal_ratio, 2):>7}% | "
                f"correlation={format_float(row.correlation, 6):>10} | "
                f"status={row.status}"
            )

    lines.extend(
        [
            "",
            "3. COVERAGE BY SOLVER",
            "-" * 92,
        ]
    )

    incomplete = coverage.loc[
        coverage["coverage"] < 1.0
    ]

    if incomplete.empty:
        lines.append(
            "All families present for a metric have 100% coverage."
        )
    else:
        lines.append(
            f"{len(incomplete)} metric/solver pairs have incomplete coverage."
        )

        for row in incomplete.head(30).itertuples(index=False):
            lines.append(
                f"{row.metric_label:20} | "
                f"{row.solver_short:20} | "
                f"{int(row.n_dataset_configs)}/"
                f"{int(row.metric_total_datasets)} datasets | "
                f"coverage={format_float(100 * row.coverage, 2)} %"
            )

    lines.extend(
        [
            "",
            "4. SENSITIVITY TO SEASONALNAIVE CONFIGURATIONS",
            "-" * 92,
        ]
    )

    if seasonal_spread.empty:
        lines.append(
            "No SeasonalNaive configurations found."
        )
    else:
        spread_summary = (
            seasonal_spread.groupby(
                [
                    "metric",
                    "metric_label",
                ],
                as_index=False,
            )
            .agg(
                n_datasets=(
                    "dataset_name",
                    "nunique",
                ),
                mean_absolute_spread=(
                    "absolute_config_spread",
                    "mean",
                ),
                median_absolute_spread=(
                    "absolute_config_spread",
                    "median",
                ),
                mean_relative_spread=(
                    "relative_config_spread",
                    "mean",
                ),
                max_relative_spread=(
                    "relative_config_spread",
                    "max",
                ),
            )
        )

        for row in spread_summary.itertuples(index=False):
            lines.append(
                f"{row.metric_label:20} | "
                f"datasets={int(row.n_datasets):3d} | "
                f"mean relative spread="
                f"{format_float(100 * row.mean_relative_spread, 3):>8}% | "
                f"maximum relative spread="
                f"{format_float(100 * row.max_relative_spread, 3):>8}%"
            )

    lines.extend(
        [
            "",
            "5. DUPLICATE DATASET/SOLVER ROWS",
            "-" * 92,
        ]
    )

    if duplicates.empty:
        lines.append(
            "No duplicate task/dataset_name/solver_name keys."
        )
    else:
        lines.append(
            f"{len(duplicates)} dataset/solver pairs have multiple rows."
        )

        for row in duplicates.head(20).itertuples(index=False):
            lines.append(
                f"{row.task:20} | "
                f"{row.dataset_name:40} | "
                f"{row.solver_name:25} | "
                f"rows={int(row.duplicate_count)}"
            )

    lines.extend(
        [
            "",
            "6. AUTOMATED SUMMARY",
            "-" * 92,
        ]
    )

    identical_pairs = important_pairs.loc[
        important_pairs["status"].isin(
            [
                "identical_within_tolerance",
                "almost_identical",
            ]
        )
    ]

    constant_metrics = metric_diagnostics.loc[
        metric_diagnostics["is_constant"]
    ]

    near_constant_metrics = metric_diagnostics.loc[
        metric_diagnostics["is_near_constant_95pct"]
    ]

    if identical_pairs.empty:
        lines.append(
            "- No key metric pairs are identical or nearly identical."
        )
    else:
        lines.append(
            f"- {len(identical_pairs)} key metric pair(s) "
            "are identical or nearly identical."
        )

        for row in identical_pairs.itertuples(index=False):
            lines.append(
                f"  • {row.label_a} / {row.label_b}: {row.status}"
            )

    if constant_metrics.empty:
        lines.append(
            "- No strictly constant metrics."
        )
    else:
        labels = ", ".join(
            constant_metrics["metric_label"].tolist()
        )
        lines.append(
            f"- Constant metrics: {labels}."
        )

    if near_constant_metrics.empty:
        lines.append(
            "- No metrics have the same value in at least 95% of valid observations."
        )
    else:
        labels = ", ".join(
            near_constant_metrics["metric_label"].tolist()
        )
        lines.append(
            f"- Near-constant metrics: {labels}."
        )

    if incomplete.empty:
        lines.append(
            "- All solvers present have complete coverage."
        )
    else:
        lines.append(
            "- Some comparisons have incomplete coverage."
        )

    if duplicates.empty:
        lines.append(
            "- No duplicate configuration keys found."
        )
    else:
        lines.append(
            "- Duplicate configuration keys exist and require inspection."
        )

    lines.extend(
        [
            "",
            "=" * 92,
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the scientific and structural consistency "
            "of the TSFM leaderboard."
        )
    )

    parser.add_argument(
        "input_file",
        nargs="?",
        default="benchmark_tsfm_results_only.parquet",
        help="Parquet file containing usable results.",
    )

    parser.add_argument(
        "--output-dir",
        default=".",
        help="Root output directory.",
    )

    args = parser.parse_args()

    input_path = Path(
        args.input_file
    ).expanduser().resolve()

    output_dir = Path(
        args.output_dir
    ).expanduser().resolve()

    csv_dir = output_dir / "csv"
    reports_dir = output_dir / "reports"

    csv_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Loading: {input_path}")

    df = load_results(input_path)
    metrics = discover_metrics(df)

    if not metrics:
        raise RuntimeError(
            "No usable metrics found."
        )

    print(f"Usable rows: {len(df)}")
    print(f"Detected metrics: {len(metrics)}")

    metric_diagnostics = build_metric_diagnostics(
        df,
        metrics,
    )

    pairwise_diagnostics = build_pairwise_diagnostics(
        df,
        metrics,
    )

    important_pairs = build_important_pair_diagnostics(
        df,
        metrics,
    )

    coverage = build_solver_coverage(
        df,
        metrics,
    )

    dataset_solver_matrix = build_dataset_solver_matrix(
        df,
        metrics,
    )

    seasonal_configs, seasonal_spread = (
        build_seasonal_naive_diagnostics(
            df,
            metrics,
        )
    )

    duplicates = build_duplicate_rows_diagnostic(
        df,
        metrics,
    )

    report = build_report(
        input_path=input_path,
        df=df,
        metrics=metrics,
        metric_diagnostics=metric_diagnostics,
        important_pairs=important_pairs,
        coverage=coverage,
        seasonal_spread=seasonal_spread,
        duplicates=duplicates,
    )

    paths = {
        "metric_diagnostics": (
            csv_dir / "metric_diagnostics.csv"
        ),
        "pairwise_diagnostics": (
            csv_dir / "metric_pairwise_comparisons.csv"
        ),
        "important_pairs": (
            csv_dir / "important_metric_comparisons.csv"
        ),
        "coverage": (
            csv_dir / "solver_metric_coverage.csv"
        ),
        "dataset_solver_matrix": (
            csv_dir / "dataset_solver_availability.csv"
        ),
        "seasonal_configs": (
            csv_dir / "seasonal_naive_config_stats.csv"
        ),
        "seasonal_spread": (
            csv_dir / "seasonal_naive_config_spread.csv"
        ),
        "duplicates": (
            csv_dir / "duplicate_dataset_solver_rows.csv"
        ),
        "report": (
            reports_dir / "leaderboard_validation_report.txt"
        ),
    }

    metric_diagnostics.to_csv(
        paths["metric_diagnostics"],
        index=False,
    )

    pairwise_diagnostics.to_csv(
        paths["pairwise_diagnostics"],
        index=False,
    )

    important_pairs.to_csv(
        paths["important_pairs"],
        index=False,
    )

    coverage.to_csv(
        paths["coverage"],
        index=False,
    )

    dataset_solver_matrix.to_csv(
        paths["dataset_solver_matrix"],
        index=False,
    )

    seasonal_configs.to_csv(
        paths["seasonal_configs"],
        index=False,
    )

    seasonal_spread.to_csv(
        paths["seasonal_spread"],
        index=False,
    )

    duplicates.to_csv(
        paths["duplicates"],
        index=False,
    )

    paths["report"].write_text(
        report,
        encoding="utf-8",
    )

    print()
    print(report)
    print()
    print("Generated files:")

    for path in paths.values():
        print(f"  - {path}")


if __name__ == "__main__":
    main()
