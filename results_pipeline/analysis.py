#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Metrics associated with each task type.
TASK_METRICS = {
    "classification": [
        "objective_accuracy",
        "objective_balanced_accuracy",
        "objective_f1_weighted",
    ],
    "anomaly_detection": [
        "objective_auc_roc",
        "objective_auc_pr",
        "objective_f1_pa",
        "objective_vus_roc",
        "objective_vus_pr",
    ],
    "forecasting": [
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
    ],
    "segmentation": [
        "objective_map_iou",
    ],
}


# Primary mapping from dataset families to task types.
#
# This is more robust than relying solely on metric-based detection:
# some GiftEval placeholder rows contain
# no performance values.
DATASET_TASKS = {
    "Dummy-Classification": "classification",
    "UCR": "classification",

    "ECG": "anomaly_detection",
    "SVDB": "anomaly_detection",
    "YAHOO": "anomaly_detection",

    "Enedis": "forecasting",
    "FEV": "forecasting",
    "GiftEval": "forecasting",
    "Monash": "forecasting",

    "MITDB": "segmentation",
}


# Ordered collection of all performance metrics.
#
# objective_name, objective_value and objective_leakage are not used
# to determine whether a row contains a usable result.
PERFORMANCE_METRICS = list(
    dict.fromkeys(
        metric
        for metrics in TASK_METRICS.values()
        for metric in metrics
    )
)


def short_name(value: object) -> str:
    """Return the component name before its parameter list."""
    if pd.isna(value):
        return "unknown"

    text = str(value)
    return text.split("[", maxsplit=1)[0].strip()


def detect_task_from_metrics(row: pd.Series) -> str:
    """
    Infer the task from non-null performance metrics.

    This function is only a fallback for datasets that are not yet listed
    in DATASET_TASKS.
    """
    detected_tasks: list[str] = []

    for task, metrics in TASK_METRICS.items():
        available_metrics = [
            metric
            for metric in metrics
            if metric in row.index
        ]

        if available_metrics and pd.to_numeric(row[available_metrics], errors="coerce").replace(
            [np.inf, -np.inf], np.nan,
        ).notna().any():
            detected_tasks.append(task)

    if len(detected_tasks) == 1:
        return detected_tasks[0]

    if len(detected_tasks) > 1:
        return "+".join(detected_tasks)

    return "unknown"


def detect_task(row: pd.Series) -> str:
    """
    Infer the task primarily from the dataset family.

    Metric-based detection is used as a fallback for future datasets that
    are not yet registered in DATASET_TASKS.
    """
    dataset_family = short_name(row.get("dataset_name"))

    if dataset_family in DATASET_TASKS:
        return DATASET_TASKS[dataset_family]

    return detect_task_from_metrics(row)


def unique_non_null(df: pd.DataFrame, column: str) -> list[str]:
    """Return sorted unique non-null values from a DataFrame column."""
    if column not in df.columns:
        return []

    return sorted(
        df[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def first_non_null(df: pd.DataFrame, column: str) -> str:
    """Return the first available value of a column."""
    values = unique_non_null(df, column)
    return values[0] if values else "not available"


def numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric Series or an all-NaN Series if the column is absent."""
    if column not in df.columns:
        return pd.Series(
            float("nan"),
            index=df.index,
            dtype="float64",
        )

    return pd.to_numeric(df[column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan,
    )


def format_seconds(seconds: float) -> str:
    """Format seconds as hours, minutes and seconds."""
    if pd.isna(seconds):
        return "not available"

    seconds = float(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)

    return (
        f"{int(hours)} h "
        f"{int(minutes):02d} min "
        f"{remaining_seconds:05.2f} s"
    )


def build_result_mask(df: pd.DataFrame) -> pd.Series:
    """
    Identify rows containing at least one real performance metric.

    Technical rows such as some GiftEval placeholder rows contain no MAE,
    RMSE, CRPS, SMAPE or other task metric. They are retained in the enriched
    dataset but excluded from performance analyses.
    """
    available_metrics = [
        metric
        for metric in PERFORMANCE_METRICS
        if metric in df.columns
    ]

    if not available_metrics:
        return pd.Series(False, index=df.index, dtype=bool)

    numeric_metrics = df[available_metrics].apply(
        pd.to_numeric,
        errors="coerce",
    )

    return numeric_metrics.replace([np.inf, -np.inf], np.nan).notna().any(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze TSFM benchmark results."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default="benchmark_tsfm_complete.parquet",
        help="Parquet file to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for generated results.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    csv_dir = output_dir / "csv"
    reports_dir = output_dir / "reports"
    figures_dir = output_dir / "figures"
    html_dir = output_dir / "html"

    for directory in [
        csv_dir,
        reports_dir,
        figures_dir,
        html_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Parquet file not found: {input_path}"
        )

    print(f"Loading: {input_path}")

    # ------------------------------------------------------------------
    # 1. Loading and enriching raw data
    # ------------------------------------------------------------------
    df = pd.read_parquet(input_path).copy()

    required_columns = [
        "dataset_name",
        "solver_name",
    ]

    missing_required = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_required:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_required)
        )

    if df[required_columns].isna().any().any():
        raise ValueError("Dataset and solver identifiers must not be missing")

    # Timing is optional; normalize it before all summaries and exports.
    df["time"] = numeric_column(df, "time")
    df["dataset_short"] = df["dataset_name"].map(short_name)
    df["solver_short"] = df["solver_name"].map(short_name)

    # Identify the task type.
    df["task"] = df.apply(detect_task, axis=1)

    # Separate usable results from non-result rows.
    df["is_result"] = build_result_mask(df)
    df["row_type"] = df["is_result"].map(
        {
            True: "result",
            False: "technical",
        }
    )

    df_results = df.loc[df["is_result"]].copy()
    df_technical = df.loc[~df["is_result"]].copy()

    # ------------------------------------------------------------------
    # 2. General statistics
    # ------------------------------------------------------------------
    total_rows = len(df)
    total_result_rows = len(df_results)
    total_technical_rows = len(df_technical)

    total_dataset_configs = df["dataset_name"].nunique()
    total_dataset_families = df["dataset_short"].nunique()

    total_solver_configs = df["solver_name"].nunique()
    total_solver_families = df["solver_short"].nunique()

    total_tasks = df["task"].nunique()

    valid_times = numeric_column(df, "time")
    total_time = valid_times.sum(min_count=1)
    mean_time = valid_times.mean()
    median_time = valid_times.median()

    result_times = numeric_column(df_results, "time")
    result_total_time = result_times.sum(min_count=1)
    result_mean_time = result_times.mean()
    result_median_time = result_times.median()

    # ------------------------------------------------------------------
    # 3. Row counts by task
    # ------------------------------------------------------------------
    task_counts = (
        df.groupby("task", dropna=False)
        .agg(
            n_rows=("dataset_name", "size"),
            n_result_rows=("is_result", "sum"),
        )
        .reset_index()
    )

    task_counts["n_technical_rows"] = (
        task_counts["n_rows"]
        - task_counts["n_result_rows"]
    )

    task_counts = task_counts.sort_values(
        "n_rows",
        ascending=False,
    )

    # ------------------------------------------------------------------
    # 4. Statistics by dataset
    # ------------------------------------------------------------------
    dataset_stats = (
        df.groupby(
            ["task", "dataset_short"],
            dropna=False,
        )
        .agg(
            n_rows=("dataset_name", "size"),
            n_result_rows=("is_result", "sum"),
            n_configurations=("dataset_name", "nunique"),
            n_solvers=("solver_short", "nunique"),
            total_time_seconds=("time", "sum"),
            mean_time_seconds=("time", "mean"),
        )
        .reset_index()
    )

    dataset_stats["n_technical_rows"] = (
        dataset_stats["n_rows"]
        - dataset_stats["n_result_rows"]
    )

    dataset_stats = dataset_stats.sort_values(
        ["task", "dataset_short"]
    )

    # ------------------------------------------------------------------
    # 5. Statistics by solver
    # ------------------------------------------------------------------
    solver_stats = (
        df.groupby(
            "solver_short",
            dropna=False,
        )
        .agg(
            n_rows=("solver_name", "size"),
            n_result_rows=("is_result", "sum"),
            n_configurations=("solver_name", "nunique"),
            n_datasets=("dataset_name", "nunique"),
            n_tasks=("task", "nunique"),
            total_time_seconds=("time", "sum"),
            mean_time_seconds=("time", "mean"),
            median_time_seconds=("time", "median"),
        )
        .reset_index()
    )

    solver_stats["n_technical_rows"] = (
        solver_stats["n_rows"]
        - solver_stats["n_result_rows"]
    )

    solver_stats = solver_stats.sort_values(
        "n_rows",
        ascending=False,
    )

    # ------------------------------------------------------------------
    # 6. Non-result row diagnostics
    # ------------------------------------------------------------------
    if not df_technical.empty:
        technical_stats = (
            df_technical.groupby(
                [
                    "task",
                    "dataset_short",
                    "solver_short",
                ],
                dropna=False,
            )
            .size()
            .rename("n_technical_rows")
            .reset_index()
            .sort_values(
                "n_technical_rows",
                ascending=False,
            )
        )
    else:
        technical_stats = pd.DataFrame(
            columns=[
                "task",
                "dataset_short",
                "solver_short",
                "n_technical_rows",
            ]
        )

    # ------------------------------------------------------------------
    # 7. Dates
    # ------------------------------------------------------------------
    run_dates = unique_non_null(df, "run_date")

    run_date_min = (
        min(run_dates)
        if run_dates
        else "not available"
    )

    run_date_max = (
        max(run_dates)
        if run_dates
        else "not available"
    )

    # ------------------------------------------------------------------
    # 8. Build the text summary
    # ------------------------------------------------------------------
    summary_lines = [
        "=" * 72,
        "                    TSFM BENCHMARK — GENERAL SUMMARY",
        "=" * 72,
        "",
        f"Input file                           : {input_path}",
        f"Total rows                           : {total_rows}",
        f"Usable result rows                   : {total_result_rows}",
        f"Non-result rows                      : {total_technical_rows}",
        f"Dataset configurations               : {total_dataset_configs}",
        f"Dataset families                     : {total_dataset_families}",
        f"Solver configurations                : {total_solver_configs}",
        f"Solver families                      : {total_solver_families}",
        f"Task types                           : {total_tasks}",
        "",
        "INTERNAL TIMINGS RECORDED BY BENCHOPT",
        "-" * 72,
        f"Total time — all rows                : {format_seconds(total_time)}",
        f"Mean time — all rows                 : {format_seconds(mean_time)}",
        f"Median time — all rows               : {format_seconds(median_time)}",
        "",
        f"Total time — result rows             : {format_seconds(result_total_time)}",
        f"Mean time — result rows              : {format_seconds(result_mean_time)}",
        f"Median time — result rows            : {format_seconds(result_median_time)}",
        "",
        "Note                                  : the time column does not represent",
        "                                   the total wall-clock time of Slurm jobs.",
        "",
        "ENVIRONMENT",
        "-" * 72,
        f"Platform                             : "
        f"{first_non_null(df, 'platform')}",
        f"Architecture                         : "
        f"{first_non_null(df, 'platform-architecture')}",
        f"CPU count                            : "
        f"{first_non_null(df, 'system-cpus')}",
        f"System memory                        : "
        f"{first_non_null(df, 'system-ram (GB)')} GB",
        f"CUDA                                 : "
        f"{first_non_null(df, 'version-cuda')}",
        f"NumPy                                : "
        f"{first_non_null(df, 'version-numpy')}",
        f"Benchmark version                    : "
        f"{first_non_null(df, 'benchmark-git-tag')}",
        "",
        "RUN DATES",
        "-" * 72,
        f"Earliest date                        : {run_date_min}",
        f"Latest date                          : {run_date_max}",
        "",
        "ROWS BY TASK",
        "-" * 72,
    ]

    for row in task_counts.itertuples(index=False):
        summary_lines.append(
            f"{str(row.task):25} : "
            f"{int(row.n_rows):4d} total | "
            f"{int(row.n_result_rows):4d} results | "
            f"{int(row.n_technical_rows):4d} non-result rows"
        )

    summary_lines.extend(
        [
            "",
            "DATASET FAMILIES",
            "-" * 72,
        ]
    )

    dataset_summary = (
        df.groupby("dataset_short", dropna=False)
        .agg(
            n_rows=("dataset_name", "size"),
            n_results=("is_result", "sum"),
        )
        .reset_index()
        .sort_values("dataset_short")
    )

    for row in dataset_summary.itertuples(index=False):
        n_technical = int(row.n_rows - row.n_results)

        summary_lines.append(
            f"{str(row.dataset_short):25} : "
            f"{int(row.n_rows):4d} total | "
            f"{int(row.n_results):4d} results | "
            f"{n_technical:4d} non-result rows"
        )

    summary_lines.extend(
        [
            "",
            "SOLVER FAMILIES",
            "-" * 72,
        ]
    )

    solver_summary = (
        df.groupby("solver_short", dropna=False)
        .agg(
            n_rows=("solver_name", "size"),
            n_results=("is_result", "sum"),
        )
        .reset_index()
        .sort_values("solver_short")
    )

    for row in solver_summary.itertuples(index=False):
        n_technical = int(row.n_rows - row.n_results)

        summary_lines.append(
            f"{str(row.solver_short):25} : "
            f"{int(row.n_rows):4d} total | "
            f"{int(row.n_results):4d} results | "
            f"{n_technical:4d} non-result rows"
        )

    summary_lines.extend(
        [
            "",
            "INTERPRETATION",
            "-" * 72,
            "Non-result rows are preserved in the enriched file.",
            "They are excluded from leaderboards, averages, wins and",
            "heatmaps because they contain no valid performance metrics.",
            "",
            "=" * 72,
        ]
    )

    summary = "\n".join(summary_lines)

    # ------------------------------------------------------------------
    # 9. Write output files
    # ------------------------------------------------------------------
    summary_path = reports_dir / "summary.txt"

    task_counts_path = csv_dir / "task_counts.csv"
    dataset_stats_path = csv_dir / "dataset_stats.csv"
    solver_stats_path = csv_dir / "solver_stats.csv"
    technical_stats_path = csv_dir / "technical_rows_stats.csv"

    enriched_results_path = (
        output_dir / "benchmark_tsfm_enriched.parquet"
    )

    valid_results_path = (
        output_dir / "benchmark_tsfm_results_only.parquet"
    )

    technical_rows_path = (
        output_dir / "benchmark_tsfm_technical_rows.parquet"
    )

    summary_path.write_text(
        summary,
        encoding="utf-8",
    )

    task_counts.to_csv(
        task_counts_path,
        index=False,
    )

    dataset_stats.to_csv(
        dataset_stats_path,
        index=False,
    )

    solver_stats.to_csv(
        solver_stats_path,
        index=False,
    )

    technical_stats.to_csv(
        technical_stats_path,
        index=False,
    )

    # Preserve all rows in the enriched export.
    df.to_parquet(
        enriched_results_path,
        index=False,
    )

    # Only usable rows are retained for scientific analyses.
    df_results.to_parquet(
        valid_results_path,
        index=False,
    )

    # Export non-result rows separately for inspection and traceability.
    df_technical.to_parquet(
        technical_rows_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # 10. Print the final summary
    # ------------------------------------------------------------------
    print()
    print(summary)
    print()
    print("Generated files:")
    print(f"  - {summary_path}")
    print(f"  - {task_counts_path}")
    print(f"  - {dataset_stats_path}")
    print(f"  - {solver_stats_path}")
    print(f"  - {technical_stats_path}")
    print(f"  - {enriched_results_path}")
    print(f"  - {valid_results_path}")
    print(f"  - {technical_rows_path}")


if __name__ == "__main__":
    main()
