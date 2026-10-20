#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Optimization direction for each metric
# ---------------------------------------------------------------------------

METRIC_DIRECTION: dict[str, Literal["min", "max"]] = {
    # Classification: higher is better
    "objective_accuracy": "max",
    "objective_balanced_accuracy": "max",
    "objective_f1_weighted": "max",

    # Anomaly detection: higher is better
    "objective_auc_roc": "max",
    "objective_auc_pr": "max",
    "objective_f1_pa": "max",
    "objective_vus_roc": "max",
    "objective_vus_pr": "max",

    # Forecasting: lower is better
    "objective_mae": "min",
    "objective_mse": "min",
    "objective_rmse": "min",
    "objective_mase": "min",
    "objective_smape": "min",
    "objective_crps": "min",
    "objective_wql": "min",
    "objective_mcis": "min",
    "objective_pinball": "min",

    # Check the exact definition in the BenchOpt objective.
    # Adjust this direction if the definition changes.
    "objective_skill_score_ratio": "max",

    # Segmentation: higher is better
    "objective_map_iou": "max",
}


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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def short_name(value: object) -> str:
    """Return the component name before its parameter list."""
    if pd.isna(value):
        return "unknown"

    return str(value).split("[", maxsplit=1)[0].strip()


def finite_numeric(series: pd.Series) -> pd.Series:
    """Convert a Series to numeric values and replace infinities with NaN."""
    numeric = pd.to_numeric(series, errors="coerce")

    return numeric.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def metric_label(metric: str) -> str:
    """Return a readable metric name."""
    return METRIC_LABELS.get(
        metric,
        metric.removeprefix("objective_").upper(),
    )


def discover_metrics(df: pd.DataFrame) -> list[str]:
    """
    Return known metrics containing at least one finite value.
    """
    available: list[str] = []

    for metric in METRIC_DIRECTION:
        if metric not in df.columns:
            continue

        values = finite_numeric(df[metric])

        if values.notna().any():
            available.append(metric)

    return available


def validate_input(df: pd.DataFrame) -> None:
    """Validate the columns required by the leaderboard pipeline."""
    required_columns = {
        "dataset_name",
        "solver_name",
        "task",
    }

    missing = sorted(required_columns.difference(df.columns))

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    if df[["dataset_name", "solver_name", "task"]].isna().any().any():
        raise ValueError("Dataset, solver and task identifiers must not be missing")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results(path: Path) -> pd.DataFrame:
    """
    Load the cleaned TSFM results.

    Technical rows are excluded again if the is_result column exists,
    even though the expected input is already results-only.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Parquet file not found: {path}"
        )

    df = pd.read_parquet(path).copy()

    validate_input(df)

    if "is_result" in df.columns:
        mask = df["is_result"].fillna(False).astype(bool)
        df = df.loc[mask].copy()

    if "dataset_short" not in df.columns:
        df["dataset_short"] = df["dataset_name"].map(short_name)

    if "solver_short" not in df.columns:
        df["solver_short"] = df["solver_name"].map(short_name)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Observation preparation
# ---------------------------------------------------------------------------

def aggregate_solver_families(
    metric_df: pd.DataFrame,
    direction: Literal["min", "max"],
) -> pd.DataFrame:
    """
    Reduce several configurations of the same solver family to one value.

    For each dataset configuration and solver family, the best available
    configuration is retained.

    This mainly affects SeasonalNaive, which has several season lengths.

    Important:
    this is an oracle-style family aggregation. The configuration-level
    results are also exported separately for full transparency.
    """
    ascending = direction == "min"

    ordered = metric_df.sort_values(
        ["dataset_name", "solver_short", "metric_value"],
        ascending=[True, True, ascending],
    )

    family_df = (
        ordered.groupby(
            [
                "task",
                "dataset_name",
                "dataset_short",
                "solver_short",
            ],
            as_index=False,
            dropna=False,
        )
        .first()
    )

    return family_df


def prepare_metric_data(
    df: pd.DataFrame,
    metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare configuration-level and family-level data for one metric.
    """
    columns = [
        "task",
        "dataset_name",
        "dataset_short",
        "solver_name",
        "solver_short",
        metric,
    ]

    metric_df = df[columns].copy()
    metric_df["metric_value"] = finite_numeric(metric_df[metric])
    metric_df = metric_df.dropna(subset=["metric_value"])

    metric_df = metric_df.drop(columns=[metric])

    # If duplicate rows exist for the exact same dataset/solver pair,
    # retain their mean rather than counting them several times.
    config_df = (
        metric_df.groupby(
            [
                "task",
                "dataset_name",
                "dataset_short",
                "solver_name",
                "solver_short",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            metric_value=("metric_value", "mean"),
            n_raw_rows=("metric_value", "size"),
        )
    )

    family_df = aggregate_solver_families(
        config_df,
        METRIC_DIRECTION[metric],
    )

    return config_df, family_df


# ---------------------------------------------------------------------------
# Ranks and wins
# ---------------------------------------------------------------------------

def rank_dataset_results(
    data: pd.DataFrame,
    metric: str,
    entity_column: str,
) -> pd.DataFrame:
    """
    Rank entities independently on every dataset configuration.

    Ties:
    - average rank is used;
    - rank 1 ties each receive a fractional win of 1 / number_of_winners.
    """
    direction = METRIC_DIRECTION[metric]
    ascending = direction == "min"

    ranked_groups: list[pd.DataFrame] = []

    for _, group in data.groupby(
        ["task", "dataset_name"],
        sort=False,
        dropna=False,
    ):
        group = group.copy()

        if group.empty:
            continue

        group["rank"] = group["metric_value"].rank(
            method="average",
            ascending=ascending,
        )

        winners = group["rank"].eq(group["rank"].min())

        n_winners = int(winners.sum())

        group["is_winner"] = winners
        group["fractional_win"] = 0.0

        if n_winners > 0:
            group.loc[winners, "fractional_win"] = 1.0 / n_winners

        group["n_competitors"] = group[entity_column].nunique()
        group["metric"] = metric
        group["metric_label"] = metric_label(metric)
        group["direction"] = direction

        ranked_groups.append(group)

    if not ranked_groups:
        return pd.DataFrame()

    return pd.concat(
        ranked_groups,
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# Summary by solver
# ---------------------------------------------------------------------------

def build_leaderboard(
    ranked: pd.DataFrame,
    entity_column: str,
    entity_type: str,
) -> pd.DataFrame:
    """
    Aggregate dataset-level ranks into a metric-level leaderboard.
    """
    if ranked.empty:
        return pd.DataFrame()

    leaderboard = (
        ranked.groupby(
            [
                "metric",
                "metric_label",
                "direction",
                "task",
                entity_column,
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            n_datasets=("dataset_name", "nunique"),
            mean_value=("metric_value", "mean"),
            median_value=("metric_value", "median"),
            std_value=("metric_value", "std"),
            min_value=("metric_value", "min"),
            max_value=("metric_value", "max"),
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            best_rank=("rank", "min"),
            worst_rank=("rank", "max"),
            wins=("fractional_win", "sum"),
            n_first_places=("is_winner", "sum"),
            mean_competitors=("n_competitors", "mean"),
        )
    )

    leaderboard = leaderboard.rename(
        columns={entity_column: "solver"}
    )

    leaderboard["entity_type"] = entity_type

    # Coverage for a metric = proportion of metric datasets on which
    # the solver has a valid result.
    metric_dataset_counts = (
        ranked.groupby(["task", "metric"])["dataset_name"]
        .nunique()
        .rename("metric_total_datasets")
    )

    leaderboard = leaderboard.merge(
        metric_dataset_counts,
        on=["task", "metric"],
        how="left",
    )

    leaderboard["coverage"] = (
        leaderboard["n_datasets"]
        / leaderboard["metric_total_datasets"]
    )

    # Sort by metric, coverage, mean rank and wins.
    leaderboard = leaderboard.sort_values(
        [
            "task",
            "metric",
            "coverage",
            "mean_rank",
            "wins",
            "solver",
        ],
        ascending=[
            True,
            True,
            False,
            True,
            False,
            True,
        ],
    )

    leaderboard["leaderboard_position"] = (
        leaderboard.groupby(["task", "metric"]).cumcount() + 1
    )

    ordered_columns = [
        "entity_type",
        "task",
        "metric",
        "metric_label",
        "direction",
        "leaderboard_position",
        "solver",
        "n_datasets",
        "metric_total_datasets",
        "coverage",
        "mean_rank",
        "median_rank",
        "best_rank",
        "worst_rank",
        "wins",
        "n_first_places",
        "mean_value",
        "median_value",
        "std_value",
        "min_value",
        "max_value",
        "mean_competitors",
    ]

    return leaderboard[ordered_columns]


# ---------------------------------------------------------------------------
# Overall ranking across metrics
# ---------------------------------------------------------------------------

def build_global_ranking(
    family_leaderboard: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a global descriptive ranking from mean ranks across metrics.

    Metrics are weighted equally. This summary must not replace
    task-specific leaderboards because the solvers do not necessarily
    participate in the same tasks or metrics.
    """
    if family_leaderboard.empty:
        return pd.DataFrame()

    global_ranking = (
        family_leaderboard.groupby(
            "solver",
            as_index=False,
            dropna=False,
        )
        .agg(
            n_tasks=("task", "nunique"),
            n_metrics=("metric", "nunique"),
            total_dataset_evaluations=("n_datasets", "sum"),
            mean_coverage=("coverage", "mean"),
            global_mean_rank=("mean_rank", "mean"),
            global_median_rank=("median_rank", "median"),
            total_fractional_wins=("wins", "sum"),
            total_first_places=("n_first_places", "sum"),
        )
    )

    global_ranking = global_ranking.sort_values(
        [
            "mean_coverage",
            "global_mean_rank",
            "total_fractional_wins",
            "solver",
        ],
        ascending=[
            False,
            True,
            False,
            True,
        ],
    ).reset_index(drop=True)

    global_ranking["global_position"] = (
        np.arange(len(global_ranking)) + 1
    )

    columns = [
        "global_position",
        "solver",
        "n_tasks",
        "n_metrics",
        "total_dataset_evaluations",
        "mean_coverage",
        "global_mean_rank",
        "global_median_rank",
        "total_fractional_wins",
        "total_first_places",
    ]

    return global_ranking[columns]


# ---------------------------------------------------------------------------
# Metric summary
# ---------------------------------------------------------------------------

def build_metric_summary(
    ranked_family: pd.DataFrame,
    leaderboard_family: pd.DataFrame,
) -> pd.DataFrame:
    """Build one summary row per available metric."""
    if ranked_family.empty:
        return pd.DataFrame()

    metric_summary = (
        ranked_family.groupby(
            [
                "task",
                "metric",
                "metric_label",
                "direction",
            ],
            as_index=False,
        )
        .agg(
            n_datasets=("dataset_name", "nunique"),
            n_solver_families=("solver_short", "nunique"),
            n_evaluations=("metric_value", "size"),
            min_observed_value=("metric_value", "min"),
            max_observed_value=("metric_value", "max"),
        )
    )

    winners = (
        leaderboard_family.loc[
            leaderboard_family["leaderboard_position"] == 1,
            [
                "metric",
                "solver",
                "mean_rank",
                "wins",
                "coverage",
            ],
        ]
        .rename(
            columns={
                "solver": "leader",
                "mean_rank": "leader_mean_rank",
                "wins": "leader_wins",
                "coverage": "leader_coverage",
            }
        )
    )

    return metric_summary.merge(
        winners,
        on="metric",
        how="left",
    )


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def format_float(value: object, digits: int = 4) -> str:
    """Safely format a numeric value."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"

    if not math.isfinite(numeric):
        return "N/A"

    return f"{numeric:.{digits}f}"


def build_report(
    input_path: Path,
    df: pd.DataFrame,
    metrics: list[str],
    family_leaderboard: pd.DataFrame,
    global_ranking: pd.DataFrame,
) -> str:
    """Create a readable text report."""
    lines = [
        "=" * 88,
        "                         TSFM BENCHMARK — LEADERBOARD",
        "=" * 88,
        "",
        f"Input file                           : {input_path}",
        f"Usable rows                          : {len(df)}",
        f"Available metrics                    : {len(metrics)}",
        f"Solver families                      : {df['solver_short'].nunique()}",
        f"Dataset configurations               : {df['dataset_name'].nunique()}",
        "",
        "METHODOLOGY",
        "-" * 88,
        "1. Solvers are ranked independently on each dataset configuration.",
        "2. Each metric uses its declared minimization or maximization direction.",
        "3. Ties receive average ranks; tied winners share one win.",
        "4. For family rankings, the best available configuration of each family",
        "   is selected separately for each dataset and metric.",
        "5. Mean ranks are less sensitive than mean raw metric values to differences",
        "   in scale across datasets.",
        "",
        "DETECTED METRICS",
        "-" * 88,
    ]

    for metric in metrics:
        lines.append(
            f"{metric_label(metric):28} "
            f"({metric:30}) "
            f"direction={METRIC_DIRECTION[metric]}"
        )

    lines.extend(
        [
            "",
            "LEADERS BY METRIC — SOLVER FAMILIES",
            "-" * 88,
        ]
    )

    for metric in metrics:
        subset = family_leaderboard.loc[
            family_leaderboard["metric"] == metric
        ].sort_values("leaderboard_position")

        if subset.empty:
            continue

        first = subset.iloc[0]

        lines.append("")
        lines.append(
            f"{metric_label(metric)} "
            f"[{first['task']}, {first['direction']}]"
        )

        top = subset.head(10)

        for row in top.itertuples(index=False):
            lines.append(
                f"  {int(row.leaderboard_position):2d}. "
                f"{str(row.solver):20} | "
                f"mean rank={format_float(row.mean_rank, 3):>7} | "
                f"wins={format_float(row.wins, 2):>6} | "
                f"coverage={format_float(100 * row.coverage, 1):>5}% | "
                f"datasets={int(row.n_datasets)}"
            )

    lines.extend(
        [
            "",
            "DESCRIPTIVE OVERALL RANKING",
            "-" * 88,
            "Note: this ranking aggregates multiple tasks and metrics.",
            "It is a descriptive summary, not evidence",
            "that a model is universally better.",
            "",
        ]
    )

    for row in global_ranking.itertuples(index=False):
        lines.append(
            f"{int(row.global_position):2d}. "
            f"{str(row.solver):20} | "
            f"overall mean rank={format_float(row.global_mean_rank, 3):>7} | "
            f"wins={format_float(row.total_fractional_wins, 2):>6} | "
            f"metrics={int(row.n_metrics):2d} | "
            f"coverage={format_float(100 * row.mean_coverage, 1):>5}%"
        )

    lines.extend(
        [
            "",
            "=" * 88,
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build TSFM benchmark leaderboards "
            "from cleaned results."
        )
    )

    parser.add_argument(
        "input_file",
        nargs="?",
        default="benchmark_tsfm_results_only.parquet",
        help="Parquet file containing usable results only.",
    )

    parser.add_argument(
        "--output-dir",
        default=".",
        help="Root output directory.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    csv_dir = output_dir / "csv"
    reports_dir = output_dir / "reports"

    csv_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {input_path}")

    df = load_results(input_path)
    metrics = discover_metrics(df)

    if not metrics:
        raise RuntimeError(
            "No usable performance metrics found."
        )

    print(f"Usable rows: {len(df)}")
    print(f"Detected metrics: {len(metrics)}")

    all_ranked_config: list[pd.DataFrame] = []
    all_ranked_family: list[pd.DataFrame] = []

    for metric in metrics:
        print(
            f"Analyzing {metric_label(metric)} "
            f"({METRIC_DIRECTION[metric]})..."
        )

        config_df, family_df = prepare_metric_data(
            df,
            metric,
        )

        ranked_config = rank_dataset_results(
            config_df,
            metric,
            entity_column="solver_name",
        )

        ranked_family = rank_dataset_results(
            family_df,
            metric,
            entity_column="solver_short",
        )

        if not ranked_config.empty:
            all_ranked_config.append(ranked_config)

        if not ranked_family.empty:
            all_ranked_family.append(ranked_family)

    if not all_ranked_family:
        raise RuntimeError(
            "Unable to build rankings."
        )

    ranked_config_all = pd.concat(
        all_ranked_config,
        ignore_index=True,
    )

    ranked_family_all = pd.concat(
        all_ranked_family,
        ignore_index=True,
    )

    config_leaderboard = build_leaderboard(
        ranked_config_all,
        entity_column="solver_name",
        entity_type="configuration",
    )

    family_leaderboard = build_leaderboard(
        ranked_family_all,
        entity_column="solver_short",
        entity_type="family",
    )

    combined_leaderboard = pd.concat(
        [
            family_leaderboard,
            config_leaderboard,
        ],
        ignore_index=True,
    )

    global_ranking = build_global_ranking(
        family_leaderboard
    )

    metric_summary = build_metric_summary(
        ranked_family_all,
        family_leaderboard,
    )

    wins_by_solver = (
        family_leaderboard.groupby(
            "solver",
            as_index=False,
        )
        .agg(
            n_tasks=("task", "nunique"),
            n_metrics=("metric", "nunique"),
            total_fractional_wins=("wins", "sum"),
            total_first_places=("n_first_places", "sum"),
            mean_rank_across_metrics=("mean_rank", "mean"),
            mean_coverage=("coverage", "mean"),
        )
        .sort_values(
            [
                "total_fractional_wins",
                "mean_rank_across_metrics",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    average_rank = (
        family_leaderboard[
            [
                "task",
                "metric",
                "metric_label",
                "solver",
                "n_datasets",
                "coverage",
                "mean_rank",
                "median_rank",
                "wins",
            ]
        ]
        .sort_values(
            [
                "task",
                "metric",
                "mean_rank",
            ]
        )
    )

    report = build_report(
        input_path=input_path,
        df=df,
        metrics=metrics,
        family_leaderboard=family_leaderboard,
        global_ranking=global_ranking,
    )

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    paths = {
        "family_leaderboard": (
            csv_dir / "leaderboard_by_metric.csv"
        ),
        "config_leaderboard": (
            csv_dir / "leaderboard_by_configuration.csv"
        ),
        "combined_leaderboard": (
            csv_dir / "leaderboard_all_entities.csv"
        ),
        "dataset_ranks_family": (
            csv_dir / "dataset_ranks_by_solver.csv"
        ),
        "dataset_ranks_config": (
            csv_dir / "dataset_ranks_by_configuration.csv"
        ),
        "wins": (
            csv_dir / "wins_by_solver.csv"
        ),
        "average_rank": (
            csv_dir / "average_rank.csv"
        ),
        "metric_summary": (
            csv_dir / "metric_summary.csv"
        ),
        "global_ranking": (
            csv_dir / "global_ranking.csv"
        ),
        "report": (
            reports_dir / "leaderboard_report.txt"
        ),
    }

    family_leaderboard.to_csv(
        paths["family_leaderboard"],
        index=False,
    )

    config_leaderboard.to_csv(
        paths["config_leaderboard"],
        index=False,
    )

    combined_leaderboard.to_csv(
        paths["combined_leaderboard"],
        index=False,
    )

    ranked_family_all.to_csv(
        paths["dataset_ranks_family"],
        index=False,
    )

    ranked_config_all.to_csv(
        paths["dataset_ranks_config"],
        index=False,
    )

    wins_by_solver.to_csv(
        paths["wins"],
        index=False,
    )

    average_rank.to_csv(
        paths["average_rank"],
        index=False,
    )

    metric_summary.to_csv(
        paths["metric_summary"],
        index=False,
    )

    global_ranking.to_csv(
        paths["global_ranking"],
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
