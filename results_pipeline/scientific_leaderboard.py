#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .leaderboard import METRIC_DIRECTION
else:
    from leaderboard import METRIC_DIRECTION


# ---------------------------------------------------------------------------
# Scientific metric policy
# ---------------------------------------------------------------------------

METRIC_POLICY = {
    # Classification
    "objective_accuracy": {
        "label": "Accuracy",
        "task": "classification",
        "use_for_task_ranking": True,
        "reason": "Primary classification metric.",
    },
    "objective_balanced_accuracy": {
        "label": "Balanced Accuracy",
        "task": "classification",
        "use_for_task_ranking": True,
        "reason": "Primary classification metric.",
    },
    "objective_f1_weighted": {
        "label": "Weighted F1",
        "task": "classification",
        "use_for_task_ranking": True,
        "reason": "Primary classification metric.",
    },

    # Anomaly detection
    "objective_auc_roc": {
        "label": "AUC-ROC",
        "task": "anomaly_detection",
        "use_for_task_ranking": True,
        "reason": "Primary anomaly-detection metric.",
    },
    "objective_auc_pr": {
        "label": "AUC-PR",
        "task": "anomaly_detection",
        "use_for_task_ranking": True,
        "reason": "Primary anomaly-detection metric.",
    },
    "objective_f1_pa": {
        "label": "F1-PA",
        "task": "anomaly_detection",
        "use_for_task_ranking": True,
        "reason": "Primary anomaly-detection metric.",
    },
    "objective_vus_roc": {
        "label": "VUS-ROC",
        "task": "anomaly_detection",
        "use_for_task_ranking": False,
        "reason": (
            "Diagnostic only: only three unique values for twelve "
            "evaluations and no observed solver discrimination."
        ),
    },
    "objective_vus_pr": {
        "label": "VUS-PR",
        "task": "anomaly_detection",
        "use_for_task_ranking": False,
        "reason": (
            "Diagnostic only: only three unique values for twelve "
            "evaluations and no observed solver discrimination."
        ),
    },

    # Forecasting
    "objective_mae": {
        "label": "MAE",
        "task": "forecasting",
        "use_for_task_ranking": True,
        "reason": "Primary point-forecast metric.",
    },
    "objective_mse": {
        "label": "MSE",
        "task": "forecasting",
        "use_for_task_ranking": True,
        "reason": "Primary squared-error metric.",
    },
    "objective_rmse": {
        "label": "RMSE",
        "task": "forecasting",
        "use_for_task_ranking": True,
        "reason": "Primary squared-error metric.",
    },
    "objective_mase": {
        "label": "MASE",
        "task": "forecasting",
        "use_for_task_ranking": True,
        "reason": "Scale-independent forecasting metric.",
    },
    "objective_smape": {
        "label": "SMAPE",
        "task": "forecasting",
        "use_for_task_ranking": True,
        "reason": "Scale-independent forecasting metric.",
    },
    "objective_crps": {
        "label": "CRPS",
        "task": "forecasting",
        "use_for_task_ranking": True,
        "reason": "Primary probabilistic forecasting metric.",
    },
    "objective_wql": {
        "label": "WQL",
        "task": "forecasting",
        "use_for_task_ranking": True,
        "reason": "Primary quantile forecasting metric.",
    },
    "objective_mcis": {
        "label": "MCIS",
        "task": "forecasting",
        "use_for_task_ranking": True,
        "reason": "Primary interval forecasting metric.",
    },
    "objective_pinball": {
        "label": "Pinball Loss",
        "task": "forecasting",
        "use_for_task_ranking": False,
        "reason": (
            "Excluded from aggregate ranking because it is perfectly "
            "correlated with CRPS and produces identical ranks."
        ),
    },
    "objective_skill_score_ratio": {
        "label": "Skill Score Ratio",
        "task": "forecasting",
        "use_for_task_ranking": True,
        "reason": "Relative forecasting skill metric.",
    },

    # Segmentation
    "objective_map_iou": {
        "label": "mAP-IoU",
        "task": "segmentation",
        "use_for_task_ranking": False,
        "reason": (
            "Only one valid constant observation; no solver comparison "
            "is possible."
        ),
    },
}


def short_name(value: object) -> str:
    if pd.isna(value):
        return "unknown"

    return str(value).split("[", maxsplit=1)[0].strip()


def build_policy_table() -> pd.DataFrame:
    rows = []

    for metric, policy in METRIC_POLICY.items():
        rows.append(
            {
                "metric": metric,
                "metric_label": policy["label"],
                "task": policy["task"],
                "use_for_task_ranking": policy[
                    "use_for_task_ranking"
                ],
                "reason": policy["reason"],
            }
        )

    return pd.DataFrame(rows)


def load_configuration_ranks(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "task",
        "dataset_name",
        "solver_name",
        "solver_short",
        "metric",
        "metric_label",
        "metric_value",
        "rank",
        "fractional_win",
    }

    missing = sorted(required.difference(df.columns))

    if missing:
        raise ValueError(
            "Missing columns: " + ", ".join(missing)
        )

    keys = ["task", "dataset_name", "solver_name", "metric"]
    if df[keys].isna().any().any() or df.duplicated(keys).any():
        raise ValueError("Missing or duplicate configuration identifiers")
    unknown = set(df["metric"]) - set(METRIC_DIRECTION)
    if unknown:
        raise ValueError(f"Unknown metric directions: {sorted(unknown)}")
    df["metric_value"] = pd.to_numeric(df["metric_value"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan,
    )
    df = df.loc[df["metric_value"].notna()].copy()
    if df.empty:
        raise ValueError("No finite metric values available for ranking")
    return df


def build_solver_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep multi-configuration baselines separate.

    Models with one configuration retain their short family name.
    SeasonalNaive configurations retain their complete configuration name.
    """
    result = df.copy()

    config_counts = (
        result.groupby("solver_short")["solver_name"]
        .nunique()
    )

    multi_config_families = set(
        config_counts.loc[config_counts > 1].index
    )

    result["solver_display"] = np.where(
        result["solver_short"].isin(multi_config_families),
        result["solver_name"],
        result["solver_short"],
    )

    result["is_multi_config_family"] = (
        result["solver_short"].isin(multi_config_families)
    )

    return result


def recompute_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute ranks after solver-display construction.

    Each SeasonalNaive configuration is treated as an independent method.
    """
    groups = []

    for (_, metric, dataset_name), group in df.groupby(
        ["task", "metric", "dataset_name"],
        sort=False,
    ):
        group = group.copy()

        ascending = METRIC_DIRECTION[metric] == "min"

        group["scientific_rank"] = group["metric_value"].rank(
            method="average",
            ascending=ascending,
        )

        best_rank = group["scientific_rank"].min()
        winner_mask = group["scientific_rank"].eq(best_rank)
        n_winners = int(winner_mask.sum())

        group["scientific_fractional_win"] = 0.0

        if n_winners:
            group.loc[
                winner_mask,
                "scientific_fractional_win",
            ] = 1.0 / n_winners

        group["scientific_is_winner"] = winner_mask
        group["n_competitors"] = (
            group["solver_display"].nunique()
        )

        groups.append(group)

    return pd.concat(groups, ignore_index=True)


def build_metric_leaderboard(
    ranked: pd.DataFrame,
) -> pd.DataFrame:
    leaderboard = (
        ranked.groupby(
            [
                "task",
                "metric",
                "metric_label",
                "solver_display",
            ],
            as_index=False,
        )
        .agg(
            n_datasets=("dataset_name", "nunique"),
            mean_rank=("scientific_rank", "mean"),
            median_rank=("scientific_rank", "median"),
            wins=("scientific_fractional_win", "sum"),
            first_places=("scientific_is_winner", "sum"),
            mean_value=("metric_value", "mean"),
            median_value=("metric_value", "median"),
        )
    )

    metric_totals = (
        ranked.groupby(["task", "metric"])["dataset_name"]
        .nunique()
        .rename("metric_total_datasets")
    )

    leaderboard = leaderboard.merge(
        metric_totals,
        on=["task", "metric"],
        how="left",
    )

    leaderboard["coverage"] = (
        leaderboard["n_datasets"]
        / leaderboard["metric_total_datasets"]
    )

    leaderboard = leaderboard.sort_values(
        [
            "task",
            "metric",
            "coverage",
            "mean_rank",
            "wins",
            "solver_display",
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

    leaderboard["metric_position"] = (
        leaderboard.groupby(["task", "metric"]).cumcount() + 1
    )

    return leaderboard


def build_task_ranking(
    metric_leaderboard: pd.DataFrame,
    policy: pd.DataFrame,
) -> pd.DataFrame:
    eligible = policy.loc[
        policy["use_for_task_ranking"],
        ["metric"],
    ]

    filtered = metric_leaderboard.merge(
        eligible,
        on="metric",
        how="inner",
    )

    task_metric_counts = (
        filtered.groupby("task")["metric"]
        .nunique()
        .rename("task_total_metrics")
    )

    ranking = (
        filtered.groupby(
            [
                "task",
                "solver_display",
            ],
            as_index=False,
        )
        .agg(
            n_metrics=("metric", "nunique"),
            total_dataset_evaluations=("n_datasets", "sum"),
            mean_metric_coverage=("coverage", "mean"),
            mean_rank_across_metrics=("mean_rank", "mean"),
            median_rank_across_metrics=("mean_rank", "median"),
            total_fractional_wins=("wins", "sum"),
            total_first_places=("first_places", "sum"),
        )
    )

    ranking = ranking.merge(
        task_metric_counts,
        on="task",
        how="left",
    )

    ranking["metric_coverage"] = (
        ranking["n_metrics"]
        / ranking["task_total_metrics"]
    )

    ranking = ranking.sort_values(
        [
            "task",
            "metric_coverage",
            "mean_metric_coverage",
            "mean_rank_across_metrics",
            "total_fractional_wins",
        ],
        ascending=[
            True,
            False,
            False,
            True,
            False,
        ],
    )

    ranking["task_position"] = (
        ranking.groupby("task").cumcount() + 1
    )

    return ranking[
        [
            "task",
            "task_position",
            "solver_display",
            "n_metrics",
            "task_total_metrics",
            "metric_coverage",
            "total_dataset_evaluations",
            "mean_metric_coverage",
            "mean_rank_across_metrics",
            "median_rank_across_metrics",
            "total_fractional_wins",
            "total_first_places",
        ]
    ]


def build_report(
    ranking: pd.DataFrame,
    policy: pd.DataFrame,
) -> str:
    lines = [
        "=" * 92,
        "                 TSFM BENCHMARK — SCIENTIFIC RANKING",
        "=" * 92,
        "",
        "POLICY",
        "-" * 92,
        "- Multiple SeasonalNaive configurations are kept separate.",
        "- No oracle selection of the best seasonal period for each dataset.",
        "- Pinball Loss is excluded from aggregation because it is redundant with CRPS.",
        "- VUS-ROC and VUS-PR remain diagnostic metrics.",
        "- mAP-IoU is excluded because no solver comparison is available.",
        "- No universal ranking across different tasks is produced.",
        "",
        "METRICS EXCLUDED FROM AGGREGATION",
        "-" * 92,
    ]

    excluded = policy.loc[
        ~policy["use_for_task_ranking"]
    ]

    for row in excluded.itertuples(index=False):
        lines.append(
            f"{row.metric_label:20} | {row.reason}"
        )

    for task, subset in ranking.groupby(
        "task",
        sort=False,
    ):
        lines.extend(
            [
                "",
                f"RANKING — {task.upper()}",
                "-" * 92,
            ]
        )

        for row in subset.itertuples(index=False):
            lines.append(
                f"{int(row.task_position):2d}. "
                f"{str(row.solver_display):32} | "
                f"mean rank={row.mean_rank_across_metrics:7.3f} | "
                f"wins={row.total_fractional_wins:8.2f} | "
                f"metrics={int(row.n_metrics)}/"
                f"{int(row.task_total_metrics)}"
            )

    lines.extend(["", "=" * 92])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "rank_file",
        nargs="?",
        default="csv/dataset_ranks_by_configuration.csv",
    )

    parser.add_argument(
        "--output-dir",
        default=".",
    )

    args = parser.parse_args()

    rank_path = Path(args.rank_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    csv_dir = output_dir / "csv"
    reports_dir = output_dir / "reports"

    csv_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    policy = build_policy_table()

    ranked = load_configuration_ranks(rank_path)
    ranked = build_solver_display(ranked)
    ranked = recompute_ranks(ranked)

    metric_leaderboard = build_metric_leaderboard(ranked)

    task_ranking = build_task_ranking(
        metric_leaderboard,
        policy,
    )

    excluded_metrics = policy.loc[
        ~policy["use_for_task_ranking"]
    ].copy()

    report = build_report(
        task_ranking,
        policy,
    )

    policy.to_csv(
        csv_dir / "scientific_metric_policy.csv",
        index=False,
    )

    ranked.to_csv(
        csv_dir / "scientific_dataset_ranks.csv",
        index=False,
    )

    metric_leaderboard.to_csv(
        csv_dir / "scientific_leaderboard_by_metric.csv",
        index=False,
    )

    task_ranking.to_csv(
        csv_dir / "scientific_ranking_by_task.csv",
        index=False,
    )

    excluded_metrics.to_csv(
        csv_dir / "excluded_metrics.csv",
        index=False,
    )

    report_path = (
        reports_dir / "scientific_leaderboard_report.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    print(report)
    print()
    print("Generated files:")
    print(f"  - {csv_dir / 'scientific_metric_policy.csv'}")
    print(f"  - {csv_dir / 'scientific_dataset_ranks.csv'}")
    print(f"  - {csv_dir / 'scientific_leaderboard_by_metric.csv'}")
    print(f"  - {csv_dir / 'scientific_ranking_by_task.csv'}")
    print(f"  - {csv_dir / 'excluded_metrics.csv'}")
    print(f"  - {report_path}")


if __name__ == "__main__":
    main()
