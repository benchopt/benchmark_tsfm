"""Pure per-task, per-metric descriptive statistics (no I/O or model loading)."""

from __future__ import annotations

import numpy as np
import pandas as pd


KEY = ["task", "dataset_id", "model_id", "metric"]
OUTPUT_COLUMNS = [
    "task", "metric", "direction", "model_id", "mean_value", "mean_rank",
    "std_value", "win_rate", "n_valid", "n_total", "coverage",
]


def validate_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Keep the supplied universe, reject ambiguous keys, normalize invalid values."""
    required = {*KEY, "value", "direction"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"Missing statistics columns: {sorted(missing)}")
    frame = observations.copy()
    if frame[KEY].isna().any().any():
        raise ValueError("Statistics identifiers must not be missing")
    if frame.duplicated(KEY).any():
        raise ValueError("Duplicate dataset/model/metric configuration; aggregate repetitions upstream")
    if not frame["direction"].isin(["min", "max"]).all():
        raise ValueError("Metric direction must be min or max")
    if (frame.groupby(["task", "metric"])["direction"].nunique() > 1).any():
        raise ValueError("Conflicting metric directions")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["value"] = frame["value"].replace([np.inf, -np.inf], np.nan)
    return frame


def rank_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Average ranks, exact ties, one fractional victory per dataset/metric."""
    frame = validate_observations(observations)
    frame = frame.loc[frame["value"].notna()].copy()
    frame["computed_rank"] = np.nan
    frame["computed_win"] = 0.0
    for _, group in frame.groupby(["task", "metric", "dataset_id"], sort=True):
        ranks = group["value"].rank(
            method="average", ascending=group["direction"].iloc[0] == "min",
        )
        winners = ranks.eq(ranks.min())
        frame.loc[group.index, "computed_rank"] = ranks
        frame.loc[group.index, "computed_win"] = winners.astype(float) / winners.sum()
    return frame


def aggregate_five_statistics(observations: pd.DataFrame) -> pd.DataFrame:
    """Use all supplied datasets of each task as its coverage denominator.

    Invalid values do not become defeats. Models without a valid observation
    for a metric remain visible with zero coverage and undefined statistics.
    Rates are fractions (0..1); display code converts them to percentages.
    """
    frame = validate_observations(observations).reset_index(drop=True)
    ranked = rank_observations(frame)
    # Reuse cached scientific ranks only after checking their conventions.
    for cached, computed in [
        ("scientific_rank", "computed_rank"),
        ("scientific_fractional_win", "computed_win"),
    ]:
        if cached in ranked:
            values = pd.to_numeric(ranked[cached], errors="coerce")
            if not np.allclose(values, ranked[computed], rtol=1e-12, atol=1e-12):
                raise ValueError(f"Inconsistent existing {cached}; refusing to change scientific policy")
            ranked[computed] = values

    rows = []
    for task, task_frame in frame.groupby("task", sort=True):
        n_total = task_frame["dataset_id"].nunique()
        for metric, metric_frame in task_frame.groupby("metric", sort=True):
            for model in sorted(task_frame["model_id"].unique()):
                subset = ranked.loc[
                    ranked["task"].eq(task) & ranked["metric"].eq(metric)
                    & ranked["model_id"].eq(model)
                ]
                n = len(subset)
                rows.append({
                    "task": task, "metric": metric,
                    "direction": metric_frame["direction"].iloc[0], "model_id": model,
                    "mean_value": subset["value"].mean(),
                    "mean_rank": subset["computed_rank"].mean(),
                    "std_value": subset["value"].std(ddof=0),
                    "win_rate": subset["computed_win"].mean(),
                    "n_valid": n, "n_total": n_total, "coverage": n / n_total,
                })
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
