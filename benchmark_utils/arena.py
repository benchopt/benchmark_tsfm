"""Data of the arena page, rendered by ``html/arena.mako.html``.

``build_arena(df)`` turns benchopt results into one leaderboard ("board") per
task, plus an overall one mixing the tasks. Each board ranks the solvers by
Elo, as the leaderboard plots do, and comes with the pairwise win rates and
the per-dataset scores, so the page only has to draw them.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_utils.elo import _elo_table, _find_anchor_idx, _pairwise_wins
from benchmark_utils.metrics import is_higher_better

# ponytail: the metrics of each task are defined by the plots,
# which import each other the same way. Move them here if this ever hurts.
sys.path.insert(0, str(Path(__file__).parent.parent / "plots"))
import anomaly_detection_leaderboard  # noqa: E402
import classification_leaderboard  # noqa: E402
import forecasting_leaderboard  # noqa: E402

# key: (label, leaderboard defining the metrics, metrics relative to baseline)
TASKS = {
    "forecasting": ("Forecasting", forecasting_leaderboard, True),
    "classification": ("Classification", classification_leaderboard, False),
    "anomaly_detection": ("Anomaly detection", anomaly_detection_leaderboard, False),
}


def _pivot(df, col):
    """Solvers × datasets mean of ``col`` over the repetitions."""
    return df.pivot_table(
        index="solver_name", columns="dataset_name", values=col, aggfunc="mean"
    )


def _num(x, digits=4):
    return None if pd.isna(x) else round(float(x), digits)


def _aggregate(values, baseline):
    """Mean over the datasets, or geometric mean of the ratio to ``baseline``."""
    if baseline is None:
        return values.mean(axis=1)
    ratio = values / values.loc[baseline]
    return np.exp(np.log(ratio.where(ratio > 0)).mean(axis=1))


def _board(df, key, label, scores, metric_of, columns):
    """Leaderboard of the solvers in ``scores``, or None without enough data.

    Parameters
    ----------
    scores : DataFrame, solvers × datasets, without missing values
        Score of each solver on each dataset, in the dataset's own metric.
    metric_of : dict
        ``{dataset: (results column, label)}`` of the metric of each dataset.
    columns : list of (label, results column, datasets, relative)
        Metric columns of the leaderboard, each aggregated over ``datasets``.
    """
    # All-tied datasets carry no information on the ranking.
    scores = scores.loc[:, scores.var() > 0]
    n_solvers, n_datasets = scores.shape
    if n_solvers < 2 or n_datasets < 2:
        return None

    higher = [is_higher_better(metric_of[d][0]) for d in scores.columns]
    loss = scores * np.where(higher, -1, 1)  # lower is better everywhere

    table, _ = _elo_table(loss)
    table = table.set_index("name")
    order = list(table.index)  # best Elo first
    wins = pd.DataFrame(
        _pairwise_wins(loss.to_numpy()) / n_datasets,
        index=loss.index,
        columns=loss.index,
    )
    ranks = loss.rank(axis=0)
    anchor_idx, _ = _find_anchor_idx(list(scores.index))
    baseline = None if anchor_idx is None else scores.index[anchor_idx]
    time = (
        df[df["dataset_name"].isin(scores.columns)]
        .groupby("solver_name")["time"]
        .median()
    )
    metrics = [
        _aggregate(
            _pivot(df, col).reindex(index=scores.index, columns=datasets),
            baseline if relative else None,
        )
        for _, col, datasets, relative in columns
    ]

    return {
        "key": key,
        "label": label,
        "anchor": baseline,
        "datasets": [
            {"name": d, "metric": metric_of[d][1], "higher": h}
            for d, h in zip(scores.columns, higher)
        ],
        "columns": [
            {"label": lab, "higher": is_higher_better(col)}
            for lab, col, _, _ in columns
        ],
        "models": [
            {
                "name": s,
                "elo": _num(table.loc[s, "elo"], 1),
                "lo": _num(table.loc[s, "ci_low"], 1),
                "hi": _num(table.loc[s, "ci_high"], 1),
                "win": _num(wins.loc[s].sum() / (n_solvers - 1)),
                "rank": _num(ranks.loc[s].mean(), 2),
                "time": _num(time.get(s)),
                "metrics": [_num(m[s]) for m in metrics],
                "scores": [_num(v) for v in scores.loc[s]],
                "ranks": [_num(v, 1) for v in ranks.loc[s]],
                "wins": [_num(v, 3) for v in wins.loc[s, order]],
            }
            for s in order
        ],
    }


def build_arena(df):
    """Arena page data: one board per task with enough results, overall first.

    Parameters
    ----------
    df : DataFrame
        Benchopt results, ideally with the unique short names of
        ``benchopt.plotting.short_labels.shorten_names``.

    Returns
    -------
    arena : dict, JSON serializable
    """
    boards, task_scores, metric_of, overall_columns = [], [], {}, []
    for key, (label, module, relative) in TASKS.items():
        primary = module._ELO_COL
        if primary not in df or df[primary].isna().all():
            continue
        # Elo needs complete games: keep the datasets every solver ran on.
        scores = _pivot(df, primary).dropna(axis=1, how="any")
        primary_label = dict(module._METRICS)[primary]
        # Per dataset, the raw metric is shown: drop the "r" of "relative".
        raw_label = primary_label[1:] if relative else primary_label
        metric_of.update({d: (primary, raw_label) for d in scores.columns})
        board = _board(
            df,
            key,
            label,
            scores,
            metric_of,
            [
                (lab, col, list(scores.columns), relative)
                for col, lab in module._METRICS
                if col in df
            ],
        )
        if board is None:
            continue
        boards.append(board)
        task_scores.append(scores)
        overall_columns.append((primary_label, primary, list(scores.columns), relative))

    if len(boards) > 1:
        # Overall: the solvers that ran every task, on all their datasets.
        scores = pd.concat(task_scores, axis=1)
        complete = scores.dropna(axis=0, how="any")
        overall = _board(df, "overall", "Overall", complete, metric_of, overall_columns)
        if overall is not None:
            boards.insert(0, overall)

    return {
        "boards": boards,
        "n_models": int(df["solver_name"].nunique()),
        "n_datasets": int(df["dataset_name"].nunique()),
        "updated": str(df["run_date"].max())[:10] if "run_date" in df else None,
    }
