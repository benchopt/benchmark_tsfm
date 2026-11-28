"""Leaderboards shared by the benchopt plots and the leaderboard page.

The leaderboard plots (``plots/*leaderboard.py``, ``plots/elo.py``) build their
tables with ``leaderboard_rows`` and the helpers above it. The leaderboard page
(``html/leaderboard.mako.html``) renders ``build_leaderboard(df)``: per task, a "board"
ranking the solvers by Elo, with their aggregated metrics, pairwise win rates
and per-dataset scores.
"""

import numpy as np
import pandas as pd

from benchmark_utils.elo import _elo_table, _find_anchor_idx, _pairwise_wins, _short
from benchmark_utils.metrics import is_higher_better

# key: (label, [(results column, label)], metrics relative to the baseline).
# The first metric ranks the solvers. Direction comes from is_higher_better.
TASKS = {
    "forecasting": (
        "Forecasting",
        [
            ("objective_wql", "rWQL"),
            ("objective_sql", "rSQL"),
            ("objective_mase", "rMASE"),
            ("objective_wape", "rWAPE"),
            ("objective_mae", "rMAE"),
            ("objective_mse", "rMSE"),
            ("objective_smape", "rSMAPE"),
        ],
        True,
    ),
    "classification": (
        "Classification",
        [
            ("objective_balanced_accuracy", "Bal. Acc"),
            ("objective_accuracy", "Accuracy"),
            ("objective_f1_weighted", "F1 weighted"),
        ],
        False,
    ),
    "anomaly_detection": (
        "Anomaly detection",
        [
            ("objective_auc_pr", "AUC-PR"),
            ("objective_auc_roc", "AUC-ROC"),
            ("objective_f1_pa", "F1-PA"),
        ],
        False,
    ),
}


def pivot(df, col):
    """Solvers × datasets mean of ``col`` over the repetitions."""
    return df.pivot_table(
        index="solver_name", columns="dataset_name", values=col, aggfunc="mean"
    )


def elo_cell(elo, ci_low, ci_high):
    """Table cell of an Elo with its 95% CI, e.g. ``**1032** (-41 / +38)``."""
    return f"**{elo:.0f}** ({ci_low - elo:+.0f} / {ci_high - elo:+.0f})"


def elo_scores(df, col):
    """Solvers × datasets ``col``, lower is better, or None without enough data."""
    if col not in df.columns:
        return None
    scores = pivot(df, col)

    # For higher-is-better metrics, negate so that _pairwise_wins
    # (which treats lower as better) computes wins correctly.
    if is_higher_better(col):
        scores = -scores

    # Strict clean: drop datasets where any solver is NaN (Elo needs
    # complete games), then drop all-tied datasets (variance == 0).
    scores = scores.loc[:, scores.notna().all(axis=0)]
    if scores.shape[1] > 0:
        scores = scores.loc[:, scores.var(axis=0, skipna=False) > 0]
    if scores.shape[0] < 2 or scores.shape[1] < 2:
        return None

    return scores


def mean_per_solver(df, col):
    """Mean of ``col`` per solver (ignoring NaN rows), keyed by short name."""
    rows = df[["solver_name", col]].dropna(subset=[col]).copy()
    rows["solver_name"] = rows["solver_name"].map(_short)
    return rows.groupby("solver_name")[col].mean().to_dict()


def relative_gmean(df, col):
    """Geometric mean of (solver / baseline) across datasets, by short name.

    For each dataset we divide every solver's ``col`` by the baseline's (Seasonal
    Naive, see ``ANCHOR_PREFERENCES``) on that same dataset, then take the
    geometric mean across datasets. Values < 1 = better than the baseline.
    """
    if col not in df.columns:
        return {}
    rows = df[["solver_name", "dataset_name", col]].dropna(subset=[col]).copy()
    rows["solver_name"] = rows["solver_name"].map(_short)
    _, baseline = _find_anchor_idx(rows["solver_name"].unique().tolist())
    if baseline is None:
        # Fall back to plain mean if no naive baseline present
        return rows.groupby("solver_name")[col].mean().to_dict()
    scores = rows.pivot_table(
        index="dataset_name", columns="solver_name", values=col, aggfunc="mean"
    )
    # Only keep datasets where baseline has a value
    scores = scores.dropna(subset=[baseline])
    rel = scores.div(scores[baseline], axis=0)
    # Geometric mean = exp(mean(log(rel))), skip non-positive values
    result = {}
    for solver in rel.columns:
        vals = rel[solver].dropna()
        vals = vals[vals > 0]
        if len(vals) > 0:
            result[solver] = float(np.exp(np.log(vals).mean()))
    return result


def leaderboard_rows(df, key):
    """Rows of the leaderboard plot of the task ``key`` of ``TASKS``.

    Rank | Solver | Elo | one cell per metric, sorted by Elo descending. Elo is
    fitted on the first metric; the metrics are the mean across datasets, or
    the geometric mean relative to the baseline for relative tasks.
    """
    label, metrics, relative = TASKS[key]
    elo_col = metrics[0][0]
    if not any(col in df.columns for col, _ in metrics):
        return [[f"(no {label.lower()} data)", *["—"] * (len(metrics) + 1)]]

    scores = elo_scores(df, elo_col)
    elo = {}
    if scores is not None:
        table, _ = _elo_table(scores)
        elo = {
            row.solver: (row.elo, row.ci_low, row.ci_high)
            for row in table.itertuples(index=False)
        }
    aggregate = relative_gmean if relative else mean_per_solver
    metric_maps = {
        col: (aggregate(df, col) if col in df.columns else {}) for col, _ in metrics
    }

    # All solvers seen in any metric
    all_solvers = set(elo).union(*metric_maps.values())

    # Sort by Elo descending, then the solvers without Elo by their metric
    def sort_key(s):
        if s in elo:
            return -elo[s][0]  # negate → descending
        if relative:  # first relative metric with values, ascending
            primary = next((c for c, _ in metrics if metric_maps[c]), None)
            return metric_maps[primary].get(s, np.inf) if primary else np.inf
        return metric_maps[elo_col].get(s, -np.inf)

    rows = []
    for rank, solver in enumerate(sorted(all_solvers, key=sort_key), start=1):
        cell = elo_cell(*elo[solver]) if solver in elo else "—"
        cells = [metric_maps[col].get(solver, "—") for col, _ in metrics]
        rows.append([str(rank), solver, cell] + cells)
    return rows


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
            pivot(df, col).reindex(index=scores.index, columns=datasets),
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
            {"label": lab, "col": col, "higher": is_higher_better(col)}
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


def _task_board(df, key):
    """``(board, scores, metric_of)`` of one task, board None without data."""
    label, metrics, relative = TASKS[key]
    primary, primary_label = metrics[0]
    if primary not in df or df[primary].isna().all():
        return None, None, None
    # Elo needs complete games: keep the datasets every solver ran on.
    scores = pivot(df, primary).dropna(axis=1, how="any")
    # Per dataset, the raw metric is shown: drop the "r" of "relative".
    raw_label = primary_label[1:] if relative else primary_label
    metric_of = {d: (primary, raw_label) for d in scores.columns}
    columns = [
        (lab, col, list(scores.columns), relative) for col, lab in metrics if col in df
    ]
    return _board(df, key, label, scores, metric_of, columns), scores, metric_of


def task_board(df, key):
    """Leaderboard of the task ``key`` of ``TASKS``, or None without data."""
    return _task_board(df, key)[0]


def build_boards(df):
    """One board per task with enough results, overall first if > 1 task."""
    boards, task_scores, metric_of, overall_columns = [], [], {}, []
    for key, (_, metrics, relative) in TASKS.items():
        board, scores, task_metric_of = _task_board(df, key)
        if board is None:
            continue
        boards.append(board)
        task_scores.append(scores)
        metric_of.update(task_metric_of)
        primary, primary_label = metrics[0]
        overall_columns.append((primary_label, primary, list(scores.columns), relative))

    if len(boards) > 1:
        # Overall: the solvers that ran every task, on all their datasets.
        scores = pd.concat(task_scores, axis=1)
        complete = scores.dropna(axis=0, how="any")
        overall = _board(df, "overall", "Overall", complete, metric_of, overall_columns)
        if overall is not None:
            boards.insert(0, overall)
    return boards


def build_leaderboard(df):
    """Leaderboard page data: the boards and a few stats, JSON serializable.

    ``df`` are benchopt results, ideally with the unique short names of
    ``benchopt.plotting.short_labels.shorten_names``.
    """
    return {
        "boards": build_boards(df),
        "n_models": int(df["solver_name"].nunique()),
        "n_datasets": int(df["dataset_name"].nunique()),
        "updated": str(df["run_date"].max())[:10] if "run_date" in df else None,
    }


def table_columns(elo_label, metrics):
    """Header of ``leaderboard_rows``, ``metrics`` being (column, label) pairs."""
    return ["Rank", "Solver", elo_label] + [
        f"{label} {'↑' if is_higher_better(col) else '↓'}" for col, label in metrics
    ]
