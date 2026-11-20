#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .global_statistics import load_statistics, write_statistics
    from .leaderboard import METRIC_DIRECTION, METRIC_LABELS
else:
    from global_statistics import load_statistics, write_statistics
    from results_pipeline.leaderboard import METRIC_DIRECTION, METRIC_LABELS


# ---------------------------------------------------------------------------
# Metric directions
#
# "max": higher values are better.
# "min": lower values are better.
# ---------------------------------------------------------------------------

METRIC_DIRECTIONS = METRIC_DIRECTION


def safe_name(value: object, max_length: int = 100) -> str:
    """
    Convert a label into a safe file or directory name.
    """
    text = str(value).strip().lower()

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    )

    text = text.strip("_")

    if not text:
        text = "unknown"

    return text[:max_length]


def format_value(value: object, digits: int = 6) -> str:
    """
    Format numerical values for HTML display.
    """
    if pd.isna(value):
        return "—"

    number = float(value)

    if not np.isfinite(number):
        return "—"

    absolute = abs(number)

    if absolute != 0 and (
        absolute >= 1_000_000
        or absolute < 0.0001
    ):
        return f"{number:.6e}"

    return f"{number:.{digits}f}"


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(
        metric,
        metric.removeprefix("objective_")
        .replace("_", " ")
        .title(),
    )


def metric_direction(metric: str) -> str:
    if metric not in METRIC_DIRECTIONS:
        raise ValueError(
            f"Unknown metric direction: {metric}"
        )

    return METRIC_DIRECTIONS[metric]


def direction_symbol(metric: str) -> str:
    return "↑" if metric_direction(metric) == "max" else "↓"


def load_results(path: Path) -> pd.DataFrame:
    """
    Load scientific_dataset_ranks.csv.

    Detailed rankings use the observed metric values.
    Previously computed overall ranks are not used for these rankings.
    """
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
        "metric_value",
    }

    missing = sorted(
        required.difference(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    keys = ["task", "dataset_name", "solver_name", "metric"]
    if df[keys].isna().any().any() or df.duplicated(keys).any():
        raise ValueError("Missing or duplicate configuration identifiers")
    df = df.copy()

    df["metric_value"] = pd.to_numeric(
        df["metric_value"],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)

    if "metric_label" not in df.columns:
        df["metric_label"] = df["metric"].map(
            metric_label
        )

    if "solver_display" not in df.columns:
        df["solver_display"] = df["solver_name"]

    unknown_metrics = sorted(
        set(df["metric"].dropna())
        .difference(METRIC_DIRECTIONS)
    )

    if unknown_metrics:
        raise ValueError(
            "Add directions for these metrics to "
            "METRIC_DIRECTIONS: "
            + ", ".join(unknown_metrics)
        )

    df = df.loc[
        df["metric_value"].notna()
    ].copy()

    return df.reset_index(drop=True)


def build_raw_results(
    dataset_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a solver-by-metric table of observed values.
    """
    values = dataset_df.pivot_table(
        index=[
            "solver_display",
            "solver_short",
        ],
        columns="metric",
        values="metric_value",
        aggfunc="first",
    ).reset_index()

    metric_order = (
        dataset_df[
            [
                "metric",
                "metric_label",
            ]
        ]
        .drop_duplicates()
        .sort_values("metric_label")
    )

    renamed_columns = {}

    ordered_metric_columns = []

    for row in metric_order.itertuples(index=False):
        metric = row.metric
        label = row.metric_label
        symbol = direction_symbol(metric)

        renamed_columns[metric] = f"{label} {symbol}"
        ordered_metric_columns.append(metric)

    values = values[
        [
            "solver_display",
            "solver_short",
            *ordered_metric_columns,
        ]
    ]

    return values.rename(
        columns=renamed_columns
    )


def build_metric_ranking(
    dataset_df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """
    Rank solvers for a single metric.
    """
    metric_df = dataset_df.loc[
        dataset_df["metric"].eq(metric),
        [
            "solver_display",
            "solver_short",
            "metric_value",
        ],
    ].copy()

    metric_df["metric_value"] = pd.to_numeric(
        metric_df["metric_value"], errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    metric_df = metric_df.loc[metric_df["metric_value"].notna()].copy()

    direction = metric_direction(metric)

    ascending = direction == "min"

    metric_df["position"] = (
        metric_df["metric_value"]
        .rank(
            method="min",
            ascending=ascending,
        )
        .astype(int)
    )

    best_value = (
        metric_df["metric_value"].min()
        if ascending
        else metric_df["metric_value"].max()
    )

    metric_df["is_winner"] = metric_df["metric_value"].eq(best_value)

    metric_df = metric_df.sort_values(
        [
            "position",
            "solver_display",
        ],
        ascending=[
            True,
            True,
        ],
    ).reset_index(drop=True)

    metric_df.insert(
        2,
        "metric",
        metric,
    )

    metric_df.insert(
        3,
        "metric_label",
        metric_label(metric),
    )

    metric_df.insert(
        4,
        "direction",
        direction_symbol(metric),
    )

    return metric_df[
        [
            "position",
            "solver_display",
            "solver_short",
            "metric",
            "metric_label",
            "direction",
            "metric_value",
            "is_winner",
        ]
    ]


def html_table(
    df: pd.DataFrame,
    numeric_columns: set[str] | None = None,
    winner_column: str | None = None,
) -> str:
    """
    Build an HTML table highlighting the winners.
    """
    if df.empty:
        return "<p>No data available.</p>"

    numeric_columns = numeric_columns or set()

    headers = "".join(
        f"<th>{html.escape(str(column))}</th>"
        for column in df.columns
        if column != winner_column
    )

    rows = []

    for record in df.to_dict(orient="records"):
        winner = (
            bool(record.get(winner_column, False))
            if winner_column
            else False
        )

        css_class = (
            ' class="winner-row"'
            if winner
            else ""
        )

        cells = []

        for column in df.columns:
            if column == winner_column:
                continue

            value = record[column]

            if column in numeric_columns:
                shown = format_value(value)
            elif pd.isna(value):
                shown = "—"
            else:
                shown = html.escape(str(value))

            cells.append(f"<td>{shown}</td>")

        rows.append(
            f"<tr{css_class}>"
            + "".join(cells)
            + "</tr>"
        )

    return f"""
    <table class="result-table">
        <thead>
            <tr>{headers}</tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def raw_results_html(
    raw_results: pd.DataFrame,
) -> str:
    """
    Display observed values in a solver-by-metric table.
    """
    if raw_results.empty:
        return "<p>No data available.</p>"

    numeric_columns = set(
        raw_results.columns
    ).difference(
        {
            "solver_display",
            "solver_short",
        }
    )

    return html_table(
        raw_results,
        numeric_columns=numeric_columns,
    )


def build_dataset_report(
    task: str,
    dataset_name: str,
    dataset_df: pd.DataFrame,
    raw_results: pd.DataFrame,
    rankings: dict[str, pd.DataFrame],
    task_index_relative_path: str,
) -> str:
    """
    Generate the complete HTML page for a dataset.
    """
    n_solvers = dataset_df["solver_display"].nunique()
    n_families = dataset_df["solver_short"].nunique()
    n_metrics = dataset_df["metric"].nunique()
    n_values = len(dataset_df)

    metric_buttons = []
    ranking_sections = []

    for metric in sorted(
        rankings,
        key=lambda item: metric_label(item),
    ):
        ranking = rankings[metric]
        label = metric_label(metric)
        symbol = direction_symbol(metric)

        section_id = safe_name(metric)

        metric_buttons.append(
            f"""
            <a class="metric-link" href="#{section_id}">
                {html.escape(label)} {symbol}
            </a>
            """
        )

        ranking_display = ranking.rename(
            columns={
                "position": "Position",
                "solver_display": "Configuration",
                "solver_short": "Famille",
                "metric_value": f"{label} {symbol}",
            }
        )[
            [
                "Position",
                "Configuration",
                "Famille",
                f"{label} {symbol}",
                "is_winner",
            ]
        ]

        direction_explanation = (
            "Higher values are better."
            if metric_direction(metric) == "max"
            else "Lower values are better."
        )

        ranking_sections.append(
            f"""
            <section class="card metric-ranking" id="{section_id}">
                <div class="section-heading">
                    <div>
                        <h2>Ranking by {html.escape(label)}</h2>
                        <p>
                            {html.escape(direction_explanation)}
                            Tied values share the same position.
                        </p>
                    </div>

                    <span class="direction-badge">
                        {symbol}
                    </span>
                </div>

                {html_table(
                    ranking_display,
                    numeric_columns={f"{label} {symbol}"},
                    winner_column="is_winner",
                )}
            </section>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>
    TSFM Benchmark — {html.escape(task)} —
    {html.escape(dataset_name)}
</title>

<style>
    :root {{
        --background: #f3f5f8;
        --card: #ffffff;
        --text: #172033;
        --muted: #667085;
        --border: #dbe1ea;
        --primary: #243d80;
        --primary-dark: #17295f;
        --primary-light: #e9eeff;
        --winner: #e5f7ed;
        --winner-border: #38a169;
        --warning: #fff6d8;
    }}

    * {{
        box-sizing: border-box;
    }}

    html {{
        scroll-behavior: smooth;
    }}

    body {{
        margin: 0;
        background: var(--background);
        color: var(--text);
        font-family:
            Inter, Arial, Helvetica, sans-serif;
        line-height: 1.5;
    }}

    .container {{
        max-width: 1500px;
        margin: 0 auto;
        padding: 28px;
    }}

    .back-link {{
        display: inline-block;
        margin-bottom: 18px;
        color: var(--primary);
        text-decoration: none;
        font-weight: 700;
    }}

    .hero {{
        padding: 32px;
        margin-bottom: 22px;
        color: white;
        background:
            linear-gradient(
                135deg,
                var(--primary-dark),
                #3857ae
            );
        border-radius: 17px;
    }}

    .hero .task {{
        margin: 0 0 8px;
        opacity: 0.82;
        font-size: 15px;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }}

    .hero h1 {{
        margin: 0 0 12px;
        font-size: clamp(24px, 4vw, 36px);
        overflow-wrap: anywhere;
    }}

    .hero p {{
        margin: 0;
        opacity: 0.9;
    }}

    .stats {{
        display: grid;
        grid-template-columns:
            repeat(auto-fit, minmax(190px, 1fr));
        gap: 15px;
        margin-bottom: 22px;
    }}

    .stat {{
        padding: 19px;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
    }}

    .stat strong {{
        display: block;
        margin-bottom: 4px;
        color: var(--primary);
        font-size: 27px;
    }}

    .stat span {{
        color: var(--muted);
        font-size: 14px;
    }}

    .card {{
        padding: 24px;
        margin-bottom: 22px;
        overflow-x: auto;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
    }}

    .card h2 {{
        margin-top: 0;
        margin-bottom: 6px;
    }}

    .card p {{
        margin-top: 0;
        color: var(--muted);
    }}

    .metric-navigation {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }}

    .metric-link {{
        padding: 9px 13px;
        color: var(--primary);
        background: var(--primary-light);
        border-radius: 999px;
        text-decoration: none;
        font-weight: 700;
        font-size: 14px;
    }}

    .metric-link:hover {{
        background: #dce5ff;
    }}

    .section-heading {{
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 15px;
    }}

    .direction-badge {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 46px;
        min-height: 46px;
        color: white;
        background: var(--primary);
        border-radius: 50%;
        font-size: 24px;
        font-weight: 800;
    }}

    .result-table {{
        width: 100%;
        border-collapse: collapse;
        white-space: nowrap;
        font-size: 14px;
    }}

    .result-table th {{
        padding: 12px;
        color: white;
        background: var(--primary);
        text-align: left;
    }}

    .result-table td {{
        padding: 11px 12px;
        border-bottom: 1px solid var(--border);
    }}

    .result-table tbody tr:hover {{
        background: #f7f9ff;
    }}

    .winner-row {{
        background: var(--winner);
        font-weight: 700;
        border-left: 5px solid var(--winner-border);
    }}

    .note {{
        padding: 15px 17px;
        margin-top: 16px;
        background: var(--warning);
        border-left: 4px solid #dba500;
        border-radius: 7px;
    }}

    footer {{
        padding: 18px;
        text-align: center;
        color: var(--muted);
    }}

    @media (max-width: 700px) {{
        .container {{
            padding: 14px;
        }}

        .hero {{
            padding: 23px;
        }}

        .card {{
            padding: 17px;
        }}
    }}
</style>
</head>

<body>
<div class="container">

    <a class="back-link" href="{html.escape(task_index_relative_path)}">
        ← Back to datasets for this task
    </a>

    <header class="hero">
        <p class="task">
            Task: {html.escape(task.replace("_", " ").title())}
        </p>

        <h1>{html.escape(dataset_name)}</h1>

        <p>
            Observed metric values and independent rankings
            for each metric.
        </p>
    </header>

    <section class="stats">
        <div class="stat">
            <strong>{n_solvers}</strong>
            <span>Solver configurations</span>
        </div>

        <div class="stat">
            <strong>{n_families}</strong>
            <span>Solver families</span>
        </div>

        <div class="stat">
            <strong>{n_metrics}</strong>
            <span>Available metrics</span>
        </div>

        <div class="stat">
            <strong>{n_values}</strong>
            <span>Observed metric values</span>
        </div>
    </section>

    <section class="card">
        <h2>Jump to a ranking</h2>

        <p>
            Select a metric to view its ranking.
        </p>

        <div class="metric-navigation">
            {''.join(metric_buttons)}
        </div>
    </section>

    <section class="card">
        <h2>Observed results for this dataset</h2>

        <p>
            Each row represents a solver configuration.
            Arrows indicate whether higher or lower values are better.
        </p>

        {raw_results_html(raw_results)}

        <div class="note">
            <strong>How to read this table:</strong>
            ↑ means higher values are better.
            ↓ means lower values are better.
            Rankings are computed separately for each metric.
        </div>
    </section>

    {''.join(ranking_sections)}

    <footer>
        TSFM Benchmark — automatically generated report.
    </footer>

</div>
</body>
</html>
"""


def build_global_summary_section(summary: pd.DataFrame) -> str:
    """Render precomputed statistics; JS only selects a metric, never aggregates."""
    if summary.empty:
        return ""
    options = []
    panels = []
    for index, (metric, group) in enumerate(summary.groupby("metric", sort=True)):
        label = str(group["metric_label"].iloc[0])
        arrow = "↓" if group["direction"].iloc[0] == "min" else "↑"
        options.append(f'<option value="{index}">{html.escape(label)} {arrow}</option>')
        rows = []
        ordered = group.copy()
        ordered["position"] = ordered["mean_rank"].rank(method="min", ascending=True)
        ordered = ordered.sort_values(
            ["mean_rank", "solver_display", "solver_name"], na_position="last",
        )
        for row in ordered.itertuples():
            position = "—" if pd.isna(row.position) else str(int(row.position))
            rate = "—" if pd.isna(row.win_rate) else f"{100 * row.win_rate:.2f} %"
            rows.append(
                f'<tr><td>{position}</td><th scope="row" title="{html.escape(str(row.solver_name), quote=True)}">'
                f'{html.escape(str(row.solver_display))}</th>'
                f'<td>{format_value(row.mean_value)}</td>'
                f'<td>{format_value(row.mean_rank)}</td>'
                f'<td>{format_value(row.std_value)}</td><td>{rate}</td>'
                f'<td>{int(row.n_valid)}/{int(row.n_total)} '
                f'({100 * row.coverage:.2f} %)</td></tr>'
            )
        panels.append(
            f'<div class="global-metric-panel" data-metric-index="{index}">'
            f'<h3>{html.escape(label)} {arrow}</h3><div class="global-table-scroll">'
            '<table><thead><tr><th scope="col">Position</th><th scope="col">Model / configuration</th>'
            '<th scope="col">Mean value</th><th scope="col">Mean rank</th>'
            '<th scope="col">Metric standard deviation</th>'
            '<th scope="col">Shared win rate</th>'
            '<th scope="col">Coverage</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div></div>'
        )
    return """
<section id="global-performance-summary" class="global-summary">
    <h2>Global Performance Summary</h2>
    <p>Five statistics per metric across the configurations of this task.
    Each dataset configuration counts once; a dataset with multiple configurations
    therefore contributes multiple observations. Model configurations remain separate.</p>
    <p><strong>Ranking:</strong> models are sorted by ascending mean rank for the
    selected metric (lower is better, including for Accuracy).
    Equal mean ranks share a position (1, 1, 3).
    Check coverage as well: models may have been evaluated on different sets of
    configurations. Each ranking uses a single metric.</p>
    <label for="global-metric-select">Metric</label>
    <select id="global-metric-select">""" + "".join(options) + """</select>
    """ + "".join(panels) + """
    <p><strong>What does the shared win rate mean?</strong>
    Each configuration awards one win: a sole winner receives 1 point;
    two tied winners receive 1/2 each; five receive 1/5 each. Other models receive 0.
    <strong>Rate = sum of points / number of valid configurations for the model × 100.</strong>
    For example, one sole win and one win shared by five models over two configurations
    give (1 + 0.2) / 2 × 100 = 60%.
    This differs from the percentage of first places with every tie counted as a full win.</p>
    <details><summary>Calculation conventions and limitations</summary>
    <p>Arithmetic mean and population standard deviation (ddof=0) of raw metric values,
    after repeated measurements have been aggregated upstream. Datasets can have
    different scales, which may dominate the mean and standard deviation.
    Standard deviation is zero for a single observation; this does not establish stability.</p>
    <p>Scientific ranks use average ranks for exact ties (1.5, 1.5, 3),
    whereas detailed tables use competition positions (1, 1, 3).
    Tied winners share one win. The win rate uses only valid configurations for the model.
    Missing results are neither zeros nor defeats. Because coverage may differ,
    win rates do not necessarily sum to 100%.</p>
    <p>Coverage: valid results for this model and metric / all configurations of the task
    present in the source CSV, including configurations without a value for this metric.
    This is not coverage of all planned experiments.
    Statistics are undefined (—) when no observations are available.
    A model evaluated alone receives rank 1 and a win, without an actual comparison.</p>
    </details>
</section>
<style>
    .global-summary { background: white; border: 1px solid #dbe1ea;
        border-radius: 16px; padding: 24px; margin: 24px 0; }
    .global-summary p { color: #52627a; line-height: 1.6; }
    .global-summary select { padding: 10px; margin: 8px; max-width: 100%; }
    .global-table-scroll { overflow-x: auto; }
    .global-summary table { width: 100%; border-collapse: collapse; margin: 16px 0; }
    .global-summary th, .global-summary td { padding: 12px; text-align: left;
        border-bottom: 1px solid #dbe1ea; font-variant-numeric: tabular-nums; }
    .global-summary thead { background: #243d80; color: white; }
    .global-summary tbody tr:nth-child(even) { background: #f3f5f8; }
</style>
<script>
(() => {
    const section = document.getElementById('global-performance-summary');
    const select = section.querySelector('#global-metric-select');
    const update = () => section.querySelectorAll('.global-metric-panel').forEach(
        panel => { panel.hidden = panel.dataset.metricIndex !== select.value; }
    );
    select.addEventListener('change', update);
    update();
})();
</script>
"""


def build_task_index(
    task: str,
    dataset_cards: list[dict[str, object]],
    root_index_relative_path: str,
    global_summary_html: str = "",
) -> str:
    """
    Generate the page listing all datasets for a task.
    """
    cards_html = []

    for card in dataset_cards:
        cards_html.append(
            f"""
            <a class="dataset-card"
               href="{html.escape(str(card['relative_report']))}">
                <div class="dataset-title">
                    {html.escape(str(card['dataset_name']))}
                </div>

                <div class="metadata">
                    <span>{int(card['n_solvers'])} solvers</span>
                    <span>{int(card['n_metrics'])} metrics</span>
                </div>

                <div class="open-report">
                    View detailed report →
                </div>
            </a>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>
    TSFM Benchmark —
    {html.escape(task.replace("_", " ").title())}
</title>

<style>
    :root {{
        --background: #f3f5f8;
        --card: #ffffff;
        --text: #172033;
        --muted: #667085;
        --border: #dbe1ea;
        --primary: #243d80;
        --primary-light: #e9eeff;
    }}

    * {{
        box-sizing: border-box;
    }}

    body {{
        margin: 0;
        background: var(--background);
        color: var(--text);
        font-family:
            Inter, Arial, Helvetica, sans-serif;
    }}

    main {{
        max-width: 1300px;
        margin: auto;
        padding: 32px;
    }}

    .back-link {{
        display: inline-block;
        margin-bottom: 20px;
        color: var(--primary);
        text-decoration: none;
        font-weight: 700;
    }}

    .header {{
        padding: 30px;
        margin-bottom: 24px;
        color: white;
        background:
            linear-gradient(
                135deg,
                #17295f,
                #3857ae
            );
        border-radius: 16px;
    }}

    .header h1 {{
        margin: 0 0 8px;
        font-size: 34px;
    }}

    .header p {{
        margin: 0;
        opacity: 0.88;
    }}

    .dataset-grid {{
        display: grid;
        grid-template-columns:
            repeat(auto-fit, minmax(300px, 1fr));
        gap: 18px;
    }}

    .dataset-card {{
        display: flex;
        flex-direction: column;
        min-height: 175px;
        padding: 22px;
        color: var(--text);
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        text-decoration: none;
        transition:
            transform 0.15s ease,
            box-shadow 0.15s ease;
    }}

    .dataset-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 12px 32px rgba(28, 44, 90, 0.13);
    }}

    .dataset-title {{
        margin-bottom: 18px;
        color: var(--primary);
        font-size: 18px;
        font-weight: 800;
        overflow-wrap: anywhere;
    }}

    .metadata {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 20px;
    }}

    .metadata span {{
        padding: 6px 10px;
        color: var(--primary);
        background: var(--primary-light);
        border-radius: 999px;
        font-size: 13px;
        font-weight: 700;
    }}

    .open-report {{
        margin-top: auto;
        color: var(--muted);
        font-weight: 700;
    }}
</style>
</head>

<body>
<main>

    <a class="back-link" href="{html.escape(root_index_relative_path)}">
        ← Back to tasks
    </a>

    <header class="header">
        <h1>
            {html.escape(task.replace("_", " ").title())}
        </h1>

        <p>
            {len(dataset_cards)} dataset(s) available.
            Select a dataset to view its metric values and per-metric rankings.
        </p>
    </header>

    {global_summary_html}

    <section class="dataset-grid">
        {''.join(cards_html)}
    </section>

</main>
</body>
</html>
"""


def build_root_index(
    task_cards: list[dict[str, object]],
) -> str:
    """
    Generate the home page listing all tasks.
    """
    cards_html = []

    for card in task_cards:
        cards_html.append(
            f"""
            <a class="task-card"
               href="{html.escape(str(card['relative_index']))}">
                <strong>
                    {html.escape(str(card['task_label']))}
                </strong>

                <span>
                    {int(card['n_datasets'])} dataset(s)
                </span>

                <small>
                    Explore task →
                </small>
            </a>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>TSFM Benchmark — Dataset Explorer</title>

<style>
    :root {{
        --background: #f3f5f8;
        --card: #ffffff;
        --text: #172033;
        --muted: #667085;
        --border: #dbe1ea;
        --primary: #243d80;
    }}

    * {{
        box-sizing: border-box;
    }}

    body {{
        margin: 0;
        background: var(--background);
        color: var(--text);
        font-family:
            Inter, Arial, Helvetica, sans-serif;
    }}

    main {{
        max-width: 1150px;
        margin: auto;
        padding: 40px 28px;
    }}

    .hero {{
        padding: 34px;
        margin-bottom: 28px;
        color: white;
        background:
            linear-gradient(
                135deg,
                #17295f,
                #3857ae
            );
        border-radius: 17px;
    }}

    .hero h1 {{
        margin: 0 0 10px;
        font-size: 36px;
    }}

    .hero p {{
        margin: 0;
        opacity: 0.9;
    }}

    .task-grid {{
        display: grid;
        grid-template-columns:
            repeat(auto-fit, minmax(250px, 1fr));
        gap: 18px;
    }}

    .task-card {{
        display: flex;
        flex-direction: column;
        gap: 12px;
        padding: 25px;
        color: var(--text);
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        text-decoration: none;
        transition:
            transform 0.15s ease,
            box-shadow 0.15s ease;
    }}

    .task-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(28, 44, 90, 0.13);
    }}

    .task-card strong {{
        color: var(--primary);
        font-size: 21px;
    }}

    .task-card span {{
        color: var(--muted);
    }}

    .task-card small {{
        margin-top: 10px;
        font-weight: 700;
    }}
</style>
</head>

<body>
<main>

    <header class="hero">
        <h1>TSFM Benchmark — Dataset Explorer</h1>

        <p>
            Select a task, then a dataset, to view metric values and per-metric rankings.
        </p>
    </header>

    <section class="task-grid">
        {''.join(cards_html)}
    </section>

</main>
</body>
</html>
"""


def process_dataset(
    task: str,
    dataset_name: str,
    dataset_df: pd.DataFrame,
    dataset_directory: Path,
) -> dict[str, object]:
    """
    Generate all output files for a dataset.
    """
    dataset_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_results = build_raw_results(dataset_df)

    raw_results_path = (
        dataset_directory / "raw_results.csv"
    )

    raw_results.to_csv(
        raw_results_path,
        index=False,
    )

    rankings: dict[str, pd.DataFrame] = {}

    for metric in sorted(
        dataset_df["metric"].unique()
    ):
        ranking = build_metric_ranking(
            dataset_df,
            metric,
        )

        rankings[metric] = ranking

        ranking_filename = (
            "ranking_"
            + safe_name(metric.removeprefix("objective_"))
            + ".csv"
        )

        ranking.to_csv(
            dataset_directory / ranking_filename,
            index=False,
        )

    report = build_dataset_report(
        task=task,
        dataset_name=dataset_name,
        dataset_df=dataset_df,
        raw_results=raw_results,
        rankings=rankings,
        task_index_relative_path="../index.html",
    )

    report_path = (
        dataset_directory / "report.html"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    return {
        "dataset_name": dataset_name,
        "report_path": report_path,
        "n_solvers": dataset_df["solver_display"].nunique(),
        "n_metrics": dataset_df["metric"].nunique(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an HTML explorer by task and dataset, "
            "with observed values and independent rankings "
            "for each metric."
        )
    )

    parser.add_argument(
        "input_file",
        nargs="?",
        default="csv/scientific_dataset_ranks.csv",
        help=(
            "CSV file containing task, dataset_name, solver_name, "
            "metric and metric_value."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="reports_by_dataset",
        help="Output directory.",
    )

    parser.add_argument(
        "--global-summary", type=Path,
        help="Precomputed CSV with its .metadata.json file; otherwise computed once here.",
    )
    args = parser.parse_args()

    input_path = Path(
        args.input_file
    ).expanduser().resolve()

    output_root = Path(
        args.output_dir
    ).expanduser().resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Read/check the global artifact once, outside the per-dataset loop.
    if args.global_summary is not None:
        global_statistics = load_statistics(input_path, args.global_summary)
    else:
        global_statistics = write_statistics(
            input_path, output_root / "dataset_global_statistics.csv",
        )
    df = load_results(input_path)

    task_cards = []

    for task, task_df in df.groupby(
        "task",
        sort=True,
    ):
        task = str(task)

        print(f"\nTask: {task}")

        task_directory = (
            output_root / safe_name(task)
        )

        task_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataset_cards = []

        used_directory_names: set[str] = set()

        for dataset_index, (
            dataset_name,
            dataset_df,
        ) in enumerate(
            task_df.groupby(
                "dataset_name",
                sort=True,
            ),
            start=1,
        ):
            dataset_name = str(dataset_name)

            base_directory_name = (
                f"{dataset_index:03d}_"
                f"{safe_name(dataset_name, max_length=70)}"
            )

            directory_name = base_directory_name
            suffix = 2

            while directory_name in used_directory_names:
                directory_name = (
                    f"{base_directory_name}_{suffix}"
                )
                suffix += 1

            used_directory_names.add(directory_name)

            dataset_directory = (
                task_directory / directory_name
            )

            print(
                f"  Dataset {dataset_index:03d} : "
                f"{dataset_name}"
            )

            generated = process_dataset(
                task=task,
                dataset_name=dataset_name,
                dataset_df=dataset_df.copy(),
                dataset_directory=dataset_directory,
            )

            relative_report = (
                generated["report_path"]
                .relative_to(task_directory)
                .as_posix()
            )

            dataset_cards.append(
                {
                    "dataset_name": dataset_name,
                    "relative_report": relative_report,
                    "n_solvers": generated["n_solvers"],
                    "n_metrics": generated["n_metrics"],
                }
            )

        task_index = build_task_index(
            task=task,
            dataset_cards=dataset_cards,
            root_index_relative_path="../index.html",
            global_summary_html=build_global_summary_section(
                global_statistics.loc[global_statistics["task"].eq(task)]
            ),
        )

        task_index_path = (
            task_directory / "index.html"
        )

        task_index_path.write_text(
            task_index,
            encoding="utf-8",
        )

        task_cards.append(
            {
                "task_label": (
                    task.replace("_", " ").title()
                ),
                "relative_index": (
                    task_index_path
                    .relative_to(output_root)
                    .as_posix()
                ),
                "n_datasets": len(dataset_cards),
            }
        )

    root_index = build_root_index(
        task_cards
    )

    root_index_path = (
        output_root / "index.html"
    )

    root_index_path.write_text(
        root_index,
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Main index: {root_index_path}")
    print(f"Number of tasks: {len(task_cards)}")
    print(
        "Number of datasets: "
        f"{df['dataset_name'].nunique()}"
    )


if __name__ == "__main__":
    main()
