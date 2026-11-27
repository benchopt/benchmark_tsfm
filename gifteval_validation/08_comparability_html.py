#!/usr/bin/env python3

from pathlib import Path
from _paths import ARTIFACT_DIR as ARTIFACTS_DIR, HTML_DIR
import json
import pandas as pd
from global_statistics import load_statistics
from global_report import render_global_report


# =============================================================================
# PATHS
# =============================================================================

INPUT_FILE = ARTIFACTS_DIR / "raw_comparison_long.csv"
OUTPUT_FILE = HTML_DIR / "gifteval_comparison.html"


# =============================================================================
# LOAD AND VALIDATE
# =============================================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

HTML_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "solver_family",
    "gift_id",
    "metric",
    "benchopt_value",
    "gifteval_value",
    "signed_difference",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")


models = sorted(df["solver_family"].dropna().unique().tolist())
datasets = sorted(df["gift_id"].dropna().unique().tolist())

metric_order = ["MAE", "MSE", "MASE", "sMAPE", "WQL"]
available_metrics = set(df["metric"].dropna().unique())

metrics = [m for m in metric_order if m in available_metrics]


# =============================================================================
# PREPARE DATA
# =============================================================================

records = []

for _, row in df.iterrows():
    records.append({
        "model": str(row["solver_family"]),
        "dataset": str(row["gift_id"]),
        "metric": str(row["metric"]),
        "benchopt": float(row["benchopt_value"]),
        "gifteval": float(row["gifteval_value"]),
        "difference": float(row["signed_difference"]),
    })


data_json = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
models_json = json.dumps(models, ensure_ascii=False)
datasets_json = json.dumps(datasets, ensure_ascii=False)
metrics_json = json.dumps(metrics, ensure_ascii=False)


# =============================================================================
# HTML TEMPLATE
# =============================================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>BenchOpt TSFM — GIFT-Eval Comparison</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        Roboto,
        Arial,
        sans-serif;

    background: #f4f6fa;
    color: #172033;
}


/* ================================================================
   HEADER
   ================================================================ */

.header {
    background: linear-gradient(135deg, #172a55, #243e7e);
    color: white;
    padding: 34px 42px 30px 42px;
}

.header-inner {
    max-width: 1380px;
    margin: auto;
}

.header h1 {
    margin: 0;
    font-size: 29px;
    font-weight: 700;
}

.header p {
    margin: 9px 0 0 0;
    opacity: 0.83;
    font-size: 15px;
}


/* ================================================================
   SUMMARY
   ================================================================ */

.summary {
    max-width: 1380px;
    margin: 22px auto;
    padding: 0 35px;

    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
}

.stat {
    background: white;
    border: 1px solid #e0e5ee;
    border-radius: 10px;
    padding: 19px 21px;
    box-shadow: 0 2px 7px rgba(20, 35, 70, 0.04);
}

.stat-value {
    font-size: 27px;
    font-weight: 750;
    color: #243e7e;
}

.stat-label {
    margin-top: 4px;
    color: #687184;
    font-size: 13px;
}


/* ================================================================
   MAIN
   ================================================================ */

.main {
    max-width: 1380px;
    margin: auto;
    padding: 0 35px 50px 35px;
}


/* ================================================================
   TABS
   ================================================================ */

.tabs {
    display: flex;
    gap: 8px;
    margin: 24px 0 18px 0;
}

.tab {
    padding: 11px 19px;

    background: white;
    color: #34415d;

    border: 1px solid #d8deea;
    border-radius: 7px;

    font-size: 14px;
    font-weight: 650;

    cursor: pointer;
}

.tab:hover {
    background: #f7f8fb;
}

.tab.active {
    background: #243e7e;
    border-color: #243e7e;
    color: white;
}


/* ================================================================
   PANELS / CARDS
   ================================================================ */

.panel {
    display: none;
}

.panel.active {
    display: block;
}

.card {
    background: white;
    border: 1px solid #e0e5ed;
    border-radius: 10px;
    padding: 24px;

    box-shadow: 0 2px 8px rgba(20, 35, 70, 0.035);
}

h2 {
    margin: 0 0 7px 0;
    color: #172a55;
    font-size: 22px;
}

.subtitle {
    color: #70798b;
    font-size: 13px;
    margin-bottom: 22px;
}


/* ================================================================
   CONTROLS
   ================================================================ */

.controls {
    display: grid;
    grid-template-columns: minmax(190px, 0.6fr) minmax(300px, 1.4fr);
    gap: 15px;
    margin-bottom: 14px;
}

.control label {
    display: block;
    font-size: 12px;
    font-weight: 700;
    color: #596274;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

select,
input {
    width: 100%;

    padding: 11px 12px;

    border: 1px solid #ccd3df;
    border-radius: 6px;

    background: white;
    color: #172033;

    font-size: 14px;
}

select:focus,
input:focus {
    outline: 2px solid rgba(36, 62, 126, 0.16);
    border-color: #526ca8;
}


/* ================================================================
   NAVIGATION
   ================================================================ */

.dataset-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;

    margin: 14px 0 23px 0;
}

.dataset-nav button {
    padding: 8px 13px;

    background: white;
    color: #34415d;

    border: 1px solid #d5dbe6;
    border-radius: 6px;

    cursor: pointer;
}

.dataset-nav button:hover {
    background: #f3f5f9;
}


/* ================================================================
   DATASET / MODEL HEADER
   ================================================================ */

.result-header {
    padding: 16px 18px;

    background: #f6f8fc;
    border-left: 4px solid #243e7e;
    border-radius: 5px;

    margin-bottom: 18px;
}

.result-title {
    font-size: 19px;
    font-weight: 700;
    color: #172a55;
}

.result-context {
    margin-top: 4px;
    color: #6c7587;
    font-size: 13px;
}


/* ================================================================
   TABLES
   ================================================================ */

.table-wrap {
    overflow-x: auto;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 8px;
}

th {
    text-align: left;

    background: #eef2f8;
    color: #34415d;

    padding: 11px 13px;

    border-bottom: 2px solid #d8deea;

    font-size: 13px;
}

td {
    padding: 11px 13px;

    border-bottom: 1px solid #e6eaf0;

    font-size: 14px;
    font-variant-numeric: tabular-nums;
}

tbody tr:hover td {
    background: #fafbfe;
}

.metric {
    font-weight: 700;
    color: #243e7e;
}

.model-name {
    font-weight: 650;
}


/* ================================================================
   DIFFERENCE
   ================================================================ */

.diff-positive {
    color: #a33d32;
    font-weight: 600;
}

.diff-negative {
    color: #237a57;
    font-weight: 600;
}

.diff-zero {
    color: #687184;
}


/* ================================================================
   METRIC SECTIONS
   ================================================================ */

.metric-block {
    margin-top: 30px;
    padding-top: 3px;
}

.metric-heading {
    display: flex;
    align-items: center;
    gap: 10px;

    margin-bottom: 10px;
}

.metric-badge {
    display: inline-block;

    background: #243e7e;
    color: white;

    padding: 5px 10px;
    border-radius: 5px;

    font-size: 13px;
    font-weight: 700;
}


/* ================================================================
   RANKINGS
   ================================================================ */

.rankings {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 15px;

    margin-top: 14px;
}

.ranking-box {
    background: #f8f9fc;

    border: 1px solid #e1e5ed;
    border-radius: 7px;

    padding: 13px 15px;
}

.ranking-title {
    color: #172a55;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 7px;
}

.ranking-row {
    display: grid;
    grid-template-columns: 32px 1fr auto;
    gap: 6px;

    padding: 6px 0;

    border-bottom: 1px solid #e7eaf0;

    font-size: 13px;
}

.ranking-row:last-child {
    border-bottom: none;
}

.rank-number {
    color: #7b8495;
}

.rank-value {
    font-variant-numeric: tabular-nums;
    color: #485267;
}


/* ================================================================
   NOTE / MISSING
   ================================================================ */

.note {
    color: #70798b;
    font-size: 12px;
    margin-top: 13px;
}

.missing {
    color: #8a6d2f;
    font-style: italic;
}

.empty {
    margin-top: 15px;

    padding: 15px;

    background: #fff8e8;
    border: 1px solid #ead9a5;
    border-radius: 6px;

    color: #725e24;
}


/* ================================================================
   FOOTER
   ================================================================ */

.footer {
    max-width: 1380px;
    margin: 0 auto 30px auto;
    padding: 0 35px;

    color: #858d9c;
    font-size: 12px;
}


/* ================================================================
   RESPONSIVE
   ================================================================ */

@media (max-width: 850px) {

    .summary {
        grid-template-columns: repeat(2, 1fr);
        padding: 0 18px;
    }

    .main {
        padding: 0 18px 35px 18px;
    }

    .header {
        padding: 28px 22px;
    }

    .controls,
    .rankings {
        grid-template-columns: 1fr;
    }

}

</style>

</head>


<body>


<!-- ===============================================================
     HEADER
     =============================================================== -->

<header class="header">

<div class="header-inner">

    <h1>BenchOpt TSFM — GIFT-Eval Comparison</h1>

    <p>
        Direct comparison of BenchOpt benchmark results
        with GIFT-Eval reference values.
    </p>

</div>

</header>


<!-- ===============================================================
     SUMMARY
     =============================================================== -->

<section class="summary">

    <div class="stat">
        <div class="stat-value">__N_MODELS__</div>
        <div class="stat-label">Models compared</div>
    </div>

    <div class="stat">
        <div class="stat-value">__N_DATASETS__</div>
        <div class="stat-label">Dataset configurations</div>
    </div>

    <div class="stat">
        <div class="stat-value">__N_METRICS__</div>
        <div class="stat-label">Metrics</div>
    </div>

    <div class="stat">
        <div class="stat-value">__N_ROWS__</div>
        <div class="stat-label">Raw comparisons</div>
    </div>

</section>


<!-- ===============================================================
     MAIN
     =============================================================== -->

<main class="main">


<!-- TABS -->

<div class="tabs">

    <button
        class="tab active"
        onclick="showPanel('model-panel', this)"
    >
        Browse by model
    </button>

    <button
        class="tab"
        onclick="showPanel('dataset-panel', this)"
    >
        Browse by dataset
    </button>

</div>


<!-- ===============================================================
     MODEL VIEW
     =============================================================== -->

<section id="model-panel" class="panel active">

<div class="card">

    <h2>Comparison by model</h2>

    <div class="subtitle">
        Select a model and a GIFT-Eval dataset configuration.
    </div>


    <div class="controls">

        <div class="control">

            <label for="model-select">
                Model
            </label>

            <select
                id="model-select"
                onchange="renderModelView()"
            ></select>

        </div>


        <div class="control">

            <label for="model-dataset-select">
                Dataset
            </label>

            <select
                id="model-dataset-select"
                onchange="changeModelDataset()"
            ></select>

        </div>

    </div>


    <div class="dataset-nav">

        <button onclick="moveModelDataset(-1)">
            ← Previous dataset
        </button>

        <button onclick="moveModelDataset(1)">
            Next dataset →
        </button>

    </div>


    <div id="model-result"></div>

</div>

</section>


<!-- ===============================================================
     DATASET VIEW
     =============================================================== -->

<section id="dataset-panel" class="panel">

<div class="card">

    <h2>Comparison by dataset</h2>

    <div class="subtitle">
        Compare all available models on one GIFT-Eval dataset configuration.
    </div>


    <div class="control">

        <label for="dataset-select">
            Dataset
        </label>

        <select
            id="dataset-select"
            onchange="changeDataset()"
        ></select>

    </div>


    <div class="dataset-nav">

        <button onclick="moveDataset(-1)">
            ← Previous dataset
        </button>

        <button onclick="moveDataset(1)">
            Next dataset →
        </button>

    </div>


    <div id="dataset-result"></div>

</div>

</section>


</main>


<footer class="footer">

    Difference = BenchOpt − GIFT-Eval.
    Lower values indicate better performance for the error metrics shown.

</footer>


<!-- ===============================================================
     JAVASCRIPT
     =============================================================== -->

<script>

const DATA = __DATA__;
const MODELS = __MODELS__;
const DATASETS = __DATASETS__;
const METRICS = __METRICS__;

let currentModelDatasetIndex = 0;
let currentDatasetIndex = 0;


/* ================================================================
   FORMATTING
   ================================================================ */

function formatNumber(value) {

    if (
        value === null ||
        value === undefined ||
        Number.isNaN(value)
    ) {
        return "—";
    }

    const absValue = Math.abs(value);

    if (
        absValue !== 0 &&
        (
            absValue >= 1000000 ||
            absValue < 0.0001
        )
    ) {
        return value.toExponential(6);
    }

    return value.toLocaleString(
        "en-US",
        {
            maximumFractionDigits: 6
        }
    );
}


function formatDifference(value) {

    if (
        value === null ||
        value === undefined ||
        Number.isNaN(value)
    ) {
        return "—";
    }

    const prefix = value > 0 ? "+" : "";

    return prefix + formatNumber(value);
}


function differenceClass(value) {

    if (value > 0) {
        return "diff-positive";
    }

    if (value < 0) {
        return "diff-negative";
    }

    return "diff-zero";
}


/* ================================================================
   TABS
   ================================================================ */

function showPanel(panelId, button) {

    document
        .querySelectorAll(".panel")
        .forEach(panel => {
            panel.classList.remove("active");
        });

    document
        .querySelectorAll(".tab")
        .forEach(tab => {
            tab.classList.remove("active");
        });

    document
        .getElementById(panelId)
        .classList.add("active");

    button.classList.add("active");
}


/* ================================================================
   INITIALIZATION
   ================================================================ */

function populateSelect(selectId, values) {

    const select = document.getElementById(selectId);

    select.innerHTML = "";

    values.forEach(value => {

        const option = document.createElement("option");

        option.value = value;
        option.textContent = value;

        select.appendChild(option);
    });
}


function initialize() {

    populateSelect(
        "model-select",
        MODELS
    );

    populateSelect(
        "model-dataset-select",
        DATASETS
    );

    populateSelect(
        "dataset-select",
        DATASETS
    );

    currentModelDatasetIndex = 0;
    currentDatasetIndex = 0;

    renderModelView();
    renderDatasetView();
}


/* ================================================================
   MODEL VIEW
   ================================================================ */

function changeModelDataset() {

    const select =
        document.getElementById(
            "model-dataset-select"
        );

    currentModelDatasetIndex =
        DATASETS.indexOf(select.value);

    renderModelView();
}


function moveModelDataset(step) {

    currentModelDatasetIndex =
        (
            currentModelDatasetIndex +
            step +
            DATASETS.length
        ) % DATASETS.length;

    document
        .getElementById("model-dataset-select")
        .value =
            DATASETS[currentModelDatasetIndex];

    renderModelView();
}


function renderModelView() {

    const model =
        document
            .getElementById("model-select")
            .value;

    const dataset =
        DATASETS[currentModelDatasetIndex];

    const rows =
        DATA.filter(row =>
            row.model === model &&
            row.dataset === dataset
        );

    const target =
        document.getElementById(
            "model-result"
        );


    let output = `
        <div class="result-header">

            <div class="result-title">
                ${model}
            </div>

            <div class="result-context">
                ${dataset}
            </div>

        </div>
    `;


    if (rows.length === 0) {

        output += `
            <div class="empty">
                No comparison available for this model and dataset.
            </div>
        `;

        target.innerHTML = output;

        return;
    }


    output += `
        <div class="table-wrap">

        <table>

            <thead>
                <tr>
                    <th>Metric</th>
                    <th>BenchOpt</th>
                    <th>GIFT-Eval</th>
                    <th>Difference</th>
                </tr>
            </thead>

            <tbody>
    `;


    METRICS.forEach(metric => {

        const row =
            rows.find(
                item => item.metric === metric
            );


        if (!row) {

            output += `
                <tr>

                    <td class="metric">
                        ${metric}
                    </td>

                    <td class="missing">Not available</td>
                    <td class="missing">Not available</td>
                    <td class="missing">—</td>

                </tr>
            `;

            return;
        }


        output += `
            <tr>

                <td class="metric">
                    ${metric}
                </td>

                <td>
                    ${formatNumber(row.benchopt)}
                </td>

                <td>
                    ${formatNumber(row.gifteval)}
                </td>

                <td class="${differenceClass(row.difference)}">
                    ${formatDifference(row.difference)}
                </td>

            </tr>
        `;
    });


    output += `
            </tbody>

        </table>

        </div>

        <div class="note">
            Difference = BenchOpt − GIFT-Eval.
        </div>
    `;


    target.innerHTML = output;
}


/* ================================================================
   DATASET VIEW
   ================================================================ */

function changeDataset() {

    const select =
        document.getElementById(
            "dataset-select"
        );

    currentDatasetIndex =
        DATASETS.indexOf(select.value);

    renderDatasetView();
}


function moveDataset(step) {

    currentDatasetIndex =
        (
            currentDatasetIndex +
            step +
            DATASETS.length
        ) % DATASETS.length;

    document
        .getElementById("dataset-select")
        .value =
            DATASETS[currentDatasetIndex];

    renderDatasetView();
}


function rankingHTML(rows, key, title) {

    const sorted =
        rows
            .slice()
            .sort(
                (a, b) =>
                    a[key] - b[key]
            );


    let output = `
        <div class="ranking-box">

            <div class="ranking-title">
                ${title}
            </div>
    `;


    sorted.forEach((row, index) => {

        output += `
            <div class="ranking-row">

                <span class="rank-number">
                    ${index + 1}.
                </span>

                <span>
                    ${row.model}
                </span>

                <span class="rank-value">
                    ${formatNumber(row[key])}
                </span>

            </div>
        `;
    });


    output += `
        </div>
    `;


    return output;
}


function renderDatasetView() {

    const dataset =
        DATASETS[currentDatasetIndex];

    const target =
        document.getElementById(
            "dataset-result"
        );


    let output = `
        <div class="result-header">

            <div class="result-title">
                ${dataset}
            </div>

            <div class="result-context">
                Comparison across available models
            </div>

        </div>
    `;


    METRICS.forEach(metric => {

        const rows =
            DATA.filter(row =>
                row.dataset === dataset &&
                row.metric === metric
            );


        output += `
            <div class="metric-block">

                <div class="metric-heading">

                    <span class="metric-badge">
                        ${metric}
                    </span>

                </div>
        `;


        if (rows.length === 0) {

            output += `
                <div class="empty">
                    This metric is not available for this dataset.
                </div>

                </div>
            `;

            return;
        }


        const tableRows =
            rows
                .slice()
                .sort(
                    (a, b) =>
                        a.benchopt - b.benchopt
                );


        output += `
            <div class="table-wrap">

            <table>

                <thead>
                    <tr>
                        <th>Model</th>
                        <th>BenchOpt</th>
                        <th>GIFT-Eval</th>
                        <th>Difference</th>
                    </tr>
                </thead>

                <tbody>
        `;


        tableRows.forEach(row => {

            output += `
                <tr>

                    <td class="model-name">
                        ${row.model}
                    </td>

                    <td>
                        ${formatNumber(row.benchopt)}
                    </td>

                    <td>
                        ${formatNumber(row.gifteval)}
                    </td>

                    <td class="${differenceClass(row.difference)}">
                        ${formatDifference(row.difference)}
                    </td>

                </tr>
            `;
        });


        output += `
                </tbody>

            </table>

            </div>


            <div class="rankings">

                ${rankingHTML(
                    rows,
                    "benchopt",
                    "BenchOpt ranking"
                )}

                ${rankingHTML(
                    rows,
                    "gifteval",
                    "GIFT-Eval ranking"
                )}

            </div>

            </div>
        `;
    });


    target.innerHTML = output;
}


/* ================================================================
   START
   ================================================================ */

initialize();

</script>


</body>
</html>
"""


# =============================================================================
# INJECT DATA
# =============================================================================

html_page = HTML_TEMPLATE

html_page = html_page.replace(
    "__DATA__",
    data_json
)

html_page = html_page.replace(
    "__MODELS__",
    models_json
)

html_page = html_page.replace(
    "__DATASETS__",
    datasets_json
)

html_page = html_page.replace(
    "__METRICS__",
    metrics_json
)

html_page = html_page.replace(
    "__N_MODELS__",
    str(len(models))
)

html_page = html_page.replace(
    "__N_DATASETS__",
    str(len(datasets))
)

html_page = html_page.replace(
    "__N_METRICS__",
    str(len(metrics))
)

html_page = html_page.replace(
    "__N_ROWS__",
    str(len(df))
)


# =============================================================================
# WRITE HTML
# =============================================================================

global_stats, global_scope = load_statistics(ARTIFACTS_DIR)
global_section = render_global_report(global_stats, global_scope)
global_button = '''<button class="tab" onclick="showPanel('global-panel', this)">
        Global Performance Summary
    </button>
'''
tabs_end = html_page.index('</div>', html_page.index('<div class="tabs">'))
html_page = html_page[:tabs_end] + global_button + html_page[tabs_end:]
html_page = html_page.replace('</main>', global_section + '</main>', 1)

OUTPUT_FILE.write_text(
    html_page,
    encoding="utf-8"
)


# =============================================================================
# FINAL REPORT
# =============================================================================

print("=" * 100)
print("GIFT-EVAL COMPARISON HTML")
print("=" * 100)

print(f"\nInput:")
print(f"  {INPUT_FILE}")

print(f"\nOutput:")
print(f"  {OUTPUT_FILE}")

print("\nContent:")
print(f"  Models       : {len(models)}")
print(f"  Datasets     : {len(datasets)}")
print(f"  Metrics      : {len(metrics)}")
print(f"  Comparisons  : {len(df)}")

print("\nModels:")
for model in models:
    print(f"  - {model}")

print("\nMetrics:")
for metric in metrics:
    count = int((df["metric"] == metric).sum())
    print(f"  - {metric:<8} {count:>4} rows")

print("\nHTML size:")
print(f"  {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")

print("\nHTML report generated successfully.")
print("=" * 100)
