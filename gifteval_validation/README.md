# BenchOpt TSFM / GIFT-Eval Comparison

This directory compares forecasting results produced by BenchOpt TSFM with
GIFT-Eval reference results, then generates an HTML report with views by model,
by dataset, and an overall summary table.

The pipeline uses previously computed results: it does not run training or
inference. It is independent of the Dataset Explorer.

## 1. Inputs and references

Paths are defined in [`_paths.py`](_paths.py).

| Input | Default location |
|---|---|
| BenchOpt results | `benchmark_tsfm/results/benchmark_tsfm_results_only.parquet` |
| Reference repository | `references/gift-eval/`, alongside `benchmark_tsfm/` |
| Reference values | `references/gift-eval/results/<model>/all_results.csv` |
| Model metadata | `references/gift-eval/results/<model>/config.json`, when available |

Set the `GIFT_EVAL_REPO` environment variable to use a different local reference
repository. The pipeline does not download these files.

The Parquet file is prepared upstream. This pipeline does not perform the
cleaning of raw Parquet files that produced the `results_only` input.

### Models included in numerical comparisons

| BenchOpt model | GIFT-Eval directory |
|---|---|
| Chronos | `Chronos_small` |
| Chronos2 | `chronos-2` |
| Toto-2.0 | `Toto-2.0-2.5B` |
| Naive | `naive` |

The inventory may also inspect SeasonalNaive, but the overall table compares
the four models above. Summary checks require a single BenchOpt configuration
per model: multiple configurations are not silently merged.

### Metrics compared

| Metric | BenchOpt column | GIFT-Eval column |
|---|---|---|
| MAE | `objective_mae` | `eval_metrics/MAE[0.5]` |
| MSE | `objective_mse` | `eval_metrics/MSE[0.5]` |
| MASE | `objective_mase` | `eval_metrics/MASE[0.5]` |
| sMAPE | `objective_smape` | `eval_metrics/sMAPE[0.5]` |
| WQL | `objective_wql` | `eval_metrics/mean_weighted_sum_quantile_loss` |

These mappings are marked `CANDIDATE` in the code: they do not, by themselves,
prove that the protocols or implementations are identical. Metrics without a
direct equivalent are not forced into the comparison. For example, BenchOpt
RMSE based on the median forecast is not equated with the reference `RMSE[mean]`.

## 2. Run the full pipeline through HTML generation

Use a Python environment containing **NumPy, pandas, PyArrow, and Matplotlib**,
with the Parquet input and reference files already available.

From the working directory containing `benchmark_tsfm/`:

```bash
bash benchmark_tsfm/scripts/gifteval_validation/run_validation.sh
```

From the `benchmark_tsfm/` repository root:

```bash
bash scripts/gifteval_validation/run_validation.sh
```

The runner uses `python` by default. To explicitly select the interpreter and
reference repository, run from the repository root:

```bash
PYTHON=/path/to/environment/bin/python \
GIFT_EVAL_REPO=/path/to/gift-eval \
bash scripts/gifteval_validation/run_validation.sh
```

On a machine without a graphical interface, add `MPLBACKEND=Agg` to the
environment for figure generation.

The [runner](../scripts/gifteval_validation/run_validation.sh) executes all
nine stages in order, logs their output in `outputs/`, and stops if a stage
fails. The final report is:

```text
benchmark_tsfm/results/gifteval_validation/html/gifteval_comparison.html
```

Open this file in a browser. No server is required. To also share downloadable
files linked from the report, preserve the results directory structure,
including the `artifacts/` directory.

## 3. What each script does, from inputs to HTML

All Python files below are located in `gifteval_validation/`.

| Stage | File | Purpose |
|---|---|---|
| Support | [`_paths.py`](_paths.py) | Centralizes input and output paths. |
| 01 | [`01_model_mapping.py`](01_model_mapping.py) | Inventories local and reference models, including their metadata. |
| 02 | [`02_reference_schema.py`](02_reference_schema.py) | Inspects available columns in the reference CSV files. |
| 03 | [`03_dataset_alignment.py`](03_dataset_alignment.py) | Builds configuration identifiers and lists common, missing, and extra configurations. |
| 04 | [`04_metric_source_audit.py`](04_metric_source_audit.py) | Locates metric-related code to support auditing. This search does not establish equivalence. |
| 05 | [`05_metric_equivalence.py`](05_metric_equivalence.py) | Maps metrics, joins local and reference values, retains finite numerical pairs, and computes detailed differences. |
| 06 | [`06_comparability_report.py`](06_comparability_report.py) | Produces long and wide tables, descriptive summaries, correlations, and detailed rankings. |
| 06b | [`global_statistics.py`](global_statistics.py) | Validates inputs, builds a common panel per metric, and computes summary means, positions, differences, and standard deviations. |
| 07 | [`07_comparability_figures.py`](07_comparability_figures.py) | Exports comparison figures to `figures/`. |
| 08 | [`08_comparability_html.py`](08_comparability_html.py) | Assembles the HTML report with model, dataset, and overall views. |
| Rendering | [`global_report.py`](global_report.py) | Builds the overall table, metric selector, colors, and percentages. Called by stage 08. |

## 4. Configuration alignment and selection

The comparison identifier combines the dataset name and forecasting term:
`gift_id = p_dataset_dataset_name + "/" + p_dataset_term`.
Stage 03 establishes the initial alignment using the Toto-2.0 reference.
The summary then verifies that all four references have the same identifiers.

Two separate selections take place:

1. **Configuration alignment**: intersect the identifiers present in local
   GiftEval results and in the references.
2. **Valid panel per metric**: among aligned configurations, retain only those
   for which all four models have a finite value in both systems. A missing
   pair removes that configuration from the summary for all models for that
   metric.

Detailed views use available pairs. The overall table uses this stricter
selection to compare models on the same configurations. The panel may differ
between metrics.

In the currently available artifacts:

| Scope | Number of configurations |
|---|---:|
| GIFT-Eval reference | 97 |
| Aligned with BenchOpt | 82 |
| Overall panel for MAE, MSE, sMAPE, and WQL | 82 per metric |
| Overall panel for MASE | 67 |

The **15 unaligned reference configurations** and the **15 aligned configurations
excluded from the MASE panel** result from different filters. These counts
describe the current inputs; they are not constants enforced by the pipeline.

See `dataset_alignment.csv`, `common_datasets.txt`, `missing_datasets.txt`, and
`extra_datasets.txt` for alignment details. Overall table exclusions are listed
in `global_statistics_exclusions.csv`, together with the models whose pairs
are missing. This table alone does not identify which side lacks a value;
inspect the inputs to determine that.

## 5. Reading the overall table

Selecting a metric displays all four models in the same table. Column names
below are English translations of the report labels.

| Column | Meaning |
|---|---|
| BenchOpt position | Ranking by the mean BenchOpt metric value. |
| GIFT-Eval position | Independent ranking by the mean reference value on the same panel. |
| Model | Model being compared. |
| BenchOpt mean | Mean of local values over the valid panel. |
| GIFT-Eval mean | Mean of reference values over that same panel. |
| Difference | Signed difference between means, followed by the percentage in parentheses. |
| BenchOpt standard deviation | Dispersion of local values across configurations. |
| GIFT-Eval standard deviation | Dispersion of reference values across configurations. |

### Means and positions

Each configuration counts once. For a given model and system, with `N` valid
configurations:

```text
mean = sum of metric values / N
```

All five metrics are minimized: a lower mean gives a better position. Rankings
are computed independently for each system using unrounded values. Exactly
equal means share the same position, with subsequent positions skipped, for
example `1, 1, 3`.

Rows are sorted by BenchOpt position, then by model name. This ranks metric
means, rather than averaging ranks obtained on individual datasets. GIFT-Eval
values are recalculated on our common panel: they do not represent a complete
official ranking across all 97 configurations.

### Difference, percentage, and colors

```text
difference = BenchOpt mean − GIFT-Eval mean
percentage = 100 × difference / abs(GIFT-Eval mean)
```

Example: BenchOpt `80`, GIFT-Eval `100` → `−20 (−20.00 %)`.

- **Green**: negative difference, lower mean error for BenchOpt.
- **Red**: positive difference, higher mean error for BenchOpt.
- **Neutral**: zero difference.

Calculations precede display rounding. If the absolute reference mean is less
than or equal to `1e-12`, the difference remains visible but the percentage is
undefined (displayed as `% non défini` in the report). An unavailable difference
is displayed as a dash.

### Standard deviations

For each system, the standard deviation is computed from metric values across
the retained configurations, using `ddof=0`:

```text
standard deviation = sqrt(sum((value − mean)²) / N)
```

It describes dispersion across configurations. With one configuration, it is
zero; with an empty panel, statistics are unavailable. The table does not
compute a difference between standard deviations, a mean rank, or a shared win
rate. The number of configurations in the panel remains visible above it.

## 6. Outputs and traceability

```text
results/gifteval_validation/
├── artifacts/    # CSV tables, alignment lists, and metadata
├── figures/      # PNG figures
├── html/         # gifteval_comparison.html
└── outputs/      # One .out log per stage, including 06b_global_statistics.out
```

Main files in `artifacts/`:

| File | Purpose |
|---|---|
| `model_mapping.csv` | Model mappings. |
| `metric_mapping.csv` | Metric mappings and their status. |
| `metric_source_locations.csv` | Locations identified during the code audit. |
| `dataset_alignment.csv` | Configuration alignment. |
| `metric_equivalence_detailed.csv` | Compared pairs, including the solver configuration. |
| `metric_equivalence_summary.csv` | Stage 05 summaries. |
| `raw_comparison_long.csv` | One row per compared model, configuration, and metric. |
| `raw_comparison_wide.csv` | Wide view of comparisons. |
| `comparability_*.csv`, `comparability_summary.txt` | Detailed descriptive analyses from stage 06. |
| `global_statistics.csv` | Statistics used by the overall table. |
| `global_statistics_scope.json` | Panels, calculation conventions, and SHA-256 hashes of checked sources and outputs. |
| `global_statistics_exclusions.csv` | Configurations excluded from the overall panel, by metric. |

Summary validation rejects duplicate keys, ambiguous model configurations,
reference files with incompatible identifiers, and inconsistencies between
comparison tables. Loading statistics for HTML generation checks metadata and
hashes: stale statistics must be regenerated.

Note the conventions: some detailed analyses use **absolute differences** and
**absolute relative differences**. The overall table's Difference column uses
a **signed** difference and its signed percentage. Detailed rankings from
stage 06 remain separate from the overall ranking.

## 7. Partial regeneration and verification

After changing the overall statistics or rendering, if upstream artifacts are
already current, run from `benchmark_tsfm/` with the Python environment activated:

```bash
python gifteval_validation/global_statistics.py
python gifteval_validation/08_comparability_html.py
```

If local data or references have changed, rerun the full pipeline. After moving
the project, regenerate metadata: it records the paths of checked sources.
Keep `GIFT_EVAL_REPO` set if references are outside the default location.

Run the targeted summary tests from the repository root:

```bash
python -m unittest discover -s tests/gifteval_validation -p test_global_statistics.py -v
```

## 8. Interpretation limits

The percentage expresses the size of the difference relative to the reference.
It is not a statistical significance test. Colors indicate only the direction
of the observed difference; the two standard deviations are insufficient to
conclude that a difference is statistically significant.

Means use raw metric values without additional normalization. Datasets with
large numerical scales can dominate a mean MAE or MSE. A dataset represented
by multiple configurations contributes multiple times.

Finally, differences from GIFT-Eval may reflect differences in checkpoints,
preprocessing, context, evaluation protocols, or metric calculations. The
pipeline's correlations and descriptive categories are neither an official
certification nor proof of exact reproduction.
