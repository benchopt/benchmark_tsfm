# TSFM Benchmark Explorer

Generate a static HTML explorer from existing BenchOpt TSFM results, with task
indexes, dataset reports, per-metric model rankings, and summary statistics.
This pipeline performs post-processing only; it does not train or run models.

## Run

Use a Python environment with **NumPy, pandas, and PyArrow**, and provide:

```text
outputs/benchmark_tsfm_standard.parquet
outputs/gifteval_all.parquet
```

From the `benchmark_tsfm/` repository root:

```bash
bash scripts/results_pipeline/run_dataset_reports.sh
```

Set `PYTHON=/path/to/python` before the command to select an interpreter.
Open **`results/reports_by_dataset/index.html`** in a browser. No server is
required. Share the entire `reports_by_dataset/` directory to preserve links.

## Pipeline

The runner uses `scripts/results_pipeline/_common.sh` and executes:

| Stage | Script in `results_pipeline/` | Purpose |
|---|---|---|
| 00 | `merge_results.py` | Merge the two input Parquet files. |
| 01 | `analysis.py` | Identify tasks and retain rows with at least one finite performance metric. |
| 02 | `leaderboard.py` | Average repeated measurements and rank model configurations. |
| 03 | `validate_leaderboard.py` | Export diagnostics for coverage, duplicates, and metric consistency. |
| 04 | `scientific_leaderboard.py` | Compute scientific ranks and fractional wins using explicit metric directions. |
| 04b | `global_statistics.py` | Compute summary statistics using `benchmark_utils/global_statistics.py`. |
| 05 | `dataset_reports.py` | Generate HTML pages and dataset-level CSV exports. |

Intermediate tables, text reports, and logs are saved in `results/csv/`,
`results/reports/`, and `results/logs/`. Execution stops if a stage fails.

## Reading the results

- **Overall position:** models are ranked by their mean rank for the selected
  metric, not by their mean raw value. Detailed dataset tables rank metric values.
- **Ties:** scientific ranks use average ranks (`1.5, 1.5, 3`); displayed positions
  use competition ranking (`1, 1, 3`). Only exact equality counts as a tie.
- **Shared win rate:** each best model receives `1/k` when `k` models tie.
  The rate is the sum of these points divided by the model's valid configurations.
- **Standard deviation:** the summary uses metric values across configurations
  with `ddof=0`, not variation across repeated runs or statistical significance.
- **Coverage:** valid configurations for the model and metric divided by all
  configurations of that task present in the source CSV. Missing values are not
  zeros or defeats; models may have different coverage.

Model configurations remain separate. Raw metric means can be dominated by
large-scale datasets. The separate cross-metric task-ranking export excludes
Pinball Loss, VUS-ROC, VUS-PR, and mAP-IoU under a fixed policy; valid values for
these metrics remain available in the explorer.

## Checks

From the repository root:

```bash
python -m unittest discover -s tests/benchmark_utils -p test_global_statistics.py -v
python -m unittest discover -s tests/results_pipeline -p test_global_statistics.py -v
```

Summary artifacts include source/output hashes to detect stale statistics.
Rerun the pipeline when inputs change. Existing output files are overwritten,
but obsolete files from earlier runs are not automatically deleted.
