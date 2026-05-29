# scripts/

Helper scripts that sit alongside the benchopt benchmark. None of them are
solvers, datasets, or objectives — they're tooling for importing upstream
benchmark results and producing statistical comparison figures.

## Prerequisites

The scripts share the benchmark's `.venv`:

```bash
uv venv --python 3.11 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -U benchopt scikit-learn aeon \
    scikit-posthocs matplotlib
```

`scikit-posthocs` (and its `statsmodels` / `seaborn` deps) is only required by
`plot_cd_grid.py`; the FEV importer needs nothing beyond `numpy` / `pandas`.

---

## `import_fev_bench.py` — import upstream FEV-bench results

Downloads per-model CSV result files from
[`autogluon/fev`](https://github.com/autogluon/fev/tree/main/benchmarks/fev_bench/results)
and rewrites them as a single benchopt-schema parquet so they can be
visualized alongside native runs in the benchopt HTML report.

### Usage

```bash
# Fresh import (downloads to /tmp/fev_bench_results, writes outputs/fev_bench_results.parquet)
python scripts/import_fev_bench.py

# Reuse cached CSVs from an earlier run (skip GitHub fetch)
python scripts/import_fev_bench.py --no-download

# Force refresh
python scripts/import_fev_bench.py --force --cache-dir /tmp/fev

# Custom output path
python scripts/import_fev_bench.py --out data/fev_full.parquet
```

| Flag | Default | Purpose |
|---|---|---|
| `--out` | `outputs/fev_bench_results.parquet` | Output parquet path |
| `--cache-dir` | `/tmp/fev_bench_results` | Where downloaded CSVs are kept between runs |
| `--no-download` | off | Reuse cached CSVs; skip GitHub fetch entirely |
| `--force` | off | Re-download all CSVs even if cached |

### What it produces

A parquet matching this repo's `objective.py` schema:

- `objective_name = "Forecasting"`
- `solver_name`   = `"<model>[framework_version=...]"`
- `dataset_name`  = `"FEV[task=...,horizon=...,n_windows=...,seasonality=...,eval_metric=...]"`
- `time`          = `training_time_s + inference_time_s`
- `objective_<m>` columns for SQL, MASE, WAPE, WQL, training_time_s,
  inference_time_s, test_error
- `p_dataset_*`, `p_solver_*` parameter columns
- Environment placeholders (`platform`, `version-*`, `run_date`, …)

Typical output for the current upstream snapshot: **2,085 rows × 21 solvers ×
100 forecasting tasks**.

### Hook into benchopt HTML

After import, generate the per-run HTML page exactly like a native run:

```bash
.venv/bin/benchopt plot . --html --no-display \
    -f outputs/fev_bench_results.parquet
```

The index page (`outputs/benchmark_tsfm.html`) will then list FEV alongside
your native runs.

### Discovery & caching

- Available models are discovered live via the GitHub Contents API
  (`/repos/autogluon/fev/contents/benchmarks/fev_bench/results`). If the API
  is rate-limited or unreachable, the script falls back to a hardcoded
  21-model list. Set `GITHUB_TOKEN` to raise the rate-limit ceiling.
- Cached CSVs are reused if they exist and are less than 1 day old; otherwise
  the script re-downloads them. Use `--force` to bypass.
- Downloads run in 8 parallel threads.

---

## `plot_cd_grid.py` — Critical Difference diagrams per metric

Renders a Demšar (2006) Critical Difference diagram for every
`objective_*` numeric column in a benchopt-schema parquet, arranged into a
single PNG grid. Uses `scipy.stats.friedmanchisquare` +
`scikit_posthocs.posthoc_nemenyi_friedman` +
`scikit_posthocs.critical_difference_diagram` under the hood.

### Usage

```bash
# Standard: 2-column grid, all metrics with sufficient data
python scripts/plot_cd_grid.py outputs/fev_bench_results.parquet --ncols 2

# Filter rows before pivoting (e.g. one TabArena problem type only)
python scripts/plot_cd_grid.py data/tabarena_full.parquet \
    --filter "p_dataset_problem_type=='binary'" --ncols 2

# Custom output path
python scripts/plot_cd_grid.py outputs/fev_bench_results.parquet \
    --out figures/cd_fev.png
```

| Flag | Default | Purpose |
|---|---|---|
| `parquet` | (required) | Benchopt-schema parquet to analyze |
| `--out` | `cd_grid_<stem>.png` next to the parquet | Output PNG path |
| `--ncols` | `2` | Columns in the subplot grid |
| `--filter` | none | pandas `query` string applied before pivoting |
| `--alpha` | `0.05` | Significance level for Friedman + Nemenyi |
| `--top-k` | none | Keep only the N best-mean-rank solvers per metric (readability) |

### What it produces

One PNG containing a grid of CD diagrams — one subplot per numeric
`objective_*` column found in the parquet. Each subplot's title reports
`k` (solvers), `N` (datasets), Friedman χ², Friedman p-value, and the
Nemenyi critical difference at α=0.05.

Thick horizontal bars on each diagram connect solvers whose mean-rank
difference is below the CD — i.e., the Demšar-style equivalence cliques.

### Handling sparse benchmarks

If the strict cleaning step (drop datasets where any solver is NaN, drop
all-tied datasets) leaves no datasets, the script falls back to a greedy
biclique trim that iteratively drops whichever row or column has the most
NaNs. This keeps the largest dense submatrix and reports `k`, `N` after
trim in the subplot title.

For benchmarks where this fallback kicks in, the trimmed solver set is
much smaller than the original — interpret with that in mind. Examples:
GIFT-Eval's 88 models drop to 5–10 after trim because most models cover
only ~half the datasets.

### Higher-is-better metrics

By default the script ranks ascending (lower-is-better). For accuracy or
ROC-AUC metrics, pass the higher-is-better variant directly — there's no
flag; just supply a metric column where lower is better. If you need
higher-is-better support, the cleanest path is to negate the column
upstream (e.g., store `roc_auc_error = 1 - roc_auc` rather than `roc_auc`).
