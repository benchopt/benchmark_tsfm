"""Strict aligned-panel summaries, independent of the Dataset Explorer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gifteval_validation._paths import ARTIFACT_DIR, BENCH_FILE, GIFT_REPO

MODELS = {"Chronos": "Chronos_small", "Chronos2": "chronos-2",
          "Toto-2.0": "Toto-2.0-2.5B", "Naive": "naive"}
METRICS = {"MAE": "objective_mae", "MSE": "objective_mse",
           "MASE": "objective_mase", "sMAPE": "objective_smape", "WQL": "objective_wql"}
SUMMARY = "global_statistics.csv"
SCOPE = "global_statistics_scope.json"
EXCLUSIONS = "global_statistics_exclusions.csv"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_statistics(raw, aligned_ids, models=tuple(MODELS), metrics=tuple(METRICS)):
    """Use one complete four-model, two-system panel per metric.

    Missing pairs remove that configuration for *all* models and both systems.
    Preserve the aligned universe with explicit NaNs for coverage denominators.
    """
    ids = sorted(set(aligned_ids))
    if not ids or not models or not metrics:
        raise ValueError("The aligned universe, models and metrics must not be empty")
    keys = ["solver_family", "gift_id", "metric"]
    required = set(keys + ["benchopt_value", "gifteval_value"])
    if missing := required.difference(raw.columns):
        raise ValueError(f"Missing comparison columns: {sorted(missing)}")
    if raw[keys].isna().any().any() or raw.duplicated(keys).any():
        raise ValueError("Missing or duplicate comparison keys; do not merge model configurations")
    if (set(raw.solver_family) - set(models) or set(raw.metric) - set(metrics)
            or set(raw.gift_id) - set(ids)):
        raise ValueError("Comparison observations outside the declared aligned scope")
    grid = pd.MultiIndex.from_product([models, ids, metrics], names=keys)
    pairs = raw.set_index(keys).reindex(grid)
    for col in ["benchopt_value", "gifteval_value"]:
        pairs[col] = pd.to_numeric(pairs[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    pairs["pair_valid"] = pairs[["benchopt_value", "gifteval_value"]].notna().all(axis=1)
    complete = pairs.groupby(level=["gift_id", "metric"])["pair_valid"].all()
    exclusions = []
    panel_ids = {}
    for metric in metrics:
        panel_ids[metric] = []
        for dataset in ids:
            if complete.loc[(dataset, metric)]:
                panel_ids[metric].append(dataset)
            else:
                missing_models = [m for m in models if not pairs.loc[(m, dataset, metric), "pair_valid"]]
                exclusions.append(dict(gift_id=dataset, metric=metric,
                                       missing_pair_models="|".join(missing_models),
                                       reason="Missing or nonfinite pair; removed from both systems for all models"))
    results = []
    flat = pairs.reset_index()
    keep = pd.Series([complete.loc[(r.gift_id, r.metric)] for r in flat.itertuples()], index=flat.index)
    for system, column in [("BenchOpt", "benchopt_value"), ("GIFT-Eval", "gifteval_value")]:
        observations = flat.rename(columns={"gift_id": "dataset_id", "solver_family": "model_id"}).copy()
        observations["value"] = flat[column].where(keep)
        observations["task"] = "forecasting"
        observations["direction"] = "min"
        stats = observations.groupby(["task", "metric", "model_id"], as_index=False).agg(
            mean_value=("value", "mean"),
            std_value=("value", lambda values: values.std(ddof=0)),
        )
        stats["system"] = system
        stats["position"] = stats.groupby("metric")["mean_value"].rank(method="min")
        results.append(stats)
    summary = pd.concat(results, ignore_index=True)
    means = summary.pivot(index=["metric", "model_id"], columns="system", values="mean_value")
    differences = (means["BenchOpt"] - means["GIFT-Eval"]).rename("mean_difference")
    summary = summary.merge(differences, on=["metric", "model_id"], validate="many_to_one")
    return (summary,
            pd.DataFrame(exclusions, columns=["gift_id", "metric", "missing_pair_models", "reason"]), panel_ids)


def read_inputs(artifacts, bench_file, gift_repo):
    """Validate model identity and the upstream alignment, retaining old files."""
    names = ["raw_comparison_long.csv", "metric_equivalence_detailed.csv",
             "metric_mapping.csv", "common_datasets.txt", "dataset_alignment.csv"]
    paths = [artifacts / n for n in names] + [bench_file]
    raw = pd.read_csv(paths[0])
    detailed = pd.read_csv(paths[1])
    bench = pd.read_parquet(bench_file)
    gift = bench.loc[bench.dataset_name.astype(str).str.startswith("GiftEval")].copy()
    gift["gift_id"] = gift.p_dataset_dataset_name.astype(str) + "/" + gift.p_dataset_term.astype(str)
    configurations = {}
    ref_sets = []
    for model, reference in MODELS.items():
        subset = gift.loc[gift.solver_name.astype(str).str.startswith(model + "[")]
        configs = subset.solver_name.dropna().unique()
        if len(configs) != 1:
            raise ValueError(f"{model}: expected one explicit configuration, found {len(configs)}")
        if subset.duplicated("gift_id").any():
            raise ValueError(f"{model}: repeated local configurations; aggregate repetitions explicitly upstream")
        configurations[model] = str(configs[0])
        detail_configs = set(detailed.loc[detailed.solver_family.eq(model), "solver_name"])
        if detail_configs - {configs[0]}:
            raise ValueError(f"{model}: detailed comparison has inconsistent model configurations")
        ref_file = gift_repo / "results" / reference / "all_results.csv"
        paths.append(ref_file)
        ref = pd.read_csv(ref_file)
        if ref.dataset.isna().any() or ref.dataset.duplicated().any():
            raise ValueError(f"Invalid reference identifiers: {reference}")
        ref_sets.append(set(ref.dataset))
    if any(s != ref_sets[0] for s in ref_sets):
        raise ValueError("Reference dataset universes differ; resolve the aligned scope explicitly")
    aligned = set(paths[3].read_text().splitlines()) - {""}
    expected = set(gift.gift_id) & ref_sets[0]
    alignment = pd.read_csv(paths[4]).iloc[0]
    if (aligned != expected or int(alignment['common']) != len(aligned)
            or int(alignment.gifteval_total) != len(ref_sets[0])):
        raise ValueError("Alignment artifacts are stale; rerun validation steps 01–06")
    # Step 06 must preserve step 05's finite paired values and identities.
    key = ["solver_family", "gift_id", "benchopt_metric"]
    cols = key + ["benchopt_value", "gifteval_value"]
    left = raw[cols].sort_values(key).reset_index(drop=True)
    right = detailed[cols].sort_values(key).reset_index(drop=True)
    # CSV reserialization in step 06 can change the last floating-point bits.
    # This tolerance checks provenance only; ranking still uses exact ties.
    pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=False,
                                  rtol=1e-12, atol=1e-14)
    mapping = pd.read_csv(paths[2])
    for label, metric in METRICS.items():
        match = mapping.loc[mapping.benchopt_metric.eq(metric)]
        if len(match) != 1 or pd.isna(match.iloc[0].gifteval_metric):
            raise ValueError(f"Missing metric mapping: {metric}")
        rows = raw.loc[raw.metric.eq(label)]
        if not (rows.benchopt_metric.eq(metric).all()
                and rows.gifteval_metric.eq(match.iloc[0].gifteval_metric).all()):
            raise ValueError(f"Inconsistent metric mapping: {metric}")
    if not raw.gifteval_reference.eq(raw.solver_family.map(MODELS)).all():
        raise ValueError("Inconsistent reference model mapping")
    return raw, aligned, configurations, len(ref_sets[0]), paths


def write_statistics(artifacts=ARTIFACT_DIR, bench_file=BENCH_FILE, gift_repo=GIFT_REPO):
    raw, ids, configurations, reference_count, inputs = read_inputs(artifacts, bench_file, gift_repo)
    stats, exclusions, panels = build_statistics(raw, ids)
    artifacts.mkdir(parents=True, exist_ok=True)
    stats.to_csv(artifacts / SUMMARY, index=False)
    exclusions.to_csv(artifacts / EXCLUSIONS, index=False)
    scope = dict(schema_version=2, unit="dataset_configuration", std_ddof=0,
                 ranking="mean_value_ascending", position_ties="min_exact",
                 difference="BenchOpt_mean_minus_GIFT_Eval_mean", panel="complete_all_models_both_systems",
                 aligned_count=len(ids), reference_count=reference_count, aligned_ids=sorted(ids),
                 models=configurations, panel_ids=panels,
                 source_hashes={str(p.resolve()): digest(p) for p in inputs},
                 summary_sha256=digest(artifacts / SUMMARY),
                 exclusions_sha256=digest(artifacts / EXCLUSIONS))
    (artifacts / SCOPE).write_text(json.dumps(scope, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return stats, scope


def load_statistics(artifacts=ARTIFACT_DIR):
    scope = json.loads((artifacts / SCOPE).read_text(encoding="utf-8"))
    if (scope.get("schema_version") != 2 or scope["summary_sha256"] != digest(artifacts / SUMMARY)
            or scope["exclusions_sha256"] != digest(artifacts / EXCLUSIONS)
            or any(not Path(p).is_file() or digest(p) != sha for p, sha in scope["source_hashes"].items())):
        raise ValueError("Stale global GIFT-Eval statistics; rerun global_statistics.py")
    return pd.read_csv(artifacts / SUMMARY), scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--bench-file", type=Path, default=BENCH_FILE)
    parser.add_argument("--gift-repo", type=Path, default=GIFT_REPO)
    args = parser.parse_args()
    stats, scope = write_statistics(args.artifacts, args.bench_file, args.gift_repo)
    print(f"Global statistics: {len(stats)} rows; {scope['aligned_count']}/{scope['reference_count']} aligned")
    print({metric: len(ids) for metric, ids in scope['panel_ids'].items()})


if __name__ == "__main__":
    main()
