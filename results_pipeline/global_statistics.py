"""Dataset Explorer statistics adapter and standalone CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

# Support both the existing direct-script runners and python -m invocation.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_utils.global_statistics import aggregate_five_statistics
from results_pipeline.leaderboard import METRIC_DIRECTION, METRIC_LABELS


def source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_dataset_statistics(source: pd.DataFrame) -> pd.DataFrame:
    required = {"task", "dataset_name", "solver_name", "metric", "metric_value"}
    if missing := required.difference(source.columns):
        raise ValueError(f"Missing Dataset Explorer columns: {sorted(missing)}")
    source = source.copy()
    directions = source["metric"].map(METRIC_DIRECTION)
    if directions.isna().any():
        raise ValueError("Unknown metric: an explicit min/max direction is required")
    if "direction" in source and not source["direction"].eq(directions).all():
        raise ValueError("Source metric directions disagree with the existing policy")
    source["direction"] = directions
    canonical = source.rename(columns={
        "dataset_name": "dataset_id", "solver_name": "model_id", "metric_value": "value",
    })
    result = aggregate_five_statistics(canonical)
    if "solver_display" not in source:
        source["solver_display"] = source["solver_name"]
    labels = source[["task", "solver_name", "solver_display"]].drop_duplicates()
    if labels.duplicated(["task", "solver_name"]).any():
        raise ValueError("Conflicting model display labels")
    result = result.rename(columns={"model_id": "solver_name"}).merge(
        labels, on=["task", "solver_name"], how="left", validate="many_to_one",
    )
    result["metric_label"] = result["metric"].map(METRIC_LABELS)
    return result


def write_statistics(input_file: Path, output_file: Path) -> pd.DataFrame:
    frame = build_dataset_statistics(pd.read_csv(input_file))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_file, index=False)
    metadata = {
        "schema_version": 1, "source_sha256": source_digest(input_file),
        "summary_sha256": source_digest(output_file),
        "unit": "dataset_configuration", "std_ddof": 0,
        "rank_ties": "average_exact", "win_ties": "fractional_exact",
        "win_denominator": "valid_model_metric_configurations",
        "coverage_denominator": "all_task_configurations_in_source",
        "rates": "fractions_0_to_1", "invalid_values": "excluded_never_zero",
    }
    output_file.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return frame


def load_statistics(input_file: Path, summary_file: Path) -> pd.DataFrame:
    metadata = json.loads(summary_file.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("schema_version") != 1
            or metadata.get("source_sha256") != source_digest(input_file)
            or metadata.get("summary_sha256") != source_digest(summary_file)):
        raise ValueError("Global summary is stale or modified; regenerate it from the current input")
    return pd.read_csv(summary_file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    frame = write_statistics(args.input_file, args.output)
    print(f"Global statistics: {len(frame)} rows -> {args.output}")


if __name__ == "__main__":
    main()
