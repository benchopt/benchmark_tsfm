#!/usr/bin/env python3

"""Merge BenchOpt TSFM result files into a single Parquet file.

The benchmark results are currently produced in two groups:

1. the standard TSFM benchmark results;
2. the GIFT-Eval benchmark results.

This script concatenates both result tables while preserving the union
of their columns. Missing columns are represented by null values.

Example
-------
python results_pipeline/merge_results.py \
    outputs/benchmark_tsfm_standard.parquet \
    outputs/gifteval_all.parquet \
    --output results/benchmark_tsfm_complete.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge multiple BenchOpt TSFM Parquet result files into "
            "one consolidated Parquet file."
        )
    )

    parser.add_argument(
        "input_files",
        nargs="+",
        help="Parquet result files to merge, in concatenation order.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path of the consolidated output Parquet file.",
    )

    return parser.parse_args()


def merge_results(input_paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate BenchOpt result files."""

    frames: list[pd.DataFrame] = []

    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Input file does not exist: {path}"
            )

        if path.suffix.lower() != ".parquet":
            raise ValueError(
                f"Expected a Parquet file, got: {path}"
            )

        frame = pd.read_parquet(path)

        print(
            f"[load] {path} "
            f"-> {len(frame)} rows, {len(frame.columns)} columns"
        )

        frames.append(frame)

    if not frames:
        raise ValueError("No input result file was provided.")

    merged = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    return merged


def main() -> None:
    args = parse_args()

    input_paths = [
        Path(path).expanduser().resolve()
        for path in args.input_files
    ]

    output_path = (
        Path(args.output)
        .expanduser()
        .resolve()
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=== TSFM result merge ===")
    print(f"Number of input files: {len(input_paths)}")

    merged = merge_results(input_paths)

    print(
        f"[merge] {len(merged)} rows, "
        f"{len(merged.columns)} columns"
    )

    merged.to_parquet(
        output_path,
        index=False,
    )

    print(f"[write] {output_path}")
    print("Merge completed successfully.")


if __name__ == "__main__":
    main()
