"""Dataset-specific identity, artifact provenance and additive rendering."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from results_pipeline.global_statistics import (
    build_dataset_statistics, load_statistics, write_statistics,
)
from results_pipeline.dataset_reports import build_global_summary_section, build_task_index


def source():
    return pd.DataFrame([
        dict(task="forecasting", dataset_name=f"D[{i}]", solver_name=model,
             solver_display=model, metric="objective_mae", metric_value=value)
        for model, values in {"A<&": [2, 4, 6], "B": [4, 4, 3]}.items()
        for i, value in enumerate(values)
    ])


class DatasetStatisticsTests(unittest.TestCase):
    def test_source_unchanged_and_display_escaped(self):
        frame = source()
        before = frame.copy(deep=True)
        stats = build_dataset_statistics(frame)
        pd.testing.assert_frame_equal(frame, before)
        section = build_global_summary_section(stats)
        self.assertIn("A&lt;&amp;", section)
        self.assertNotIn("A<&", section)
        self.assertIn("50.00 %", section)
        self.assertIn("3/3", section)

    def test_artifact_roundtrip_and_stale_input_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_file, output = Path(tmp) / "input.csv", Path(tmp) / "stats.csv"
            source().to_csv(input_file, index=False)
            write_statistics(input_file, output)
            self.assertEqual(len(load_statistics(input_file, output)), 2)
            input_file.write_text(input_file.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "stale"):
                load_statistics(input_file, output)

    def test_summary_is_additive_before_dataset_cards(self):
        cards = [dict(relative_report="d/report.html", dataset_name="D", n_solvers=2, n_metrics=1)]
        before = build_task_index("forecasting", cards, "../index.html")
        section = build_global_summary_section(build_dataset_statistics(source()))
        after = build_task_index("forecasting", cards, "../index.html", section)
        self.assertEqual(after.replace(section, ""), before)
        self.assertLess(after.index('id="global-performance-summary"'), after.index('class="dataset-grid"'))
        self.assertIn('href="d/report.html"', after)

    def test_unknown_metric_and_conflicting_configuration_label(self):
        frame = source()
        frame.loc[0, "metric"] = "unknown"
        with self.assertRaises(ValueError):
            build_dataset_statistics(frame)
        frame = source()
        frame.loc[0, "solver_display"] = "different label"
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            build_dataset_statistics(frame)

    def test_positions_follow_mean_rank_with_ties_and_missing_last(self):
        from html.parser import HTMLParser

        class Rows(HTMLParser):
            def __init__(self):
                super().__init__()
                self.rows = []
                self.current = None
                self.cell = None

            def handle_starttag(self, tag, attrs):
                if tag == "tr":
                    self.current = []
                if tag in ("td", "th"):
                    self.cell = ""

            def handle_data(self, data):
                if self.cell is not None:
                    self.cell += data

            def handle_endtag(self, tag):
                if tag in ("td", "th"):
                    self.current.append(self.cell)
                    self.cell = None
                if tag == "tr":
                    self.rows.append(self.current)

        stats = build_dataset_statistics(source())
        prototype = stats.iloc[0].to_dict()
        rows = []
        for name, rank in [("A", 4), ("Z", 1.5), ("Y", 1.5), ("B", float("nan"))]:
            rows.append(dict(prototype, solver_name=name, solver_display=name, mean_rank=rank))
        parser = Rows()
        parser.feed(build_global_summary_section(pd.DataFrame(rows)))
        self.assertEqual([r[:2] for r in parser.rows[1:]],
                         [["1", "Y"], ["1", "Z"], ["3", "A"], ["—", "B"]])


class PipelineRegressionTests(unittest.TestCase):
    def test_exact_winners_agree_between_pipeline_and_html(self):
        from results_pipeline.leaderboard import rank_dataset_results
        from results_pipeline.dataset_reports import build_metric_ranking
        frame = pd.DataFrame([
            dict(task="forecasting", dataset_name="D", solver_name=name,
                 solver_short=name, solver_display=name, metric_value=value)
            for name, value in [("A", 1.0), ("B", 1.0 + 5e-11)]
        ])
        ranked = rank_dataset_results(frame, "objective_mae", "solver_name")
        self.assertEqual(ranked["fractional_win"].tolist(), [1, 0])
        detail = build_metric_ranking(ranked, "objective_mae")
        self.assertEqual(detail["is_winner"].tolist(), [True, False])
        self.assertEqual(detail["position"].tolist(), [1, 2])

    def test_explicit_metric_direction_and_task_isolation(self):
        from results_pipeline.scientific_leaderboard import recompute_ranks
        from results_pipeline.leaderboard import rank_dataset_results
        frame = pd.DataFrame([
            dict(task=task, dataset_name="same", solver_name=model,
                 solver_display=model, metric="objective_accuracy", metric_value=value,
                 rank=1)
            for task, model, value in [("one", "A", .9), ("one", "B", .8),
                                       ("two", "A", .1), ("two", "B", .2)]
        ])
        # Deliberately uninformative old ranks must not reverse a max metric.
        ranks = recompute_ranks(frame)
        self.assertEqual(ranks.scientific_rank.tolist(), [1, 2, 2, 1])
        upstream = rank_dataset_results(frame, "objective_accuracy", "solver_name")
        self.assertEqual(upstream["rank"].tolist(), [1, 2, 2, 1])

    def test_diagnostic_coverage_is_per_task(self):
        from results_pipeline.validate_leaderboard import build_solver_coverage
        frame = pd.DataFrame([
            dict(task=task, dataset_name=dataset, dataset_short=dataset,
                 solver_name="A", solver_short="A", objective_mae=1.)
            for task, dataset in [("one", "a"), ("two", "b"), ("two", "c")]
        ])
        coverage = build_solver_coverage(frame, ["objective_mae"])
        self.assertEqual(coverage["coverage"].tolist(), [1., 1.])
        self.assertEqual(coverage["metric_total_datasets"].tolist(), [1, 2])

    def test_analysis_rejects_nonfinite_only_results(self):
        from results_pipeline.analysis import build_result_mask, detect_task_from_metrics
        frame = pd.DataFrame({"objective_mae": [None, float("inf"), "bad", 0.]})
        self.assertEqual(build_result_mask(frame).tolist(), [False, False, False, True])
        self.assertEqual(detect_task_from_metrics(frame.iloc[1]), "unknown")

    def test_dataset_loader_rejects_duplicates_and_filters_infinity(self):
        from results_pipeline.dataset_reports import load_results
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.csv"
            frame = source().astype({"metric_value": float}).assign(
                solver_short=lambda x: x.solver_name)
            frame.loc[0, "metric_value"] = float("inf")
            frame.to_csv(path, index=False)
            self.assertEqual(len(load_results(path)), 5)
            pd.concat([frame, frame]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_results(path)

    def test_analysis_without_timing_column(self):
        import subprocess
        import sys
        from results_pipeline import analysis
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame([dict(dataset_name="GiftEval[test]", solver_name="A",
                               objective_mae=1.)]).to_parquet(root / "input.parquet")
            result = subprocess.run(
                [sys.executable, analysis.__file__, str(root / "input.parquet"),
                 "--output-dir", str(root / "results")],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            values = pd.read_parquet(root / "results/benchmark_tsfm_results_only.parquet")
            self.assertEqual(len(values), 1)
            self.assertTrue(values["time"].isna().all())

    def test_scientific_loader_rejects_duplicate_configurations(self):
        from results_pipeline.scientific_leaderboard import load_configuration_ranks
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.csv"
            frame = source().assign(solver_short=lambda x: x.solver_name,
                                    metric_label="MAE", rank=1, fractional_win=1)
            pd.concat([frame, frame]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_configuration_ranks(path)


if __name__ == "__main__":
    unittest.main()
