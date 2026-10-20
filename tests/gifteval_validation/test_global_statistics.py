"""Strict panel, independent ranks, missingness and additive rendering tests."""
import math
import unittest
import tempfile
from pathlib import Path

import pandas as pd

from gifteval_validation.global_statistics import (
    build_statistics, write_statistics, load_statistics, MODELS, METRICS,
)
from gifteval_validation.global_report import render_global_report


def fixture():
    rows = []
    # Opposite winners across systems; an exact tie on d2.
    for model, bench, ref in [("A", [2, 4, 6], [4, 4, 3]),
                              ("B", [4, 4, 3], [2, 4, 6])]:
        for i in range(3):
            rows.append(dict(solver_family=model, gift_id=f"d{i}", metric="MAE",
                             benchopt_value=bench[i], gifteval_value=ref[i]))
    return pd.DataFrame(rows).astype({"benchopt_value": float, "gifteval_value": float})


def compute(frame):
    return build_statistics(frame, ["d0", "d1", "d2"], models=("A", "B"), metrics=("MAE",))


class GiftGlobalTests(unittest.TestCase):
    def test_positions_use_metric_mean_not_mean_rank(self):
        frame = fixture()
        frame.loc[frame.solver_family.eq("A"), "benchopt_value"] = [1., 1., 100.]
        frame.loc[frame.solver_family.eq("B"), "benchopt_value"] = [2., 2., 3.]
        stats, _, _ = compute(frame)
        bench = stats.loc[stats.system.eq("BenchOpt")].set_index("model_id")
        # A wins two datasets but its mean error is larger: it must rank second.
        self.assertEqual(bench.loc["A", "position"], 2)
        self.assertEqual(bench.loc["B", "position"], 1)

    def test_means_std_difference_and_value_positions(self):
        stats, exclusions, panels = compute(fixture())
        a = stats.loc[stats.model_id.eq("A")].set_index("system")
        self.assertEqual(a.loc["BenchOpt", "mean_value"], 4)
        self.assertAlmostEqual(a.loc["BenchOpt", "std_value"], math.sqrt(8 / 3))
        self.assertAlmostEqual(a.loc["GIFT-Eval", "mean_value"], 11 / 3)
        self.assertEqual(a.loc["BenchOpt", "position"], 2)
        self.assertEqual(a.loc["GIFT-Eval", "position"], 1)
        self.assertAlmostEqual(a.loc["BenchOpt", "mean_difference"], 1 / 3)
        self.assertFalse({"mean_rank", "win_rate", "coverage", "n_valid", "n_total"} & set(stats.columns))
        self.assertEqual(panels["MAE"], ["d0", "d1", "d2"])
        self.assertTrue(exclusions.empty)

    def test_one_sided_missing_removes_configuration_from_every_model(self):
        frame = fixture()
        frame.loc[0, "gifteval_value"] = float("nan")
        stats, exclusions, panels = compute(frame)
        self.assertEqual(panels["MAE"], ["d1", "d2"])
        self.assertEqual(exclusions.iloc[0].missing_pair_models, "A")
        a = stats.loc[stats.model_id.eq("A")].set_index("system")
        self.assertEqual(a.loc["BenchOpt", "mean_value"], 5)
        self.assertEqual(a.loc["GIFT-Eval", "mean_value"], 3.5)
        self.assertEqual(a.loc["BenchOpt", "position"], 2)
        self.assertEqual(a.loc["GIFT-Eval", "position"], 1)

    def test_missing_row_and_nonfinite_value(self):
        frame = fixture().drop(index=0)
        frame.loc[1, "benchopt_value"] = float("inf")
        stats, _, panels = compute(frame)
        self.assertEqual(panels["MAE"], ["d2"])
        self.assertTrue(stats.std_value.eq(0).all())

    def test_model_entirely_absent_empty_panel_not_zero_scores(self):
        frame = fixture().loc[lambda x: x.solver_family.eq("A")]
        stats, _, panels = compute(frame)
        self.assertEqual(panels["MAE"], [])
        self.assertTrue(stats[["mean_value", "std_value", "mean_difference", "position"]].isna().all().all())

    def test_duplicates_and_unknown_scope_rejected(self):
        frame = fixture()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compute(pd.concat([frame, frame.iloc[:1]], ignore_index=True))
        frame.loc[0, "gift_id"] = "outside"
        with self.assertRaisesRegex(ValueError, "scope"):
            compute(frame)

    def test_metrics_do_not_share_panels(self):
        frame = fixture()
        other = frame.assign(metric="MSE")
        other.loc[0, "benchopt_value"] = None
        stats, _, panels = build_statistics(pd.concat([frame, other], ignore_index=True),
                                            ["d0", "d1", "d2"], ("A", "B"), ("MAE", "MSE"))
        self.assertEqual(len(panels["MAE"]), 3)
        self.assertEqual(len(panels["MSE"]), 2)

    def test_html_escapes_labels_and_documents_scope(self):
        frame = fixture().replace({"solver_family": {"A": "A<&"}})
        stats, _, panels = build_statistics(frame, ["d0", "d1", "d2"], ("A<&", "B"), ("MAE",))
        page = render_global_report(stats, dict(panel_ids=panels, models={"A<&": "config", "B": "config"},
                                               aligned_count=3, reference_count=4))
        self.assertIn("A&lt;&amp;", page)
        self.assertNotIn("A<&", page)
        self.assertIn("3/4 configurations", page)
        self.assertIn('id="global-panel"', page)
        self.assertNotIn("Victoires partagées", page)

    def test_unified_table_order_positions_and_signed_differences(self):
        from html.parser import HTMLParser

        class TableRows(HTMLParser):
            def __init__(self):
                super().__init__()
                self.rows, self.row, self.cell = [], None, None

            def handle_starttag(self, tag, attrs):
                if tag == "tr":
                    self.row = []
                elif tag in ("td", "th"):
                    self.cell = ""

            def handle_data(self, data):
                if self.cell is not None:
                    self.cell += data

            def handle_endtag(self, tag):
                if tag in ("td", "th"):
                    self.row.append(self.cell)
                    self.cell = None
                elif tag == "tr":
                    self.rows.append(self.row)

        frame = fixture()
        frame.loc[0, "gifteval_value"] = float("nan")
        stats, _, panels = compute(frame)
        scope = dict(panel_ids=panels, models={"A": "a", "B": "b"},
                     aligned_count=3, reference_count=4)
        page = render_global_report(stats, scope)
        self.assertEqual(page.count("<table>"), 1)
        self.assertNotIn('id="gift-global-model"', page)
        parser = TableRows()
        parser.feed(page)
        b, a = parser.rows[1:]
        self.assertEqual(b[:3], ["1", "2", "B"])
        self.assertEqual(a[:3], ["2", "1", "A"])
        self.assertEqual(len(a), 8)
        self.assertEqual(a[3:6], ["5", "3.5", "+1.5 (+42.86 %)"])
        self.assertIn('class="gift-global-delta gift-global-better"', page)
        self.assertIn('class="gift-global-delta gift-global-worse"', page)
        self.assertEqual(a[6:8], ["1", "0.5"])
        self.assertNotIn("Rang moyen", page)
        self.assertNotIn("Couverture", page)

    def test_empty_panel_renders_undefined_differences(self):
        stats, _, panels = compute(fixture().iloc[:0])
        page = render_global_report(stats, dict(panel_ids=panels, models={"A": "a", "B": "b"},
                                               aligned_count=3, reference_count=4))
        self.assertIn('<td class="gift-global-delta">—</td>', page)
        self.assertIn("Panel commun : 0/3", page)

    def test_zero_reference_and_equal_values_display(self):
        frame = fixture()
        frame["gifteval_value"] = 0.0
        stats, _, panels = compute(frame)
        scope = dict(panel_ids=panels, models={"A": "a", "B": "b"},
                     aligned_count=3, reference_count=4)
        page = render_global_report(stats, scope)
        self.assertIn("(% non défini)", page)
        frame["gifteval_value"] = frame["benchopt_value"]
        stats, _, _ = compute(frame)
        page = render_global_report(stats, scope)
        self.assertIn('<td class="gift-global-delta">+0 (+0.00 %)</td>', page)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir()
        self.bench = self.root / "bench.parquet"
        self.refs = self.root / "references"
        bench, raw, detailed = [], [], []
        for model, reference in MODELS.items():
            refdir = self.refs / "results" / reference
            refdir.mkdir(parents=True)
            pd.DataFrame({"dataset": ["d/short"]}).to_csv(refdir / "all_results.csv", index=False)
            bench.append(dict(dataset_name="GiftEval[d]", solver_name=model + "[v1]",
                              p_dataset_dataset_name="d", p_dataset_term="short"))
            for label, metric in METRICS.items():
                row = dict(solver_family=model, gift_id="d/short", metric=label,
                           benchopt_metric=metric, gifteval_metric=label,
                           gifteval_reference=reference, benchopt_value=2., gifteval_value=3.)
                raw.append(row)
                detailed.append(dict(row, solver_name=model + "[v1]"))
        pd.DataFrame(bench).to_parquet(self.bench)
        pd.DataFrame(raw).to_csv(self.artifacts / "raw_comparison_long.csv", index=False)
        pd.DataFrame(detailed).to_csv(self.artifacts / "metric_equivalence_detailed.csv", index=False)
        pd.DataFrame([dict(benchopt_metric=m, gifteval_metric=l) for l, m in METRICS.items()]).to_csv(
            self.artifacts / "metric_mapping.csv", index=False)
        (self.artifacts / "common_datasets.txt").write_text("d/short\n")
        pd.DataFrame([dict(common=1, gifteval_total=1)]).to_csv(self.artifacts / "dataset_alignment.csv", index=False)

    def test_roundtrip_and_stale_sources(self):
        stats, scope = write_statistics(self.artifacts, self.bench, self.refs)
        self.assertEqual(len(stats), 40)
        self.assertTrue(stats.position.eq(1).all())
        self.assertEqual(len(load_statistics(self.artifacts)[0]), 40)
        (self.artifacts / "common_datasets.txt").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "Stale"):
            load_statistics(self.artifacts)

    def test_multiple_model_configurations_rejected(self):
        bench = pd.read_parquet(self.bench)
        extra = bench.iloc[[0]].copy()
        extra["solver_name"] = "Chronos[v2]"
        pd.concat([bench, extra], ignore_index=True).to_parquet(self.bench)
        with self.assertRaisesRegex(ValueError, "one explicit configuration"):
            write_statistics(self.artifacts, self.bench, self.refs)

    def test_mismatched_reference_scope_rejected(self):
        pd.DataFrame({"dataset": ["other/short"]}).to_csv(
            self.refs / "results/naive/all_results.csv", index=False)
        with self.assertRaisesRegex(ValueError, "universes differ"):
            write_statistics(self.artifacts, self.bench, self.refs)


if __name__ == "__main__":
    unittest.main()
