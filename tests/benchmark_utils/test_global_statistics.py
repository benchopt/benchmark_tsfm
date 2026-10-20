"""Hand-computable examples for the five global statistics."""
import math
import unittest

import pandas as pd

from benchmark_utils.global_statistics import aggregate_five_statistics


def observations(values, direction="min", metric="mae", task="forecasting"):
    return pd.DataFrame([
        dict(task=task, dataset_id=f"d{i}", model_id=model,
             metric=metric, value=value, direction=direction)
        for model, sequence in values.items()
        for i, value in enumerate(sequence)
    ])


class GlobalStatisticsTests(unittest.TestCase):
    def test_known_values_ranks_and_fractional_wins(self):
        result = aggregate_five_statistics(observations({"A": [2, 4, 6], "B": [4, 4, 3]}))
        a = result.set_index("model_id").loc["A"]
        self.assertEqual(a.mean_value, 4)
        self.assertAlmostEqual(a.std_value, math.sqrt(8 / 3))
        self.assertEqual(a.mean_rank, 1.5)
        self.assertEqual(a.win_rate, .5)
        self.assertEqual((a.n_valid, a.n_total, a.coverage), (3, 3, 1))

    def test_missing_values_and_absent_model_are_not_defeats(self):
        frame = observations({"A": [1, None, float("inf")], "B": [2, 2, 2]})
        frame = frame.loc[~(frame.model_id.eq("A") & frame.dataset_id.eq("d2"))]
        a = aggregate_five_statistics(frame).set_index("model_id").loc["A"]
        self.assertEqual((a.n_valid, a.n_total), (1, 3))
        self.assertEqual(a.win_rate, 1)
        self.assertEqual(a.std_value, 0)
        self.assertAlmostEqual(a.coverage, 1 / 3)

    def test_zero_valid_values_and_task_universe(self):
        frame = observations({"A": [None, "bad"], "B": [None, None]})
        other = observations({"A": [1, 2, 3]}, metric="mse")
        result = aggregate_five_statistics(pd.concat([frame, other], ignore_index=True))
        a = result.loc[result.model_id.eq("A") & result.metric.eq("mae")].iloc[0]
        self.assertEqual((a.n_valid, a.n_total, a.coverage), (0, 3, 0))
        self.assertTrue(all(pd.isna(a[k]) for k in ["mean_value", "std_value", "mean_rank", "win_rate"]))
        self.assertEqual(len(result), 4)  # B also appears for its absent MSE.

    def test_maximization_and_exact_ties(self):
        result = aggregate_five_statistics(observations({"A": [3], "B": [3], "C": [1]}, "max"))
        a = result.set_index("model_id").loc["A"]
        self.assertEqual((a.mean_rank, a.win_rate), (1.5, .5))

    def test_distinct_configurations_and_duplicate_rejection(self):
        frame = observations({"Seasonal[1]": [2], "Seasonal[7]": [3]})
        self.assertEqual(len(aggregate_five_statistics(frame)), 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate_five_statistics(pd.concat([frame, frame], ignore_index=True))

    def test_cached_ranks_checked_and_no_task_mixing(self):
        frame = observations({"A": [1], "B": [2]})
        frame["scientific_rank"] = [2, 1]
        with self.assertRaisesRegex(ValueError, "scientific_rank"):
            aggregate_five_statistics(frame)
        frame = observations({"A": [1]}, task="one")
        other = observations({"A": [100]}, task="two")
        result = aggregate_five_statistics(pd.concat([frame, other], ignore_index=True))
        self.assertEqual(result.mean_value.tolist(), [1, 100])

    def test_rank_mean_is_not_rank_of_mean(self):
        result = aggregate_five_statistics(observations({"A": [1, 1, 100], "B": [2, 2, 3]}))
        a, b = result.set_index("model_id").loc[["A", "B"]].to_dict("records")
        self.assertGreater(a["mean_value"], b["mean_value"])
        self.assertLess(a["mean_rank"], b["mean_rank"])

    def test_empty_input_and_unknown_direction(self):
        frame = observations({"A": [1]})
        self.assertTrue(aggregate_five_statistics(frame.iloc[:0]).empty)
        frame["direction"] = "unknown"
        with self.assertRaises(ValueError):
            aggregate_five_statistics(frame)


if __name__ == "__main__":
    unittest.main()
