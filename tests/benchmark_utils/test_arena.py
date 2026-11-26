"""Check the arena page data on a small synthetic results table."""

import json

import numpy as np
import pandas as pd

from benchmark_utils.arena import build_arena


def _results():
    rng = np.random.default_rng(0)
    # Skill of each solver: "Best" always wins, "Worst" always loses.
    skill = {"Best": 0.0, "SeasonalNaive": 1.0, "Worst": 2.0, "ClfOnly": 0.5}
    tasks = {
        "objective_wql": ["F1", "F2", "F3"],
        "objective_balanced_accuracy": ["C1", "C2", "C3"],
    }
    rows = []
    for col, datasets in tasks.items():
        for dataset in datasets:
            for solver, s in skill.items():
                if solver == "ClfOnly" and col == "objective_wql":
                    continue
                loss = 1 + s + rng.uniform(0, 0.1)
                rows.append(
                    {
                        "solver_name": solver,
                        "dataset_name": dataset,
                        "time": 1.0 + s,
                        # Accuracy is higher-is-better: flip the loss.
                        col: loss if col == "objective_wql" else 1 / loss,
                    }
                )
    return pd.DataFrame(rows)


def test_build_arena():
    arena = build_arena(_results())
    json.dumps(arena, allow_nan=False)  # embedded as JSON in the page

    boards = {b["key"]: b for b in arena["boards"]}
    assert list(boards) == ["overall", "forecasting", "classification"]
    assert len(boards["overall"]["datasets"]) == 6
    # ClfOnly did not run every task: it is only ranked on its own.
    assert "ClfOnly" not in [m["name"] for m in boards["overall"]["models"]]

    for board in boards.values():
        names = [m["name"] for m in board["models"]]
        assert names[0] == "Best" and names[-1] == "Worst"
        assert board["anchor"] == "SeasonalNaive"
        anchor = board["models"][names.index("SeasonalNaive")]
        assert anchor["elo"] == 1000
        # Row i, column j: share of datasets where i beats j.
        wins = np.array([m["wins"] for m in board["models"]])
        off_diagonal = ~np.eye(len(names), dtype=bool)
        assert np.allclose((wins + wins.T)[off_diagonal], 1)

    # Forecasting metrics are relative to the baseline, lower is better.
    forecasting = boards["forecasting"]
    assert forecasting["columns"][0] == {"label": "rWQL", "higher": False}
    rwql = {m["name"]: m["metrics"][0] for m in forecasting["models"]}
    assert rwql["SeasonalNaive"] == 1 and rwql["Best"] < 1 < rwql["Worst"]
