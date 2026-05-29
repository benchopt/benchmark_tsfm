import io
import numpy as np
import matplotlib.pyplot as plt

from benchopt import BasePlot


class Plot(BasePlot):
    name = "elo"
    type = "image"
    options = {}
    BOOTSTRAP_ROUNDS = 200
    CI_LOW = 5
    CI_HIGH = 95
    RANDOM_SEED = 0

    _TASK_METRICS = {
        "forecasting": ("objective_wql", True),
        "classification": ("objective_balanced_accuracy", False),
        "anomaly_detection": ("objective_auc_pr", False),
    }

    def _expected_score(self, rating_a, rating_b):
        return 1.0 / (1.0 + np.power(10.0, (rating_b - rating_a) / 400.0))

    def _resolve_task_metric(self, df_dataset):
        candidates = []
        for task_name, (metric_col, minimize) in self._TASK_METRICS.items():
            if metric_col in df_dataset.columns and df_dataset[metric_col].notna().any():
                candidates.append((task_name, metric_col, minimize))

        if len(candidates) == 0:
            dataset_name = str(df_dataset["dataset_name"].iloc[0])
            raise ValueError(
                f"Dataset '{dataset_name}' has none of the required metrics: "
                "objective_wql / objective_balanced_accuracy / objective_auc_pr."
            )

        if len(candidates) > 1:
            dataset_name = str(df_dataset["dataset_name"].iloc[0])
            metrics = [c[1] for c in candidates]
            raise ValueError(
                f"Dataset '{dataset_name}' is ambiguous: multiple task metrics present {metrics}."
            )

        return candidates[0]

    def _prepare_dataset_matches(self, df):
        if "dataset_name" not in df.columns or "solver_name" not in df.columns:
            raise ValueError("Input dataframe must contain 'dataset_name' and 'solver_name' columns.")

        solver_names = sorted(df["solver_name"].dropna().unique())
        solver_to_idx = {name: i for i, name in enumerate(solver_names)}
        dataset_matches = []
        n_matches = 0

        for _, df_dataset in df.groupby("dataset_name"):
            _, metric_col, minimize = self._resolve_task_metric(df_dataset)

            rows = df_dataset[["solver_name", metric_col]].dropna(subset=[metric_col])
            if rows.empty:
                dataset_name = str(df_dataset["dataset_name"].iloc[0])
                raise ValueError(
                    f"Dataset '{dataset_name}' has metric column '{metric_col}' but no non-null values."
                )

            duplicated = rows["solver_name"].duplicated(keep=False)
            if duplicated.any():
                dataset_name = str(df_dataset["dataset_name"].iloc[0])
                dup_solvers = sorted(rows.loc[duplicated, "solver_name"].unique().tolist())
                raise ValueError(
                    f"Dataset '{dataset_name}' has multiple rows per solver for metric '{metric_col}'. "
                    f"Duplicated solvers: {dup_solvers}"
                )

            if rows.shape[0] < 2:
                dataset_name = str(df_dataset["dataset_name"].iloc[0])
                raise ValueError(
                    f"Dataset '{dataset_name}' must have at least 2 solvers with non-null '{metric_col}'."
                )

            scores = dict(zip(rows["solver_name"], rows[metric_col]))
            solvers = sorted(scores.keys())
            pair_i = []
            pair_j = []
            outcomes_i = []

            for i in range(len(solvers)):
                for j in range(i + 1, len(solvers)):
                    a = solvers[i]
                    b = solvers[j]
                    sa = scores[a]
                    sb = scores[b]

                    if abs(sa - sb) <= 1e-12:
                        result_a = 0.5
                    else:
                        if minimize:
                            result_a = 1.0 if sa < sb else 0.0
                        else:
                            result_a = 1.0 if sa > sb else 0.0

                    pair_i.append(solver_to_idx[a])
                    pair_j.append(solver_to_idx[b])
                    outcomes_i.append(result_a)
                    n_matches += 1

            dataset_matches.append(
                (
                    np.asarray(pair_i, dtype=np.int64),
                    np.asarray(pair_j, dtype=np.int64),
                    np.asarray(outcomes_i, dtype=np.float64),
                )
            )

        return solver_names, dataset_matches, n_matches

    def _compute_elo_from_matches(
        self, dataset_matches, n_solvers, dataset_indices, rng, initial_rating=1000.0, k_factor=24.0
    ):
        ratings = np.full(n_solvers, float(initial_rating), dtype=np.float64)
        for ds_idx in dataset_indices:
            pair_i, pair_j, outcomes_i = dataset_matches[int(ds_idx)]

            # Shuffle pair order within each dataset so results don't depend on
            # the arbitrary alphabetical pair ordering built in _prepare_dataset_matches.
            perm = rng.permutation(len(pair_i))
            pair_i = pair_i[perm]
            pair_j = pair_j[perm]
            outcomes_i = outcomes_i[perm]

            # Sequential update (order now matters and is randomised above).
            for k in range(len(pair_i)):
                i, j = int(pair_i[k]), int(pair_j[k])
                exp_i = self._expected_score(ratings[i], ratings[j])
                delta = k_factor * (outcomes_i[k] - exp_i)
                ratings[i] += delta
                ratings[j] -= delta

        return ratings

    def _compute_bootstrap_distribution(self, df):
        solver_names, dataset_matches, n_matches = self._prepare_dataset_matches(df)
        n_solvers = len(solver_names)
        n_datasets = len(dataset_matches)
        if n_datasets == 0:
            raise ValueError("No datasets available to compute ELO.")

        rng = np.random.default_rng(self.RANDOM_SEED)

        boot = np.empty((self.BOOTSTRAP_ROUNDS, n_solvers), dtype=np.float64)
        for r in range(self.BOOTSTRAP_ROUNDS):
            sampled = rng.integers(0, n_datasets, size=n_datasets)
            boot[r] = self._compute_elo_from_matches(
                dataset_matches, n_solvers=n_solvers, dataset_indices=sampled, rng=rng
            )

        # Point estimate = mean of the bootstrap distribution.
        # This guarantees the bar is always inside the CI lines, and is the
        # statistically correct "expected ELO" under dataset resampling.
        point_ratings = boot.mean(axis=0)
        ci_low = np.percentile(boot, self.CI_LOW, axis=0)
        ci_high = np.percentile(boot, self.CI_HIGH, axis=0)

        return {
            "solver_names": solver_names,
            "point_ratings": point_ratings,
            "boot_ratings": boot,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_datasets": n_datasets,
            "n_matches": n_matches,
        }

    def _get_bootstrap_distribution(self, df):
        cache_key = (id(df), df.shape)
        cached = getattr(self, "_bootstrap_cache", None)
        if cached is not None and cached.get("key") == cache_key:
            return cached["stats"]

        stats = self._compute_bootstrap_distribution(df)
        self._bootstrap_cache = {"key": cache_key, "stats": stats}
        return stats

    def plot(self, df):
        stats = self._get_bootstrap_distribution(df)
        solver_names = stats["solver_names"]
        point_ratings = stats["point_ratings"]
        ci_low = stats["ci_low"]
        ci_high = stats["ci_high"]
        n_datasets = stats["n_datasets"]

        # Highest ELO first, break ties by solver name.
        order = sorted(
            range(len(solver_names)),
            key=lambda i: (-point_ratings[i], solver_names[i]),
        )

        labels = []
        ratings = []
        lows = []
        highs = []
        colors = []
        for rank, idx in enumerate(order, start=1):
            solver = solver_names[idx]
            labels.append(f"{rank:02d}. {solver}")
            ratings.append(float(point_ratings[idx]))
            lows.append(float(ci_low[idx]))
            highs.append(float(ci_high[idx]))
            colors.append(self.get_style(solver)["color"])

        n = len(labels)
        x = np.arange(n)

        fig, ax = plt.subplots(figsize=(max(6, n * 1.2), 5))

        ax.bar(x, ratings, color=colors, alpha=0.85, zorder=2)

        # Vertical CI lines from low to high.
        for xi, low, high, color in zip(x, lows, highs, colors):
            ax.vlines(xi, low, high, colors=color, linewidth=2.5, zorder=3)
            ax.hlines([low, high], xi - 0.15, xi + 0.15, colors=color, linewidth=1.5, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("ELO rating")
        ax.set_title(
            "Task-aware ELO ranking\n"
            "(forecasting: wql · classification: balanced_accuracy · anomaly: auc_pr)"
        )
        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=1)
        ax.annotate(
            f"datasets={n_datasets}, bootstrap_rounds={self.BOOTSTRAP_ROUNDS}, "
            f"CI={self.CI_LOW}–{self.CI_HIGH}%",
            xy=(0.01, 0.01), xycoords="axes fraction",
            fontsize=7, color="grey",
        )

        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        from PIL import Image
        img_array = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
        plt.close(fig)

        return [{"image": img_array}]

    def get_metadata(self, df):
        stats = self._get_bootstrap_distribution(df)
        return {
            "title": (
                "Task-aware ELO ranking "
                "(forecasting: wql, classification: balanced_accuracy, anomaly: auc_pr)"
            ),
            "ylabel": "ELO rating",
            "grid": True,
            "summary": (
                f"datasets={stats['n_datasets']}, pairwise matches={stats['n_matches']}, "
                f"bootstrap_rounds={self.BOOTSTRAP_ROUNDS}, "
                f"CI={self.CI_LOW}-{self.CI_HIGH}% (vertical lines), "
                "left-to-right sorted by descending point ELO"
            ),
        }
