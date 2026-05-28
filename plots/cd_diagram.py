import matplotlib.pyplot as plt
import numpy as np
import scikit_posthocs as sp
from benchopt import BasePlot


class Plot(BasePlot):
    name = "cd_diagram"
    type = "image"
    options = {
        "objective": ...,
        "objective_column": ...,
    }

    def _figure_to_image_payload(self, fig, label=None):
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        if rgba.ndim == 1:
            width, height = fig.canvas.get_width_height()
            rgba = rgba.reshape((height, width, 4))
        image = rgba[:, :, :3].astype(np.float32) / 255.0
        plt.close(fig)
        item = {"image": image}
        if label:
            item["label"] = label
        return [item]

    def _make_empty_figure(self, message, title):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
        ax.set_title(title)
        return self._figure_to_image_payload(fig, label="message")

    def plot(self, df, objective, objective_column):
        title = f"Critical Difference Diagram — {objective}, {objective_column}"

        try:
            import scikit_posthocs as sp
        except ModuleNotFoundError:
            return self._make_empty_figure(
                "Optional dependency 'scikit-posthocs' is not installed. "
                "Install it to render the critical difference diagram.",
                title,
            )

        if objective_column not in df.columns:
            available = ", ".join(df.columns)
            return self._make_empty_figure(
                f"Column '{objective_column}' not found. Available columns: {available}",
                title,
            )

        if "objective_name" in df.columns:
            df = df[df["objective_name"] == objective]

        required_cols = {"dataset_name", "solver_name", objective_column}
        missing = required_cols.difference(df.columns)
        if missing:
            return self._make_empty_figure(
                f"Missing required columns for CD plot: {sorted(missing)}",
                title,
            )

        grouped = df.groupby(["dataset_name", "solver_name"], as_index=False)[
            objective_column
        ].mean()
        pivot_df = grouped.pivot(
            index="dataset_name", columns="solver_name", values=objective_column
        )
        pivot_df = pivot_df.dropna(axis=0, how="any")

        if pivot_df.shape[0] < 2 or pivot_df.shape[1] < 2:
            return self._make_empty_figure(
                "Need at least 2 datasets and 2 solvers (without missing values) "
                "to build a critical difference diagram.",
                title,
            )

        rank_df = pivot_df.rank(axis=1, ascending=True)
        mean_ranks = rank_df.mean(axis=0)
        nemenyi_p_values = sp.posthoc_nemenyi_friedman(pivot_df)

        fig, _ = plt.subplots(figsize=(10, 5))
        sp.critical_difference_diagram(mean_ranks, nemenyi_p_values)
        plt.title(title)
        plt.tight_layout()
        return self._figure_to_image_payload(fig, label="CD diagram")

    def get_metadata(self, df, objective, objective_column):
        return {
            "title": f"Critical Difference Diagram — {objective}",
            "xlabel": "Average rank (lower is better)",
            "ylabel": "",
            "objective_column": objective_column,
        }
