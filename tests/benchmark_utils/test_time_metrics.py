"""Differential tests: our TIME metrics vs. the upstream TIME implementation.

``_upstream_per_window_metrics`` below is vendored from
``compute_per_window_metrics_from_quantiles`` in

    repo   github.com/zqiao11/TIME
    path   src/timebench/evaluation/metrics.py
    commit 6d3ba1150dfe8e551bc280cba2971bdb276f4f6b

(unchanged at 3ca5c41d71c76f3c70632c0121142d325350ee32), with its comments
and docstring stripped and ruff formatting applied; every statement is
unchanged. It is the reference only: production code must not use it.
"""

import numpy as np
import pytest

from benchmark_utils import metrics as M
from benchmark_utils.outputs import ForecastOutput

# Upstream's nanmean over an all-NaN MAPE cell warns; that's expected.
pytestmark = pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# upstream metric key → our registered metric name
NAMES = {
    "MSE": "time_mse",
    "MAE": "time_mae",
    "RMSE": "time_rmse",
    "MAPE": "time_mape",
    "sMAPE": "time_smape",
    "MASE": "time_mase",
    "ND": "time_nd",
    "CRPS": "time_crps",
}
CELLS = {
    "MSE": M.time_mse_cells,
    "MAE": M.time_mae_cells,
    "MAPE": M.time_mape_cells,
    "sMAPE": M.time_smape_cells,
    "ND": M.time_nd_cells,
    "CRPS": M.time_crps_cells,
}


# --- vendored upstream reference (see module docstring) -------------------
def _upstream_per_window_metrics(
    predictions_quantiles,
    ground_truth,
    context,
    seasonality=1,
    quantile_levels=None,
):
    if quantile_levels is None:
        quantile_levels = LEVELS
    (
        num_series,
        num_windows,
        num_quantiles,
        num_variates,
        pred_len,
    ) = predictions_quantiles.shape
    if len(quantile_levels) != num_quantiles:
        raise ValueError(
            "quantile_levels length must match predictions_quantiles' "
            "quantile dimension"
        )
    if 0.5 not in quantile_levels:
        raise ValueError("quantile_levels must include 0.5 for median-based metrics")
    median_idx = quantile_levels.index(0.5)

    mse = np.zeros((num_series, num_windows, num_variates))
    mae = np.zeros((num_series, num_windows, num_variates))
    rmse = np.zeros((num_series, num_windows, num_variates))
    mape = np.zeros((num_series, num_windows, num_variates))
    smape = np.zeros((num_series, num_windows, num_variates))
    mase = np.zeros((num_series, num_windows, num_variates))
    nd = np.zeros((num_series, num_windows, num_variates))
    crps = np.zeros((num_series, num_windows, num_variates))

    for s in range(num_series):
        for w in range(num_windows):
            for v in range(num_variates):
                q_preds = predictions_quantiles[s, w, :, v]
                gt = ground_truth[s, w, v]
                ctx = context[s, w, v]
                ctx = ctx[~np.isnan(ctx)]

                median_pred = q_preds[median_idx]

                valid_mask = ~np.isnan(gt)
                if not np.any(valid_mask):
                    mse[s, w, v] = mae[s, w, v] = rmse[s, w, v] = np.nan
                    mape[s, w, v] = smape[s, w, v] = mase[s, w, v] = np.nan
                    nd[s, w, v] = crps[s, w, v] = np.nan
                    continue

                gt = gt[valid_mask]
                median_pred = median_pred[valid_mask]
                q_preds = q_preds[:, valid_mask]

                error = gt - median_pred
                abs_error = np.abs(error)

                mse[s, w, v] = np.mean(error**2)
                mae[s, w, v] = np.mean(abs_error)
                rmse[s, w, v] = np.sqrt(mse[s, w, v])

                with np.errstate(divide="ignore", invalid="ignore"):
                    mape_vals = abs_error / np.abs(gt)
                    mape_vals = np.where(np.isfinite(mape_vals), mape_vals, np.nan)
                    mape[s, w, v] = np.nanmean(mape_vals)

                with np.errstate(divide="ignore", invalid="ignore"):
                    smape_vals = 2 * abs_error / (np.abs(gt) + np.abs(median_pred))
                    smape_vals = np.where(np.isfinite(smape_vals), smape_vals, np.nan)
                    smape[s, w, v] = np.nanmean(smape_vals)

                if len(ctx) > seasonality:
                    naive_errors = np.abs(ctx[seasonality:] - ctx[:-seasonality])
                    seasonal_error = (
                        np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
                    )
                    if seasonal_error > 0:
                        mase[s, w, v] = mae[s, w, v] / seasonal_error
                    else:
                        mase[s, w, v] = np.nan
                else:
                    mase[s, w, v] = np.nan

                abs_label_sum = np.sum(np.abs(gt))
                if abs_label_sum > 0:
                    nd[s, w, v] = np.sum(abs_error) / abs_label_sum
                else:
                    nd[s, w, v] = np.nan

                if abs_label_sum > 0:
                    weighted_quantile_losses = []
                    for q, q_pred in zip(quantile_levels, q_preds):
                        q_error = gt - q_pred
                        indicator = (q_pred >= gt).astype(float)
                        q_loss = 2 * np.abs(q_error * (indicator - q))
                        weighted_ql = np.sum(q_loss) / abs_label_sum
                        weighted_quantile_losses.append(weighted_ql)
                    crps[s, w, v] = np.mean(weighted_quantile_losses)
                else:
                    crps[s, w, v] = np.nan

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "sMAPE": smape,
        "MASE": mase,
        "ND": nd,
        "CRPS": crps,
    }


# --- fixtures ----------------------------------------------------------------


def _make_task(seed, n_series=6, n_windows=4, C=3, H=7, seasonality=5):
    """Series + upstream-style windows with every edge case upstream handles.

    Returns ``(series, cutoffs, quantiles)`` where ``quantiles[i]`` is
    ``(n_windows, H, C, 9)`` and windows are laid out as in TIME.
    """
    rng = np.random.default_rng(seed)
    test_length = n_windows * H + 3  # non-multiple: last 3 steps unscored
    series, cutoffs, quantiles = [], [], []
    for i in range(n_series):
        T = test_length + int(rng.integers(2, 60))
        x = rng.normal(10, 5, size=(T, C)).astype(np.float32)
        x[rng.random((T, C)) < 0.15] = np.nan  # gaps in context and gt
        if i == 0:
            x[:] = 0.0  # constant (scale 0) and zero-sum gt
        if i == 1:
            x[: T - test_length] = np.nan  # very short history
            x[T - test_length - 1] = 3.0
        cs = [T - test_length + w * H for w in range(n_windows)]
        if i == 2:
            x[cs[1] : cs[1] + H, 0] = np.nan  # all-NaN window, variate 0
        if i == 3 and C >= 3:
            x[: T - test_length, 1] = 4.0  # constant history in variate 1
            x[cs[0] : cs[0] + H, 2] = np.array([0, 0, 0, 0, 0, 0, 1.5])[:H]
        center = rng.normal(10, 5, size=(n_windows, H, C, 1))
        offsets = np.sort(rng.normal(0, 3, size=(n_windows, H, C, 9)), axis=-1)
        q = (center + offsets).astype(np.float32)
        if i == 4:
            q[0, 0, 0, :] = q[0, 0, 0, 4]  # crossing-free degenerate step
        series.append(x)
        cutoffs.append(cs)
        quantiles.append(q)
    return series, cutoffs, quantiles


def _flat_inputs(series, cutoffs, quantiles):
    """Our layout: (M, H, C) truth, ForecastOutput, per-row contexts."""
    H = quantiles[0].shape[1]
    y_true = np.concatenate(
        [np.stack([x[c : c + H] for c in cs]) for x, cs in zip(series, cutoffs)]
    )
    contexts = [x[:c] for x, cs in zip(series, cutoffs) for c in cs]
    return y_true, ForecastOutput(quantiles=quantiles, quantile_levels=LEVELS), contexts


def _upstream_inputs(series, cutoffs, quantiles):
    """Upstream layout, built as timebench.evaluation.saver does."""
    S, (W, H, C, Q) = len(series), quantiles[0].shape
    max_ctx = max(cs[-1] for cs in cutoffs)
    preds = np.zeros((S, W, Q, C, H))
    gt = np.zeros((S, W, C, H))
    ctx = np.full((S, W, C, max_ctx), np.nan)
    for s, (x, cs, q) in enumerate(zip(series, cutoffs, quantiles)):
        for w, c in enumerate(cs):
            preds[s, w] = q[w].transpose(2, 1, 0)  # (H, C, Q) -> (Q, C, H)
            gt[s, w] = x[c : c + H].T
            ctx[s, w, :, :c] = x[:c].T
    return preds, gt, ctx


def _assert_matches_upstream(series, cutoffs, quantiles, forecast, seasonality):
    y_true, _, contexts = _flat_inputs(series, cutoffs, quantiles)
    S, W, C = len(series), len(cutoffs[0]), series[0].shape[1]
    ref = _upstream_per_window_metrics(
        *_upstream_inputs(series, cutoffs, quantiles), seasonality=seasonality
    )
    ours = {k: f(y_true, forecast) for k, f in CELLS.items()}
    ours["RMSE"] = np.sqrt(ours["MSE"])
    ours["MASE"] = M.time_mase_cells(y_true, forecast, contexts, seasonality)
    for key, ref_cells in ref.items():
        np.testing.assert_allclose(
            ours[key].reshape(S, W, C),
            ref_cells,
            rtol=1e-12,
            atol=0,
            equal_nan=True,
            err_msg=key,
        )
        ours_task = M.ALL_METRICS[NAMES[key]](
            y_true, forecast, contexts=contexts, seasonality=seasonality
        )
        np.testing.assert_allclose(
            ours_task, np.nanmean(ref_cells), rtol=1e-12, err_msg=key
        )
    return ref


# --- tests -------------------------------------------------------------------


def test_matches_upstream():
    for seasonality in (1, 24):
        series, cutoffs, quantiles = _make_task(0, seasonality=seasonality)
        _, forecast, _ = _flat_inputs(series, cutoffs, quantiles)
        ref = _assert_matches_upstream(
            series, cutoffs, quantiles, forecast, seasonality
        )
        # The fixture really exercises every NaN branch upstream has.
        assert np.isnan(ref["MASE"][0]).all()  # scale 0
        assert np.isnan(ref["CRPS"][0]).all() and np.isnan(ref["ND"][0]).all()
        assert np.isnan(ref["MAE"][2, 1, 0])  # all-NaN window
        assert np.isnan(ref["MASE"][3, 0, 1])  # constant history in one variate
        if seasonality > 1:
            assert np.isnan(ref["MASE"][1, 0]).all()  # history too short
