"""
Metric wrappers for all three tasks.

Relies on aeon (forecasting metrics), sklearn (classification + AD),
and a minimal numpy implementation of MASE (not yet in aeon).

All functions follow the signature:
    metric(y_true, y_pred, **kwargs) -> float
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

def mae(y_true, y_pred):
    """Mean Absolute Error, averaged over all windows and channels."""
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def mse(y_true, y_pred):
    """Mean Squared Error, averaged over all windows and channels."""
    return float(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def mase(y_true, y_pred, y_train, seasonality=1):
    """Mean Absolute Scaled Error.

    Parameters
    ----------
    y_true : array-like (M, H, C) or list of (H, C)
    y_pred : array-like (M, H, C) or list of (H, C)
    y_train : list of (T_i, C) training series used to compute the naive scale
    seasonality : int
        Seasonal period for the naive seasonal baseline (default 1 = random
        walk baseline).
    """
    y_true = np.array(y_true)   # (M, H, C)
    y_pred = np.array(y_pred)

    # Scale: MAE of seasonal naive on training data
    scales = []
    for ts in y_train:
        ts = np.array(ts)  # (T, C)
        if ts.shape[0] > seasonality:
            naive_err = np.mean(
                np.abs(ts[seasonality:] - ts[:-seasonality])
            )
            scales.append(naive_err)
    scale = np.mean(scales) if scales else 1.0
    if scale == 0:
        scale = 1.0

    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def accuracy(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))


def balanced_accuracy(y_true, y_pred):
    return float(balanced_accuracy_score(y_true, y_pred))


def f1_weighted(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def auc_roc(y_true, y_score):
    """Area under ROC curve. Expects point-level scores and labels.

    Parameters
    ----------
    y_true  : list of (T_j,) int arrays, concatenated
    y_score : list of (T_j,) float arrays, concatenated
    """
    y_true = np.concatenate([np.asarray(y) for y in y_true])
    y_score = np.concatenate([np.asarray(y) for y in y_score])
    if y_true.sum() == 0:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def auc_pr(y_true, y_score):
    """Area under Precision-Recall curve."""
    y_true = np.concatenate([np.asarray(y) for y in y_true])
    y_score = np.concatenate([np.asarray(y) for y in y_score])
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def f1_pa(y_true, y_score, threshold=None):
    """F1 with point-adjust (PA): if any point in an anomaly segment is
    detected, the entire segment is counted as detected.

    Parameters
    ----------
    y_true  : list of (T_j,) int arrays
    y_score : list of (T_j,) float arrays
    threshold : float or None
        If None, the threshold is chosen to maximise F1 on the test set
        (oracle threshold — for benchmarking purposes only).
    """
    y_true_cat = np.concatenate([np.asarray(y) for y in y_true])
    y_score_cat = np.concatenate([np.asarray(y) for y in y_score])

    if threshold is None:
        # Oracle: sweep thresholds and pick best F1 after point-adjust
        thresholds = np.percentile(y_score_cat, np.arange(0, 100, 1))
        best_f1 = 0.0
        for thr in thresholds:
            y_pred = (y_score_cat >= thr).astype(int)
            y_pred_pa = _point_adjust(y_true_cat, y_pred)
            f = float(f1_score(y_true_cat, y_pred_pa, zero_division=0))
            if f > best_f1:
                best_f1 = f
        return best_f1

    y_pred = (y_score_cat >= threshold).astype(int)
    y_pred_pa = _point_adjust(y_true_cat, y_pred)
    return float(f1_score(y_true_cat, y_pred_pa, zero_division=0))


def _point_adjust(y_true, y_pred):
    """If any predicted anomaly overlaps with a true anomaly segment,
    label all points in that segment as detected."""
    y_pred_adj = y_pred.copy()
    in_anomaly = False
    seg_start = 0
    for i, label in enumerate(y_true):
        if label == 1 and not in_anomaly:
            in_anomaly = True
            seg_start = i
        elif label == 0 and in_anomaly:
            # segment ended: if any detection in [seg_start, i), fill it
            if y_pred[seg_start:i].any():
                y_pred_adj[seg_start:i] = 1
            in_anomaly = False
    if in_anomaly and y_pred[seg_start:].any():
        y_pred_adj[seg_start:] = 1
    return y_pred_adj


# ---------------------------------------------------------------------------
# Registry: maps metric name → function
# ---------------------------------------------------------------------------

FORECASTING_METRICS = {
    "mae": mae,
    "mse": mse,
    "rmse": rmse,
    "mase": mase,
    "smape": smape,
}

CLASSIFICATION_METRICS = {
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "f1_weighted": f1_weighted,
}

AD_METRICS = {
    "auc_roc": auc_roc,
    "auc_pr": auc_pr,
    "f1_pa": f1_pa,
}

ALL_METRICS = {**FORECASTING_METRICS, **CLASSIFICATION_METRICS, **AD_METRICS}


# ---------------------------------------------------------------------------
# Direction: which metrics are better when larger
# ---------------------------------------------------------------------------
# Direction is an intrinsic property of a metric, so it is declared here next
# to the definitions — this is the single source of truth. The objective
# re-exports it and every plot imports ``is_higher_better`` from here; nothing
# redeclares this knowledge. Forecasting losses (mae/mse/rmse/mase/smape) and
# FEV error metrics (wql/sql/wape/...) are all lower-is-better and so are
# simply absent from the set below.
HIGHER_IS_BETTER = frozenset({
    "accuracy", "balanced_accuracy", "f1_weighted",   # classification
    "auc_roc", "auc_pr", "f1_pa",                      # anomaly detection
})


def is_higher_better(metric: str) -> bool:
    """Whether a larger value of ``metric`` means a better model.

    Accepts either a bare metric name (``"auc_pr"``) or a benchopt results
    column (``"objective_auc_pr"``). Unknown metrics default to
    lower-is-better, the convention for forecasting/error losses.
    """
    name = metric[len("objective_"):] if metric.startswith("objective_") else metric
    return name in HIGHER_IS_BETTER
