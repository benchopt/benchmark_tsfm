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

# VUS metrics
# Ported from https://github.com/thedatumorg/VUS, vus/utils/metrics.py
# (`RangeAUC_volume_opt`) and vus/analysis/robustness_eval.py
# (`generate_curve`). Reference: Paparrizos et al., "Volume Under the
# Surface: A New Accuracy Evaluation Measure for Time-Series Anomaly
# Detection".


def _segments(labels):
    """Return list of (start, end) inclusive anomaly segments."""
    labels = np.asarray(labels)
    out = []
    i = 0
    n = len(labels)
    while i < n:
        if labels[i] == 0:
            i += 1
            continue
        j = i
        while j < n and labels[j] != 0:
            j += 1
        out.append((i, j - 1))
        i = j
    return out


def _extend_labels(labels, segments, window):
    """Soft-extend each anomaly segment by `window // 2` on each side with
    a sqrt fade, clipped to [0, 1]."""
    extended = labels.astype(float).copy()
    n = len(extended)
    if window == 0:
        return extended
    for s, e in segments:
        x1 = np.arange(e + 1, min(e + window // 2 + 1, n))
        if len(x1):
            extended[x1] += np.sqrt(1 - (x1 - e) / window)
        x2 = np.arange(max(s - window // 2, 0), s)
        if len(x2):
            extended[x2] += np.sqrt(1 - (s - x2) / window)
    return np.minimum(extended, 1.0)


def _merge_segments(segments, window, n):
    """Merge segments whose `window // 2` halos overlap."""
    if not segments:
        return []
    half = window // 2
    a = max(segments[0][0] - half, 0)
    merged = []
    for i in range(len(segments) - 1):
        if segments[i][1] + half < segments[i + 1][0] - half:
            merged.append((a, segments[i][1] + half))
            a = segments[i + 1][0] - half
    merged.append((a, min(segments[-1][1] + half, n - 1)))
    return merged


def _range_auc_volume(labels, score, window_size, thre=250):
    """Compute (VUS_ROC, VUS_PR) for one (labels, score) pair."""
    labels = np.asarray(labels)
    score = np.asarray(score, dtype=float)
    n = len(labels)
    P = labels.sum()
    seq = _segments(labels)
    l_full = _merge_segments(seq, window_size, n)

    score_sorted = -np.sort(-score)
    thresholds_idx = np.linspace(0, n - 1, thre).astype(int)
    N_pred = np.array([(score >= score_sorted[i]).sum() for i in thresholds_idx])

    auc = np.zeros(window_size + 1)
    ap = np.zeros(window_size + 1)

    for w in range(window_size + 1):
        labels_ext = _extend_labels(labels, seq, w)
        L = _merge_segments(seq, w, n)

        tf = np.zeros((thre + 2, 2))
        prec = np.ones(thre + 1)

        for j, i in enumerate(thresholds_idx, start=1):
            pred = score >= score_sorted[i]
            lab = labels_ext.copy()
            existence = 0
            for s, e in L:
                lab[s : e + 1] = labels_ext[s : e + 1] * pred[s : e + 1]
                if pred[s : e + 1].any():
                    existence += 1
            for s, e in seq:
                lab[s : e + 1] = 1

            TP = 0.0
            N_labels = 0.0
            for s, e in l_full:
                TP += np.dot(lab[s : e + 1], pred[s : e + 1])
                N_labels += lab[s : e + 1].sum()

            FP = N_pred[j - 1] - TP
            existence_ratio = existence / len(L) if L else 0.0

            P_new = (P + N_labels) / 2
            recall = min(TP / P_new, 1) if P_new > 0 else 0.0
            tpr = recall * existence_ratio
            fpr = FP / (n - P_new) if (n - P_new) > 0 else 0.0
            precision = TP / N_pred[j - 1] if N_pred[j - 1] > 0 else 0.0

            tf[j] = (tpr, fpr)
            prec[j] = precision

        tf[-1] = (1, 1)

        width = tf[1:, 1] - tf[:-1, 1]
        height = (tf[1:, 0] + tf[:-1, 0]) / 2
        auc[w] = np.dot(width, height)

        width_pr = tf[1:-1, 0] - tf[:-2, 0]
        height_pr = prec[1:]
        ap[w] = np.dot(width_pr, height_pr)

    return float(auc.mean()), float(ap.mean())


def _vus_per_series(y_true, y_score, slidingWindow, thre):
    """Average a chosen VUS scalar across non-empty series."""
    rocs, prs = [], []
    for yt, ys in zip(y_true, y_score):
        yt = np.asarray(yt)
        ys = np.asarray(ys)
        if yt.sum() == 0:
            continue
        roc, pr = _range_auc_volume(yt, ys, slidingWindow, thre)
        rocs.append(roc)
        prs.append(pr)
    if not rocs:
        return float("nan"), float("nan")
    return float(np.mean(rocs)), float(np.mean(prs))


def vus_roc(y_true, y_score, slidingWindow=100, thre=250):
    """Volume Under the Surface (ROC).

    Averaged per series. `slidingWindow` is the upper bound of the window
    axis for the volume integration; callers benchmarking heterogeneous
    series should pass a per-dataset value.
    """
    return _vus_per_series(y_true, y_score, slidingWindow, thre)[0]


def vus_pr(y_true, y_score, slidingWindow=100, thre=250):
    """Volume Under the Surface (PR). Averaged per series."""
    return _vus_per_series(y_true, y_score, slidingWindow, thre)[1]

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
    "vus_roc": vus_roc,
    "vus_pr": vus_pr,
}

ALL_METRICS = {**FORECASTING_METRICS, **CLASSIFICATION_METRICS, **AD_METRICS}
