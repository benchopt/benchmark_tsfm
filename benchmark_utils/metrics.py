"""
Metric wrappers for all four tasks.

Relies on sklearn (classification + AD) and numpy for forecasting and
event detection. Forecasting metrics consume a :class:`ForecastOutput`
so probabilistic metrics (CRPS, WQL, MCIS, Pinball) see the full
quantile fan; point metrics extract the median internally.

Signatures
----------
forecasting        : metric(y_true, forecast: ForecastOutput, **kw) -> float
classification     : metric(y_true, y_pred) -> float
anomaly_detection  : metric(y_true, y_score) -> float
event_detection    : metric(y_true, y_pred, **kw) -> float
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from benchmark_utils.outputs import ForecastOutput


# ---------------------------------------------------------------------------
# Forecasting — internal helpers
# ---------------------------------------------------------------------------

def _stacked(forecast: ForecastOutput):
    """Return (quantiles (M,Q,H,C), levels (Q,)) — flattening if needed."""
    if len(forecast.quantiles) != 1:
        forecast = forecast.flatten()
    if not forecast.quantiles:
        raise ValueError("ForecastOutput is empty — no predictions to score")
    return forecast.quantiles[0], np.asarray(forecast.quantile_levels, dtype=np.float64)


def _point_from_forecast(forecast: ForecastOutput) -> np.ndarray:
    """Extract (M, H, C) point forecast — median when available, else mean."""
    quants, levels = _stacked(forecast)
    if 0.5 in levels:
        return quants[:, int(np.where(levels == 0.5)[0][0])]
    return quants.mean(axis=1)


def _seasonal_naive_scale(y_train, seasonality: int) -> float:
    """MAE of the seasonal-naive forecast on training series. Used by MASE."""
    scales = []
    for ts in y_train:
        ts = np.asarray(ts)
        if ts.shape[0] > seasonality:
            scales.append(float(np.mean(np.abs(ts[seasonality:] - ts[:-seasonality]))))
    scale = float(np.mean(scales)) if scales else 1.0
    return scale if scale != 0 else 1.0


def _pinball_per_level(y_true: np.ndarray, forecast: ForecastOutput) -> np.ndarray:
    """Pinball loss array of shape (Q,): mean over (M, H, C) for each level."""
    quants, levels = _stacked(forecast)            # (M,Q,H,C), (Q,)
    diff = y_true[:, None] - quants                # (M,Q,H,C)
    levels_b = levels.reshape(1, -1, 1, 1)
    loss = np.maximum(levels_b * diff, (levels_b - 1.0) * diff)
    return loss.mean(axis=(0, 2, 3))               # (Q,)


# ---------------------------------------------------------------------------
# Forecasting — point metrics
# ---------------------------------------------------------------------------

def mae(y_true, forecast: ForecastOutput, **_):
    """Mean Absolute Error, averaged over all windows, horizons, channels."""
    return float(np.mean(np.abs(y_true - _point_from_forecast(forecast))))


def mse(y_true, forecast: ForecastOutput, **_):
    """Mean Squared Error, averaged over all windows, horizons, channels."""
    return float(np.mean((y_true - _point_from_forecast(forecast)) ** 2))


def rmse(y_true, forecast: ForecastOutput, **_):
    return float(np.sqrt(mse(y_true, forecast)))


def mase(y_true, forecast: ForecastOutput, y_train, seasonality=1, **_):
    """Mean Absolute Scaled Error. Scale is naive-seasonal MAE on y_train."""
    scale = _seasonal_naive_scale(y_train, seasonality)
    return float(np.mean(np.abs(y_true - _point_from_forecast(forecast))) / scale)


def smape(y_true, forecast: ForecastOutput, **_):
    """Symmetric Mean Absolute Percentage Error."""
    y_pred = _point_from_forecast(forecast)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def skill_score_ratio(y_true, forecast: ForecastOutput, y_train, seasonality=1, **_):
    """1 - MAE_model / MAE_naive.

    ``MAE_naive`` is the seasonal-naive MAE on the training series (same
    scale as :func:`mase`), so the score is identical to ``1 - mase``.
    Positive = better than naive; 0 = parity; negative = worse.
    """
    scale = _seasonal_naive_scale(y_train, seasonality)
    model_mae = float(np.mean(np.abs(y_true - _point_from_forecast(forecast))))
    return float(1.0 - model_mae / scale)


# ---------------------------------------------------------------------------
# Forecasting — probabilistic metrics
# ---------------------------------------------------------------------------

def pinball(y_true, forecast: ForecastOutput, **_):
    """Mean pinball (quantile) loss, averaged over all quantile levels."""
    return float(_pinball_per_level(y_true, forecast).mean())


def crps(y_true, forecast: ForecastOutput, **_):
    """CRPS approximated by the quantile-score formula: 2 * mean pinball loss.

    Converges to the true CRPS as the quantile grid becomes dense.
    """
    return 2.0 * pinball(y_true, forecast)


def wql(y_true, forecast: ForecastOutput, **_):
    """Weighted Quantile Loss (Salinas et al. / Chronos).

    ``WQL = (1/Q) sum_q [ 2 * sum_t pinball_q(y_t) / sum_t |y_t| ]``
    """
    _, levels = _stacked(forecast)
    denom = float(np.sum(np.abs(y_true)))
    if denom == 0:
        return float("nan")
    # _pinball_per_level returns per-level mean; multiply back by N to get sum.
    n_elem = float(y_true.size)
    per_level_sum = _pinball_per_level(y_true, forecast) * n_elem
    return float(2.0 * per_level_sum.sum() / (len(levels) * denom))


def mcis(y_true, forecast: ForecastOutput, alpha=0.05, **_):
    """Mean Coverage Interval Score for a ``(1 - alpha)`` prediction interval.

    ``IS_alpha = (U - L) + (2/alpha)(L - y) 1[y < L] + (2/alpha)(y - U) 1[y > U]``

    The lower / upper bounds are taken from the quantile levels closest to
    ``alpha/2`` and ``1 - alpha/2``. With the standard Chronos quantile
    grid (0.1, …, 0.9), an ``alpha=0.05`` request snaps to the 0.1 / 0.9
    bounds — effectively scoring an 80% PI on its 95% schedule.
    """
    quants, levels = _stacked(forecast)
    li = int(np.argmin(np.abs(levels - alpha / 2.0)))
    ui = int(np.argmin(np.abs(levels - (1.0 - alpha / 2.0))))
    lower = quants[:, li]                              # (M, H, C)
    upper = quants[:, ui]
    under = np.maximum(0.0, lower - y_true)
    over = np.maximum(0.0, y_true - upper)
    score = (upper - lower) + (2.0 / alpha) * (under + over)
    return float(np.mean(score))


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
    """Compute (VUS_ROC, VUS_PR) for one (labels, score) pair.

    Vectorized port of `metricor.RangeAUC_volume_opt`. Algebraic identity
    used: TP = dot(labels_ext, pred), N_labels = P + TP - dot(labels, pred).
    Threshold sweep is vectorized via a cumulative sum of labels_ext in
    descending score order. Existence count uses per-segment max-score
    plus searchsorted.
    """
    labels = np.asarray(labels)
    score = np.asarray(score, dtype=float)
    n = len(labels)
    P = float(labels.sum())
    seq = _segments(labels)

    # Constant across windows: threshold values and N_pred per threshold.
    score_sorted = -np.sort(-score)
    score_asc = score_sorted[::-1]
    thresholds = score_sorted[np.linspace(0, n - 1, thre).astype(int)]
    ks = n - np.searchsorted(score_asc, thresholds, side="left")

    # Constant across windows: TP_strict = dot(labels, pred) per threshold,
    # via cumulative sum of binary labels in descending score order.
    order = np.argsort(-score, kind="stable")
    B_cum = np.cumsum(labels[order].astype(float))
    TP_strict = B_cum[ks - 1]

    auc = np.zeros(window_size + 1)
    ap = np.zeros(window_size + 1)

    for w in range(window_size + 1):
        labels_ext = _extend_labels(labels, seq, w)
        L = _merge_segments(seq, w, n)

        TP = np.cumsum(labels_ext[order])[ks - 1]
        N_labels = P + TP - TP_strict
        P_new = (P + N_labels) / 2
        N_new = n - P_new

        # P_new >= P > 0 (labels_ext >= labels) and N_new > 0 for any non-
        # pathological input, so plain division is safe here.
        recall = np.minimum(TP / P_new, 1.0)
        fpr = (ks - TP) / N_new
        precision = TP / ks

        if L:
            max_scores = np.sort(np.array([score[s:e + 1].max() for s, e in L]))
            existence = len(L) - np.searchsorted(max_scores, thresholds, side="left")
            existence_ratio = existence / len(L)
        else:
            existence_ratio = np.zeros(thre)

        tpr = recall * existence_ratio

        tf = np.zeros((thre + 2, 2))
        tf[1:thre + 1, 0] = tpr
        tf[1:thre + 1, 1] = fpr
        tf[-1] = (1, 1)
        prec = np.ones(thre + 1)
        prec[1:] = precision

        auc[w] = np.dot(tf[1:, 1] - tf[:-1, 1], (tf[1:, 0] + tf[:-1, 0]) / 2)
        ap[w] = np.dot(tf[1:-1, 0] - tf[:-2, 0], prec[1:])

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
# Event detection
# ---------------------------------------------------------------------------

def _iou_1d(s1, w1, s2, w2):
    s1, w1, s2, w2 = float(s1), float(w1), float(s2), float(w2)
    inter = max(0.0, min(s1 + w1, s2 + w2) - max(s1, s2))
    union = w1 + w2 - inter
    return inter / union if union > 0.0 else 0.0


def _ap_from_tp_fp(tp, fp, n_gt):
    """Area under the precision-recall step function."""
    if n_gt == 0:
        return float("nan")
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / (tp_cum + fp_cum)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def map_iou(y_true, y_pred, iou_threshold=0.5):
    """Mean Average Precision at a 1-D IoU threshold for event detection.

    Parameters
    ----------
    y_true : list of np.ndarray (N_gt, 2+K)
        Ground-truth events per series.  Cols: [start_norm, width_norm, *one_hot].
    y_pred : list of np.ndarray (N_pred, 2+K)
        Predicted events per series.  Cols: [start_norm, width_norm, *class_scores].
        Score for class k is y_pred[i, 2+k]; confidence = per-class score.
    iou_threshold : float
        Minimum IoU to count a prediction as a true positive (default 0.5).
    """
    if not y_true:
        return float("nan")

    n_classes = y_true[0].shape[1] - 2
    aps = []

    for k in range(n_classes):
        # Collect GT boxes for class k, grouped by series index
        gt_by_series = {}
        n_gt = 0
        for i, gt in enumerate(y_true):
            boxes = [(row[0], row[1]) for row in gt
                     if len(gt) > 0 and np.argmax(row[2:]) == k]
            gt_by_series[i] = boxes
            n_gt += len(boxes)

        # Collect all predictions for class k: (series_idx, start, width, score)
        preds = []
        for i, pred in enumerate(y_pred):
            for row in pred:
                preds.append((i, row[0], row[1], float(row[2 + k])))
        preds.sort(key=lambda x: -x[3])

        matched = {i: [False] * len(gt_by_series[i]) for i in gt_by_series}
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))

        for j, (i, s, w, _) in enumerate(preds):
            best_iou, best_gi = 0.0, -1
            for gi, (gs, gw) in enumerate(gt_by_series.get(i, [])):
                iou = _iou_1d(s, w, gs, gw)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= iou_threshold and best_gi >= 0 and not matched[i][best_gi]:
                tp[j] = 1.0
                matched[i][best_gi] = True
            else:
                fp[j] = 1.0

        aps.append(_ap_from_tp_fp(tp, fp, n_gt))

    valid = [ap for ap in aps if not np.isnan(ap)]
    return float(np.mean(valid)) if valid else float("nan")


# ---------------------------------------------------------------------------
# Registry: maps metric name → function
# ---------------------------------------------------------------------------

FORECASTING_METRICS = {
    "mae": mae,
    "mse": mse,
    "rmse": rmse,
    "mase": mase,
    "smape": smape,
    "crps": crps,
    "wql": wql,
    "mcis": mcis,
    "pinball": pinball,
    "skill_score_ratio": skill_score_ratio,
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

EVENT_METRICS = {
    "map_iou": map_iou,
}

ALL_METRICS = {**FORECASTING_METRICS, **CLASSIFICATION_METRICS, **AD_METRICS,
               **EVENT_METRICS}
