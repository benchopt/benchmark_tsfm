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


def _collect_tp_pairs(gt_by_series, preds, iou_threshold):
    """Greedy IoU matching. Returns list of ([s,w]_pred, [s,w]_gt) for each TP."""
    matched = {i: [False] * len(boxes) for i, boxes in gt_by_series.items()}
    pairs = []
    for i, s, w, _ in preds:
        best_iou, best_gi = 0.0, -1
        for gi, (gs, gw) in enumerate(gt_by_series.get(i, [])):
            iou = _iou_1d(s, w, gs, gw)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= iou_threshold and best_gi >= 0 and not matched[i][best_gi]:
            gs, gw = gt_by_series[i][best_gi]
            pairs.append(([s, w], [gs, gw]))
            matched[i][best_gi] = True
    return pairs


def _match_events(gt_by_series, preds, iou_threshold):
    """Greedy IoU matching (highest-score first). Returns (tp, fp, fn)."""
    matched = {i: [False] * len(boxes) for i, boxes in gt_by_series.items()}
    n_gt = sum(len(b) for b in gt_by_series.values())
    tp = fp = 0
    for i, s, w, _ in preds:
        best_iou, best_gi = 0.0, -1
        for gi, (gs, gw) in enumerate(gt_by_series.get(i, [])):
            iou = _iou_1d(s, w, gs, gw)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= iou_threshold and best_gi >= 0 and not matched[i][best_gi]:
            tp += 1
            matched[i][best_gi] = True
        else:
            fp += 1
    return tp, fp, n_gt - tp


def f1_det(y_true, y_pred, iou_threshold=0.5, score_threshold=None):
    """Macro F1 for event detection.

    Parameters
    ----------
    y_true, y_pred : same format as map_iou
    iou_threshold  : minimum IoU to count a match (default 0.5)
    score_threshold: fixed class score threshold; if None, sweep to maximise F1
                     per class (oracle — for benchmarking purposes only)
    """
    if not y_true:
        return float("nan")

    n_classes = y_true[0].shape[1] - 2
    f1s = []

    for k in range(n_classes):
        gt_by_series = {}
        n_gt = 0
        for i, gt in enumerate(y_true):
            boxes = [(row[0], row[1]) for row in gt
                     if len(gt) > 0 and np.argmax(row[2:]) == k]
            gt_by_series[i] = boxes
            n_gt += len(boxes)

        if n_gt == 0:
            f1s.append(float("nan"))
            continue

        all_preds = []
        for i, pred in enumerate(y_pred):
            for row in pred:
                all_preds.append((i, row[0], row[1], float(row[2 + k])))
        all_preds.sort(key=lambda x: -x[3])

        if score_threshold is None:
            scores = np.array([p[3] for p in all_preds])
            thresholds = (
                np.percentile(scores, np.arange(0, 100, 1))
                if len(scores) else [0.5]
            )
            best_f1 = 0.0
            for thr in thresholds:
                preds = [p for p in all_preds if p[3] >= thr]
                tp, fp, fn = _match_events(gt_by_series, preds, iou_threshold)
                denom = 2 * tp + fp + fn
                best_f1 = max(best_f1, (2 * tp / denom) if denom > 0 else 0.0)
            f1s.append(best_f1)
        else:
            preds = [p for p in all_preds if p[3] >= score_threshold]
            tp, fp, fn = _match_events(gt_by_series, preds, iou_threshold)
            denom = 2 * tp + fp + fn
            f1s.append((2 * tp / denom) if denom > 0 else 0.0)

    valid = [f for f in f1s if not np.isnan(f)]
    return float(np.mean(valid)) if valid else float("nan")
# _span_iou is an alias kept consistent with _iou_1d above


def _span_iou(start_a, len_a, start_b, len_b):
    """Intersection-over-Union for two [start, start+length) intervals."""
    return _iou_1d(start_a, len_a, start_b, len_b)


def event_span_iou(y_true, y_pred, iou_threshold=0.5):
    """Mean span IoU for event detection with greedy matching.

    Computes precision/recall/F1 of event spans across all series using greedy
    IoU matching. A predicted span is a true positive if its IoU with an
    unmatched ground-truth span exceeds ``iou_threshold``.

    Parameters
    ----------
    y_true : List[np.ndarray (N=10, 2+k)]
        Ground-truth padded event targets. All-zero rows = empty slots.
    y_pred : List[np.ndarray (N=10, 2+k)]
        Predicted outputs. Positions (cols 0-1) in [0,1]; class probs in [0,1].
    iou_threshold : float
        Minimum IoU to count as a correct span detection (default 0.5).

    Returns
    -------
    float — span F1 score averaged over all series
    """
    f1_scores = []
    for gt, pr in zip(y_true, y_pred):
        gt = np.asarray(gt)
        pr = np.asarray(pr)

        gt_mask = gt[:, 2:].sum(axis=1) > 0
        pr_mask = pr[:, 2:].max(axis=1) > 0.5

        gt_spans = gt[gt_mask, :2]
        pr_spans = pr[pr_mask, :2]

        G = gt_spans.shape[0]
        P = pr_spans.shape[0]

        if G == 0 and P == 0:
            f1_scores.append(1.0)
            continue
        if G == 0 or P == 0:
            f1_scores.append(0.0)
            continue

        matched_gt = set()
        tp = 0
        for pi in range(P):
            best_iou = 0.0
            best_gi = -1
            for gi in range(G):
                if gi in matched_gt:
                    continue
                iou = _span_iou(
                    pr_spans[pi, 0], pr_spans[pi, 1],
                    gt_spans[gi, 0], gt_spans[gi, 1],
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= iou_threshold and best_gi >= 0:
                matched_gt.add(best_gi)
                tp += 1

        precision = tp / P if P > 0 else 0.0
        recall = tp / G if G > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall > 0 else 0.0)
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def event_class_f1(y_true, y_pred, iou_threshold=0.5):
    """Micro-F1 over binary class columns on IoU-matched event slots.

    For each predicted span that is IoU-matched to a ground-truth span, we
    threshold class probabilities at 0.5 and compute micro-F1 over the k
    binary class columns. Unmatched ground-truth spans count as false negatives
    for all their active classes.

    Parameters
    ----------
    y_true : List[np.ndarray (N=10, 2+k)]
    y_pred : List[np.ndarray (N=10, 2+k)]
    iou_threshold : float

    Returns
    -------
    float — micro-F1 over class columns
    """
    tp_total = fp_total = fn_total = 0

    for gt, pr in zip(y_true, y_pred):
        gt = np.asarray(gt)
        pr = np.asarray(pr)

        gt_mask = gt[:, 2:].sum(axis=1) > 0
        pr_mask = pr[:, 2:].max(axis=1) > 0.5

        gt_spans = gt[gt_mask]
        pr_spans = pr[pr_mask]

        G = gt_spans.shape[0]
        P = pr_spans.shape[0]

        matched_gt = {}
        matched_pr = set()
        for gi in range(G):
            best_iou = 0.0
            best_pi = -1
            for pi in range(P):
                if pi in matched_pr:
                    continue
                iou = _span_iou(
                    pr_spans[pi, 0], pr_spans[pi, 1],
                    gt_spans[gi, 0], gt_spans[gi, 1],
                )
                if iou > best_iou:
                    best_iou = iou
                    best_pi = pi
            if best_iou >= iou_threshold and best_pi >= 0:
                matched_gt[gi] = best_pi
                matched_pr.add(best_pi)

        for gi, pi in matched_gt.items():
            gt_cls = (gt_spans[gi, 2:] > 0.5).astype(int)
            pr_cls = (pr_spans[pi, 2:] > 0.5).astype(int)
            tp_total += int((gt_cls & pr_cls).sum())
            fp_total += int(((1 - gt_cls) & pr_cls).sum())
            fn_total += int((gt_cls & (1 - pr_cls)).sum())

        for gi in range(G):
            if gi not in matched_gt:
                fn_total += int((gt_spans[gi, 2:] > 0.5).sum())

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    if precision + recall > 0:
        return float(2 * precision * recall / (precision + recall))
    return 0.0


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

EVENT_METRICS = {
    "map_iou": map_iou,
    << << << < HEAD
    "f1_det": f1_det,
    == == == =
    "event_span_iou": event_span_iou,
    "event_class_f1": event_class_f1,
    >>>>>> > 0d18cdca10c37ad36a1377ba9308990025a2a078
}

ALL_METRICS = {
    **FORECASTING_METRICS,
    **CLASSIFICATION_METRICS,
    **AD_METRICS,
    **EVENT_METRICS,
}
