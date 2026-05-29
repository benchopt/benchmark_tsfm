"""Unit tests for map_iou, f1_det and their helpers."""

import math

import numpy as np
import pytest

from benchmark_utils.metrics import (
    _ap_from_tp_fp,
    _f1_from_class_counts,
    _iou_1d,
    _match_spans,
    event_iou_f1,
    event_span_iou,
    f1_det,
    map_iou,
)


def _evt(start, width, *class_cols):
    """Single-row event array: [start, width, *class_cols]."""
    return np.array([[start, width, *class_cols]], dtype=float)


def _no_evts(n_classes):
    return np.zeros((0, 2 + n_classes))


# ---------------------------------------------------------------------------
# _iou_1d
# ---------------------------------------------------------------------------

def test_iou_identical():
    assert _iou_1d(0.0, 0.5, 0.0, 0.5) == pytest.approx(1.0)

def test_iou_no_overlap():
    assert _iou_1d(0.0, 0.3, 0.4, 0.3) == pytest.approx(0.0)

def test_iou_adjacent():
    # Segments touch at a single point — inter = 0
    assert _iou_1d(0.0, 0.5, 0.5, 0.5) == pytest.approx(0.0)

def test_iou_partial():
    # [0, 1) and [0.5, 1.5) → inter=0.5, union=1.5
    assert _iou_1d(0.0, 1.0, 0.5, 1.0) == pytest.approx(0.5 / 1.5)

def test_iou_contained():
    # [0, 1) contains [0.2, 0.6) → inter=0.4, union=1.0
    assert _iou_1d(0.0, 1.0, 0.2, 0.4) == pytest.approx(0.4 / 1.0)

def test_iou_zero_width_returns_zero():
    assert _iou_1d(0.5, 0.0, 0.5, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _ap_from_tp_fp
# ---------------------------------------------------------------------------

def test_ap_no_gt_returns_nan():
    assert math.isnan(_ap_from_tp_fp(np.array([1.0]), np.array([0.0]), 0))

def test_ap_all_tp():
    ap = _ap_from_tp_fp(np.array([1.0, 1.0]), np.array([0.0, 0.0]), n_gt=2)
    assert ap == pytest.approx(1.0)

def test_ap_all_fp():
    ap = _ap_from_tp_fp(np.array([0.0, 0.0]), np.array([1.0, 1.0]), n_gt=2)
    assert ap == pytest.approx(0.0)

def test_ap_tp_then_fp():
    # Rank 1: TP (precision=1, recall=0.5), Rank 2: FP (recall still 0.5)
    # AP = Δrecall × precision = 0.5 × 1.0 = 0.5
    ap = _ap_from_tp_fp(np.array([1.0, 0.0]), np.array([0.0, 1.0]), n_gt=2)
    assert ap == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# map_iou
# ---------------------------------------------------------------------------

def test_map_empty_y_true_returns_nan():
    assert math.isnan(map_iou([], []))

def test_map_no_predictions_returns_zero():
    assert map_iou([_evt(0.0, 0.5, 1)], [_no_evts(1)]) == pytest.approx(0.0)

def test_map_no_gt_events_returns_nan():
    # Series has no GT events — AP is undefined
    assert math.isnan(map_iou([_no_evts(1)], [_evt(0.0, 0.5, 0.9)]))

def test_map_perfect_match():
    assert map_iou([_evt(0.1, 0.4, 1)], [_evt(0.1, 0.4, 1.0)]) == pytest.approx(1.0)

def test_map_no_overlap():
    assert map_iou([_evt(0.0, 0.3, 1)], [_evt(0.7, 0.3, 1.0)]) == pytest.approx(0.0)

def test_map_overlap_below_threshold_is_miss():
    # IoU ≈ 0.333 < 0.5
    assert map_iou([_evt(0.0, 1.0, 1)], [_evt(0.5, 1.0, 1.0)], iou_threshold=0.5) == pytest.approx(0.0)

def test_map_overlap_above_custom_threshold_is_hit():
    # Same IoU ≈ 0.333 > 0.3
    assert map_iou([_evt(0.0, 1.0, 1)], [_evt(0.5, 1.0, 1.0)], iou_threshold=0.3) == pytest.approx(1.0)

def test_map_duplicate_pred_only_first_matched():
    # Two identical predictions for one GT: first TP, second FP → AP=1.0
    preds = [np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 0.9]])]
    assert map_iou([_evt(0.0, 0.5, 1)], preds) == pytest.approx(1.0)

def test_map_predictions_ranked_by_score():
    # High-score pred misses (FP at rank 1), low-score pred hits (TP at rank 2)
    # AP = Δrecall × precision at rank 2 = 1.0 × 0.5 = 0.5
    preds = [np.array([[0.8, 0.1, 0.9], [0.0, 0.5, 0.5]])]
    assert map_iou([_evt(0.0, 0.5, 1)], preds) == pytest.approx(0.5)

def test_map_two_classes_both_perfect():
    gt = np.array([[0.0, 0.3, 1, 0], [0.5, 0.3, 0, 1]])
    pred = np.array([[0.0, 0.3, 1.0, 0.0], [0.5, 0.3, 0.0, 1.0]])
    assert map_iou([gt], [pred]) == pytest.approx(1.0)

def test_map_two_classes_one_missed():
    gt = np.array([[0.0, 0.3, 1, 0], [0.5, 0.3, 0, 1]])
    pred = np.array([
        [0.0, 0.3, 1.0, 0.0],   # class 0: perfect hit
        [0.0, 0.1, 0.0, 1.0],   # class 1: no overlap with GT at [0.5, 0.3]
    ])
    assert map_iou([gt], [pred]) == pytest.approx(0.5)

def test_map_multi_series_both_matched():
    y_true = [_evt(0.0, 0.5, 1), _evt(0.2, 0.4, 1)]
    y_pred = [_evt(0.0, 0.5, 1.0), _evt(0.2, 0.4, 1.0)]
    assert map_iou(y_true, y_pred) == pytest.approx(1.0)

def test_map_prediction_in_wrong_series_is_fp():
    # GT in series 0, pred only in series 1 — should not match
    y_true = [_evt(0.0, 0.5, 1), _no_evts(1)]
    y_pred = [_no_evts(1), _evt(0.0, 0.5, 1.0)]
    assert map_iou(y_true, y_pred) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Helpers for padded-array format used by event_span_iou / event_iou_f1
# ---------------------------------------------------------------------------

def _series(events, n_slots=5, n_classes=2):
    """Padded (n_slots, 2+n_classes) array from list of (start, width, *cols)."""
    arr = np.zeros((n_slots, 2 + n_classes))
    for i, ev in enumerate(events):
        arr[i] = ev
    return arr


# ---------------------------------------------------------------------------
# _match_spans
# ---------------------------------------------------------------------------

def test_match_spans_empty_gt():
    gt = np.zeros((0, 4))
    pr = np.array([[0.0, 0.5, 0.9, 0.0]])
    assert _match_spans(gt, pr, 0.5, "greedy") == []

def test_match_spans_empty_pred():
    gt = np.array([[0.0, 0.5, 1.0, 0.0]])
    pr = np.zeros((0, 4))
    assert _match_spans(gt, pr, 0.5, "greedy") == []

def test_match_spans_perfect_match_greedy():
    gt = np.array([[0.0, 0.5, 1.0, 0.0]])
    pr = np.array([[0.0, 0.5, 0.9, 0.0]])
    assert _match_spans(gt, pr, 0.5, "greedy") == [(0, 0)]

def test_match_spans_perfect_match_hungarian():
    gt = np.array([[0.0, 0.5, 1.0, 0.0]])
    pr = np.array([[0.0, 0.5, 0.9, 0.0]])
    assert _match_spans(gt, pr, 0.5, "hungarian") == [(0, 0)]

def test_match_spans_below_threshold_no_match():
    # IoU([0,1), [0.5,1.5)) ≈ 0.33 < 0.5
    gt = np.array([[0.0, 1.0, 1.0, 0.0]])
    pr = np.array([[0.5, 1.0, 0.9, 0.0]])
    assert _match_spans(gt, pr, 0.5, "greedy") == []

def test_match_spans_greedy_vs_hungarian_differ():
    # pred0: IoU=1.0 with GT, score=0.6
    # pred1: IoU=0.8 with GT, score=0.9
    # Greedy: pred1 (higher score) goes first → matches GT → [(0, 1)]
    # Hungarian: pred0 (higher IoU) is assigned → [(0, 0)]
    gt = np.array([[0.0, 0.5, 1.0, 0.0]])
    pr = np.array([
        [0.0, 0.5, 0.6, 0.0],   # pred0: IoU=1.0, score=0.6
        [0.1, 0.4, 0.9, 0.0],   # pred1: IoU=0.8, score=0.9
    ])
    assert _match_spans(gt, pr, 0.5, "greedy") == [(0, 1)]
    assert _match_spans(gt, pr, 0.5, "hungarian") == [(0, 0)]

def test_match_spans_duplicate_pred_only_one_matched():
    gt = np.array([[0.0, 0.5, 1.0, 0.0]])
    pr = np.array([[0.0, 0.5, 0.9, 0.0], [0.0, 0.5, 0.8, 0.0]])
    assert len(_match_spans(gt, pr, 0.5, "greedy")) == 1

def test_match_spans_two_gt_two_pred_both_matched():
    gt = np.array([[0.0, 0.3, 1, 0], [0.5, 0.3, 0, 1]])
    pr = np.array([[0.0, 0.3, 1.0, 0.0], [0.5, 0.3, 0.0, 1.0]])
    assert set(_match_spans(gt, pr, 0.5, "greedy")) == {(0, 0), (1, 1)}

def test_match_spans_invalid_strategy_raises():
    gt = np.array([[0.0, 0.5, 1.0, 0.0]])
    pr = np.array([[0.0, 0.5, 0.9, 0.0]])
    with pytest.raises(ValueError):
        _match_spans(gt, pr, 0.5, "invalid")


# ---------------------------------------------------------------------------
# _f1_from_class_counts
# ---------------------------------------------------------------------------

def test_f1_counts_micro_perfect():
    assert _f1_from_class_counts(np.array([2, 1]), np.array([0, 0]), np.array([0, 0]), "micro") == pytest.approx(1.0)

def test_f1_counts_micro_all_zeros():
    assert _f1_from_class_counts(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), "micro") == pytest.approx(0.0)

def test_f1_counts_micro_mixed():
    # tp=1, fp=1, fn=1 → P=0.5, R=0.5 → F1=0.5
    assert _f1_from_class_counts(np.array([1]), np.array([1]), np.array([1]), "micro") == pytest.approx(0.5)

def test_f1_counts_macro_mixed():
    # class 0: tp=1, fp=0, fn=0 → F1=1.0; class 1: tp=0, fp=1, fn=1 → F1=0 → mean=0.5
    assert _f1_from_class_counts(np.array([1, 0]), np.array([0, 1]), np.array([0, 1]), "macro") == pytest.approx(0.5)

def test_f1_counts_micro_macro_differ():
    # class 0: tp=1,fp=0,fn=0 → F1=1.0  |  class 1: tp=0,fp=0,fn=1 → F1=0
    # micro: tp_s=1,fp_s=0,fn_s=1 → P=1,R=0.5 → F1=2/3
    # macro: mean([1.0, 0.0]) = 0.5
    tp = np.array([1, 0])
    fp = np.array([0, 0])
    fn = np.array([0, 1])
    assert _f1_from_class_counts(tp, fp, fn, "micro") == pytest.approx(2 / 3)
    assert _f1_from_class_counts(tp, fp, fn, "macro") == pytest.approx(0.5)

def test_f1_counts_invalid_mode_raises():
    with pytest.raises(ValueError):
        _f1_from_class_counts(np.array([1]), np.array([0]), np.array([0]), "invalid")


# ---------------------------------------------------------------------------
# event_span_iou
# ---------------------------------------------------------------------------

def test_event_span_iou_both_empty():
    s = _series([])
    assert event_span_iou([s], [s]) == pytest.approx(1.0)

def test_event_span_iou_pred_empty():
    gt = _series([[0.0, 0.5, 1, 0]])
    pr = _series([])
    assert event_span_iou([gt], [pr]) == pytest.approx(0.0)

def test_event_span_iou_gt_empty():
    gt = _series([])
    pr = _series([[0.0, 0.5, 0.9, 0.0]])
    assert event_span_iou([gt], [pr]) == pytest.approx(0.0)

def test_event_span_iou_perfect_match():
    gt = _series([[0.0, 0.5, 1, 0]])
    pr = _series([[0.0, 0.5, 0.9, 0.0]])
    assert event_span_iou([gt], [pr]) == pytest.approx(1.0)

def test_event_span_iou_no_overlap():
    gt = _series([[0.0, 0.3, 1, 0]])
    pr = _series([[0.7, 0.3, 0.9, 0.0]])
    assert event_span_iou([gt], [pr]) == pytest.approx(0.0)

def test_event_span_iou_pred_below_score_threshold_ignored():
    # max class score 0.3 < 0.5 → pred filtered out → G=1, P=0 → F1=0
    gt = _series([[0.0, 0.5, 1, 0]])
    pr = _series([[0.0, 0.5, 0.3, 0.0]])
    assert event_span_iou([gt], [pr]) == pytest.approx(0.0)

def test_event_span_iou_multi_series_averaged():
    # series 0: perfect → F1=1.0; series 1: no pred → F1=0.0; mean=0.5
    gt0, pr0 = _series([[0.0, 0.5, 1, 0]]), _series([[0.0, 0.5, 0.9, 0.0]])
    gt1, pr1 = _series([[0.2, 0.4, 1, 0]]), _series([])
    assert event_span_iou([gt0, gt1], [pr0, pr1]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# event_iou_f1
# ---------------------------------------------------------------------------

def test_event_iou_f1_no_arrays_returns_nan():
    assert math.isnan(event_iou_f1([], []))

def test_event_iou_f1_perfect_match():
    gt = _series([[0.0, 0.5, 1, 0]])
    pr = _series([[0.0, 0.5, 0.9, 0.1]])
    assert event_iou_f1([gt], [pr]) == pytest.approx(1.0)

def test_event_iou_f1_unmatched_gt_is_fn():
    gt = _series([[0.0, 0.5, 1, 0]])
    pr = _series([])
    assert event_iou_f1([gt], [pr]) == pytest.approx(0.0)

def test_event_iou_f1_unmatched_pred_is_fp():
    gt = _series([])
    pr = _series([[0.0, 0.5, 0.9, 0.0]])
    assert event_iou_f1([gt], [pr]) == pytest.approx(0.0)

def test_event_iou_f1_wrong_class_prediction():
    # Span matched by IoU but class assignment is swapped → class errors
    # tp=[0,0], fp=[1,0], fn=[0,1]
    # micro: tp_s=0, fp_s=1, fn_s=1 → F1=0
    gt = _series([[0.0, 0.5, 1, 0]])   # class 0 active
    pr = _series([[0.0, 0.5, 0.4, 0.9]])  # predicts class 1 (max=0.9 > 0.5)
    assert event_iou_f1([gt], [pr]) == pytest.approx(0.0)

def test_event_iou_f1_micro_vs_macro_differ():
    # GT0=class0, GT1=class1; Pred0 correct, Pred1 wrong class
    # After matching: (GT0,P0) tp=[1,0]; (GT1,P1) gt_cls=[0,1] pr_cls=[1,0]
    #   → tp+=[0,0], fp+=[1,0], fn+=[0,1]
    # total: tp=[1,0], fp=[1,0], fn=[0,1]
    # micro: tp_s=1,fp_s=1,fn_s=1 → P=0.5,R=0.5 → F1=0.5
    # macro: class0 P=0.5,R=1→F1=2/3; class1 P=0,R=0→F1=0 → mean=1/3
    gt = _series([[0.0, 0.3, 1, 0], [0.5, 0.3, 0, 1]])
    pr = _series([[0.0, 0.3, 0.9, 0.1], [0.5, 0.3, 0.6, 0.4]])
    assert event_iou_f1([gt], [pr], mode="micro") == pytest.approx(0.5)
    assert event_iou_f1([gt], [pr], mode="macro") == pytest.approx(1 / 3)

def test_event_iou_f1_hungarian_strategy():
    gt = _series([[0.0, 0.5, 1, 0]])
    pr = _series([[0.0, 0.5, 0.9, 0.1]])
    assert event_iou_f1([gt], [pr], matching_strategy="hungarian") == pytest.approx(1.0)

def test_event_iou_f1_multi_series():
    gt0 = _series([[0.0, 0.5, 1, 0]])
    pr0 = _series([[0.0, 0.5, 0.9, 0.0]])
    gt1 = _series([[0.2, 0.4, 0, 1]])
    pr1 = _series([[0.2, 0.4, 0.1, 0.8]])
    assert event_iou_f1([gt0, gt1], [pr0, pr1]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# f1_det
# ---------------------------------------------------------------------------

def test_f1det_empty_y_true_returns_nan():
    assert math.isnan(f1_det([], []))

def test_f1det_no_gt_events_returns_nan():
    assert math.isnan(f1_det([_no_evts(1)], [_evt(0.0, 0.5, 0.9)]))

def test_f1det_perfect_match_fixed_threshold():
    assert f1_det([_evt(0.0, 0.5, 1)], [_evt(0.0, 0.5, 0.9)], score_threshold=0.5) == pytest.approx(1.0)

def test_f1det_no_overlap_fixed_threshold():
    # TP=0, FP=1, FN=1 → F1=0
    assert f1_det([_evt(0.0, 0.3, 1)], [_evt(0.7, 0.3, 0.9)], score_threshold=0.5) == pytest.approx(0.0)

def test_f1det_prediction_below_threshold_becomes_fn():
    # Pred score 0.3 < threshold 0.5 → filtered out → TP=0, FP=0, FN=1 → F1=0
    assert f1_det([_evt(0.0, 0.5, 1)], [_evt(0.0, 0.5, 0.3)], score_threshold=0.5) == pytest.approx(0.0)

def test_f1det_oracle_finds_best_threshold():
    # Low-score pred (0.2) would be filtered by fixed threshold=0.5 but oracle should find it
    assert f1_det([_evt(0.0, 0.5, 1)], [_evt(0.0, 0.5, 0.2)], score_threshold=None) == pytest.approx(1.0)

def test_f1det_duplicate_pred_second_is_fp():
    # TP=1, FP=1, FN=0 → F1 = 2/(2+1+0) = 2/3
    preds = [np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 0.9]])]
    assert f1_det([_evt(0.0, 0.5, 1)], preds, score_threshold=0.5) == pytest.approx(2 / 3)

def test_f1det_two_classes_both_perfect():
    gt = np.array([[0.0, 0.3, 1, 0], [0.5, 0.3, 0, 1]])
    pred = np.array([[0.0, 0.3, 1.0, 0.0], [0.5, 0.3, 0.0, 1.0]])
    assert f1_det([gt], [pred], score_threshold=0.5) == pytest.approx(1.0)

def test_f1det_two_classes_one_missed():
    gt = np.array([[0.0, 0.3, 1, 0], [0.5, 0.3, 0, 1]])
    pred = np.array([
        [0.0, 0.3, 1.0, 0.0],  # class 0: perfect hit
        [0.0, 0.1, 0.0, 0.9],  # class 1: no overlap with GT at [0.5, 0.3]
    ])
    # class 0: F1=1.0, class 1: TP=0 FP=1 FN=1 → F1=0 → mean=0.5
    assert f1_det([gt], [pred], score_threshold=0.5) == pytest.approx(0.5)

def test_f1det_multi_series_both_matched():
    y_true = [_evt(0.0, 0.5, 1), _evt(0.2, 0.4, 1)]
    y_pred = [_evt(0.0, 0.5, 0.9), _evt(0.2, 0.4, 0.8)]
    assert f1_det(y_true, y_pred, score_threshold=0.5) == pytest.approx(1.0)

def test_f1det_prediction_in_wrong_series_is_fp():
    y_true = [_evt(0.0, 0.5, 1), _no_evts(1)]
    y_pred = [_no_evts(1), _evt(0.0, 0.5, 0.9)]
    # TP=0, FP=1, FN=1 → F1=0
    assert f1_det(y_true, y_pred, score_threshold=0.5) == pytest.approx(0.0)
