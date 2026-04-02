"""Unit tests for src.eval.metrics — frame metrics, hysteresis, segment-mAP."""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.metrics import (
    frame_metrics,
    hysteresis_segments,
    segment_ap,
    segment_map,
    temporal_iou,
)


class TestTemporalIoU:

    def test_perfect_overlap(self):
        assert temporal_iou((0, 9), (0, 9)) == 1.0

    def test_no_overlap(self):
        assert temporal_iou((0, 4), (10, 14)) == 0.0

    def test_partial_overlap(self):
        iou = temporal_iou((0, 9), (5, 14))
        # inter=5, union=15
        assert abs(iou - 5 / 15) < 1e-6

    def test_contained(self):
        iou = temporal_iou((2, 5), (0, 9))
        # inter=4, union=10
        assert abs(iou - 4 / 10) < 1e-6


class TestHysteresisSegments:

    def test_basic(self):
        probs = np.array([0.1, 0.6, 0.7, 0.4, 0.2])
        segs = hysteresis_segments(probs, on=0.5, off=0.3, min_dur=1, bridge_gap=0)
        assert len(segs) == 1
        assert segs[0] == (1, 3)

    def test_empty_when_below_threshold(self):
        probs = np.array([0.1, 0.2, 0.3])
        segs = hysteresis_segments(probs, on=0.5, off=0.3, min_dur=1, bridge_gap=0)
        assert segs == []

    def test_min_dur_filters_short(self):
        probs = np.array([0.1, 0.6, 0.2])
        segs = hysteresis_segments(probs, on=0.5, off=0.3, min_dur=3, bridge_gap=0)
        assert segs == []

    def test_bridge_gap_merges(self):
        probs = np.array([0.6, 0.7, 0.2, 0.6, 0.7])
        segs = hysteresis_segments(probs, on=0.5, off=0.3, min_dur=1, bridge_gap=5)
        assert len(segs) == 1

    def test_segment_at_end(self):
        probs = np.array([0.1, 0.2, 0.6, 0.7, 0.8])
        segs = hysteresis_segments(probs, on=0.5, off=0.3, min_dur=1, bridge_gap=0)
        assert len(segs) == 1
        assert segs[0][1] == 4


class TestFrameMetrics:

    def test_perfect_prediction(self):
        preds = np.array([0.9, 0.8, 0.1, 0.0])
        targets = np.array([1.0, 1.0, 0.0, 0.0])
        m = frame_metrics(preds, targets)
        assert m["precision"] > 0.99
        assert m["recall"] > 0.99
        assert m["f1"] > 0.99

    def test_all_wrong(self):
        preds = np.array([0.9, 0.9, 0.9])
        targets = np.array([0.0, 0.0, 0.0])
        m = frame_metrics(preds, targets)
        assert m["precision"] < 0.01
        assert m["recall"] < 0.01


class TestSegmentAP:

    def test_perfect_match(self):
        ap = segment_ap(
            pred_segments=[(0, 9)],
            pred_scores=[0.9],
            gt_segments=[(0, 9)],
            tiou_threshold=0.5,
        )
        assert ap == 1.0

    def test_no_predictions_with_gt(self):
        ap = segment_ap([], [], [(0, 9)], tiou_threshold=0.5)
        assert ap == 0.0

    def test_no_gt_no_preds(self):
        ap = segment_ap([], [], [], tiou_threshold=0.5)
        assert ap == 1.0

    def test_no_gt_with_preds(self):
        ap = segment_ap([(0, 5)], [0.9], [], tiou_threshold=0.5)
        assert ap == 0.0


class TestSegmentMAP:

    def test_basic(self):
        result = segment_map(
            all_pred_segments=[[(0, 9)]],
            all_pred_scores=[[0.9]],
            all_gt_segments=[[(0, 9)]],
            tiou_thresholds=[0.5],
        )
        assert "mAP@0.50" in result
        assert result["mAP"] == 1.0
