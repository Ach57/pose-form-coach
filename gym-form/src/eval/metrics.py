"""Evaluation metrics for temporal action detection.

Provides:
- Frame-level precision, recall, F1 per label
- Segment-level mAP at configurable tIoU thresholds
- Hysteresis post-processing (logits → segments)
"""

from __future__ import annotations

import numpy as np


# ── Post-processing: logits → segments ───────────────────────────────


def hysteresis_segments(
    probs: np.ndarray,
    on: float = 0.5,
    off: float = 0.3,
    min_dur: int = 3,
    bridge_gap: int = 5,
) -> list[tuple[int, int]]:
    """Convert per-frame probabilities to (start, end) segments using hysteresis.

    Parameters
    ----------
    probs : (T,) array of probabilities in [0, 1]
    on : float — threshold to activate a segment
    off : float — threshold to deactivate
    min_dur : int — minimum segment duration in frames (discard shorter)
    bridge_gap : int — merge segments separated by fewer than this many frames

    Returns
    -------
    List of (start_frame, end_frame) inclusive tuples.
    """
    T = len(probs)
    segments: list[tuple[int, int]] = []
    active = False
    start = 0

    for t in range(T):
        if not active and probs[t] >= on:
            active = True
            start = t
        elif active and probs[t] < off:
            active = False
            segments.append((start, t - 1))

    if active:
        segments.append((start, T - 1))

    # Minimum duration filter
    segments = [(s, e) for s, e in segments if (e - s + 1) >= min_dur]

    # Bridge close gaps
    if len(segments) > 1:
        merged = [segments[0]]
        for s, e in segments[1:]:
            prev_s, prev_e = merged[-1]
            if s - prev_e - 1 <= bridge_gap:
                merged[-1] = (prev_s, e)
            else:
                merged.append((s, e))
        segments = merged

    return segments


# ── Frame-level metrics ──────────────────────────────────────────────


def frame_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Per-frame binary precision, recall, F1.

    Parameters
    ----------
    preds : (T,) or (T, L) — predicted probabilities
    targets : (T,) or (T, L) — ground-truth binary labels

    Returns
    -------
    Dict with keys: precision, recall, f1
    """
    binary = (preds >= threshold).astype(np.float32)
    tp = (binary * targets).sum()
    fp = (binary * (1 - targets)).sum()
    fn = ((1 - binary) * targets).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


# ── Segment-level mAP ───────────────────────────────────────────────


def temporal_iou(
    pred: tuple[int, int],
    gt: tuple[int, int],
) -> float:
    """Temporal IoU between two (start, end) segments (inclusive)."""
    inter_start = max(pred[0], gt[0])
    inter_end = min(pred[1], gt[1])
    inter = max(0, inter_end - inter_start + 1)
    union = (pred[1] - pred[0] + 1) + (gt[1] - gt[0] + 1) - inter
    return inter / max(union, 1)


def segment_ap(
    pred_segments: list[tuple[int, int]],
    pred_scores: list[float],
    gt_segments: list[tuple[int, int]],
    tiou_threshold: float = 0.5,
) -> float:
    """Average Precision for a single video and label at one tIoU threshold.

    Parameters
    ----------
    pred_segments : list of (start, end) predicted segments
    pred_scores : confidence score for each predicted segment
    gt_segments : list of (start, end) ground-truth segments
    tiou_threshold : minimum tIoU to count as a match

    Returns
    -------
    AP value in [0, 1].
    """
    if len(gt_segments) == 0:
        return 1.0 if len(pred_segments) == 0 else 0.0
    if len(pred_segments) == 0:
        return 0.0

    # Sort predictions by descending score
    order = np.argsort(-np.array(pred_scores))
    gt_matched = [False] * len(gt_segments)
    tp = np.zeros(len(pred_segments))
    fp = np.zeros(len(pred_segments))

    for rank, idx in enumerate(order):
        pred = pred_segments[idx]
        best_iou = 0.0
        best_gt = -1
        for gi, gt in enumerate(gt_segments):
            iou = temporal_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = gi
        if best_iou >= tiou_threshold and not gt_matched[best_gt]:
            tp[rank] = 1.0
            gt_matched[best_gt] = True
        else:
            fp[rank] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / len(gt_segments)
    precision = cum_tp / (cum_tp + cum_fp)

    # Interpolated AP (VOC-style)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx])
    return float(ap)


def segment_map(
    all_pred_segments: list[list[tuple[int, int]]],
    all_pred_scores: list[list[float]],
    all_gt_segments: list[list[tuple[int, int]]],
    tiou_thresholds: list[float] | None = None,
) -> dict[str, float]:
    """Mean Average Precision over multiple videos at multiple tIoU thresholds.

    Parameters
    ----------
    all_pred_segments : per-video list of predicted segments
    all_pred_scores : per-video list of confidence scores
    all_gt_segments : per-video list of ground-truth segments
    tiou_thresholds : list of tIoU thresholds (default [0.1, 0.25, 0.5])

    Returns
    -------
    Dict with keys: 'mAP@{threshold}' for each threshold, and 'mAP' (average).
    """
    tiou_thresholds = tiou_thresholds or [0.1, 0.25, 0.5]
    results: dict[str, float] = {}

    for tiou in tiou_thresholds:
        aps = []
        for preds, scores, gts in zip(
            all_pred_segments, all_pred_scores, all_gt_segments
        ):
            aps.append(segment_ap(preds, scores, gts, tiou))
        results[f"mAP@{tiou:.2f}"] = float(np.mean(aps)) if aps else 0.0

    results["mAP"] = float(np.mean(list(results.values())))
    return results
