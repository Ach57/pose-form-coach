#!/usr/bin/env python3
"""
Audit the dataset — compute class-balance stats needed before training.

Outputs:
  - Per-label positive/negative frame ratio.
  - Recommended pos_weight values for BCEWithLogitsLoss.
  - Segment count and duration distributions.
  - Per-split breakdowns.

Usage:
    python scripts/audit_data.py --config configs/dataset.ohp.yaml
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import load_json, load_yaml, rasterize_segments


def audit_data(config_path: str | Path) -> dict:
    config = load_yaml(config_path)
    labels_dir = REPO_ROOT / config["paths"]["labels_dir"]
    splits_dir = REPO_ROOT / config["paths"]["source_splits_dir"]
    fps = config["fps"]
    label_names: list[str] = config["labels"]
    source_videos_dir = REPO_ROOT / config["paths"]["source_videos_dir"]

    print(f"Labels dir : {labels_dir}")
    print(f"Splits dir : {splits_dir}")
    print(f"FPS        : {fps}")
    print(f"Labels     : {label_names}")

    # ── Load splits ──────────────────────────────────────────────────
    splits: dict[str, list[str]] = {}
    for name in ["train", "val", "test"]:
        path = splits_dir / f"{name}_keys.json"
        if path.exists():
            splits[name] = load_json(path)

    all_ids = set()
    for ids in splits.values():
        all_ids.update(ids)

    # ── Gather video durations ───────────────────────────────────────
    # We estimate duration from label segment endpoints since we may not
    # have extracted poses yet. If the source videos directory is available
    # we could use OpenCV, but segments-only estimation is fine for audit.

    print(f"\nScanning {len(all_ids)} videos...")

    # Per-label accumulators (segment-level)
    seg_counts: dict[str, int] = {ln: 0 for ln in label_names}
    seg_durations: dict[str, list[float]] = {ln: [] for ln in label_names}
    videos_with_error: dict[str, int] = {ln: 0 for ln in label_names}

    # Per-label accumulators (estimated frame-level, using max timestamp as proxy for duration)
    total_positive_time: dict[str, float] = {ln: 0.0 for ln in label_names}
    total_video_time: float = 0.0

    max_timestamps: list[float] = []  # per video

    for video_id in sorted(all_ids):
        label_path = labels_dir / f"{video_id}.json"
        if not label_path.exists():
            continue

        label_data = load_json(label_path)

        # Estimate video length from maximum segment endpoint
        max_t = 0.0
        for ln in label_names:
            segs = label_data.get(ln, [])
            for s, e in segs:
                max_t = max(max_t, e)
        if max_t == 0.0:
            # No positive segments — try to get duration from video file
            max_t = _get_video_duration(source_videos_dir, video_id)
        if max_t == 0.0:
            continue  # can't estimate without either segments or video

        max_timestamps.append(max_t)
        total_video_time += max_t

        for ln in label_names:
            segs = label_data.get(ln, [])
            seg_counts[ln] += len(segs)
            if segs:
                videos_with_error[ln] += 1
            for s, e in segs:
                dur = e - s
                seg_durations[ln].append(dur)
                total_positive_time[ln] += dur

    # ── Compute pos_weight ───────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("CLASS BALANCE REPORT")
    print(f"{'=' * 60}")
    print(f"Total videos analyzed: {len(max_timestamps)}")
    if max_timestamps:
        print(f"Estimated total time : {total_video_time:.1f}s  "
              f"(median video {statistics.median(max_timestamps):.1f}s)")

    pos_weights: list[float] = []
    for ln in label_names:
        pos_time = total_positive_time[ln]
        neg_time = total_video_time - pos_time
        if pos_time > 0:
            pw = neg_time / pos_time
        else:
            pw = 1.0
        pos_weights.append(round(pw, 2))

        print(f"\n  [{ln}]")
        print(f"    Segments        : {seg_counts[ln]}")
        print(f"    Videos w/ error : {videos_with_error[ln]} / {len(max_timestamps)}")
        if seg_durations[ln]:
            durs = seg_durations[ln]
            print(f"    Segment dur (s) : min={min(durs):.2f}  "
                  f"median={statistics.median(durs):.2f}  "
                  f"max={max(durs):.2f}  "
                  f"mean={statistics.mean(durs):.2f}")
        print(f"    Positive time   : {pos_time:.1f}s / {total_video_time:.1f}s "
              f"({100 * pos_time / max(total_video_time, 1e-9):.2f}%)")
        print(f"    Recommended pos_weight: {pw:.2f}")

    print(f"\n{'=' * 60}")
    print(f"pos_weight list for train config: {pos_weights}")
    print(f"Copy this into configs/train.ohp.yaml → train.pos_weight")
    print(f"{'=' * 60}")

    # ── Per-split breakdown ──────────────────────────────────────────
    print(f"\nPER-SPLIT BREAKDOWN:")
    for split_name, split_ids in splits.items():
        print(f"\n  [{split_name}] ({len(split_ids)} videos)")
        for ln in label_names:
            n_pos = 0
            for vid in split_ids:
                lp = labels_dir / f"{vid}.json"
                if lp.exists():
                    ld = load_json(lp)
                    if ld.get(ln, []):
                        n_pos += 1
            print(f"    {ln}: {n_pos}/{len(split_ids)} videos with error")

    return {"pos_weights": pos_weights, "label_names": label_names}


def _get_video_duration(videos_dir: Path, video_id: str) -> float:
    """Try to get video duration via OpenCV. Returns 0.0 on failure."""
    try:
        import cv2
    except ImportError:
        return 0.0

    for ext in (".mp4", ".avi", ".mov", ".mkv"):
        vpath = videos_dir / f"{video_id}{ext}"
        if vpath.exists():
            cap = cv2.VideoCapture(str(vpath))
            if cap.isOpened():
                n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if fps > 0:
                    return n_frames / fps
            cap.release()
    return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit dataset for gym-form pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset.ohp.yaml",
        help="Path to dataset config YAML",
    )
    args = parser.parse_args()
    audit_data(args.config)
