#!/usr/bin/env python3
"""
Extract poses + features for all videos in a split.

Runs PoseExtractor → OHPFeatureExtractor for each video, caching intermediate
.npz (poses) and .npy (features) files.  Skips videos already processed.

Usage:
    python scripts/extract_all.py --config configs/dataset.ohp.yaml [--splits train val test]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.extract.pose_extractor import PoseExtractor
from src.features.feature_extractor import OHPFeatureExtractor
from src.utils.io import load_json, load_yaml


def extract_all(config_path: str | Path, splits: list[str]) -> None:
    config = load_yaml(config_path)
    videos_dir = REPO_ROOT / config["paths"]["source_videos_dir"]
    poses_dir = REPO_ROOT / config["paths"]["poses_dir"]
    features_dir = REPO_ROOT / "data" / "features" / "ohp"
    splits_dir = REPO_ROOT / config["paths"]["source_splits_dir"]
    fps = config["fps"]

    poses_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    # Collect all video IDs from requested splits
    video_ids: list[str] = []
    for split in splits:
        path = splits_dir / f"{split}_keys.json"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping split '{split}'")
            continue
        ids = load_json(path)
        video_ids.extend(ids)
        print(f"  {split}: {len(ids)} videos")

    # Deduplicate while preserving order
    seen = set()
    unique_ids = []
    for vid in video_ids:
        if vid not in seen:
            seen.add(vid)
            unique_ids.append(vid)
    video_ids = unique_ids
    print(f"\nTotal unique videos to process: {len(video_ids)}")

    # Resolve video file paths (try common extensions)
    def find_video(video_id: str) -> Path | None:
        for ext in (".mp4", ".avi", ".mov", ".mkv"):
            p = videos_dir / f"{video_id}{ext}"
            if p.exists():
                return p
        return None

    # Init extractors
    pose_extractor = PoseExtractor(
        model_path="artifacts/pose_landmarker_heavy.task",
    )
    feat_extractor = OHPFeatureExtractor(fps=fps, smooth_window=5)

    done = 0
    skipped_cache = 0
    skipped_missing = 0
    failed = 0
    t0 = time.time()

    for i, video_id in enumerate(video_ids):
        feat_path = features_dir / f"{video_id}.npy"
        pose_path = poses_dir / f"{video_id}.npz"

        # Skip if features already exist
        if feat_path.exists():
            skipped_cache += 1
            continue

        video_file = find_video(video_id)
        if video_file is None:
            skipped_missing += 1
            continue

        try:
            # Pose extraction (or load from cache)
            if pose_path.exists():
                pose_result = PoseExtractor.load_cache(pose_path)
            else:
                pose_result = pose_extractor.extract_video(video_file)
                PoseExtractor.save_cache(pose_result, pose_path)

            # Feature extraction
            features = feat_extractor.extract(pose_result.landmarks)  # (T, 16)
            np.save(feat_path, features)
            done += 1

            if (done % 50) == 0 or done == 1:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  [{done}/{len(video_ids)}] {video_id}  "
                      f"T={pose_result.n_frames}  success={pose_result.success_rate:.1%}  "
                      f"feat={features.shape}  ({rate:.1f} vid/s)")

        except Exception as e:
            failed += 1
            print(f"  FAIL: {video_id} — {e}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.1f}s")
    print(f"  Extracted : {done}")
    print(f"  Cached    : {skipped_cache}")
    print(f"  Missing   : {skipped_missing}")
    print(f"  Failed    : {failed}")
    print(f"  Features  : {features_dir}")
    print(f"  Poses     : {poses_dir}")
    print(f"{'=' * 60}")


# Needed because we use np.save outside of io.py
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract poses and features for gym-form")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset.ohp.yaml",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
    )
    args = parser.parse_args()
    extract_all(args.config, args.splits)
