#!/usr/bin/env python3
"""
Preprocess labels — Convert the Fitness-AQA dataset label format into the
gym-form pipeline format.

Source format (one JSON per error type, all videos in one dict):
    error_elbows.json  →  {"66089_2": [], "72676_1": [[2.01, 2.95], [4.45, 5.1]], ...}
    error_knees.json   →  {"66089_2": [[0.0, 1.13]], ...}

Target format (one JSON per video, all labels merged):
    data/labels/ohp/66089_2.json  →  {"ohp_elbow": [], "ohp_knee": [[0.0, 1.13]]}
    data/labels/ohp/72676_1.json  →  {"ohp_elbow": [[2.01, 2.95], [4.45, 5.1]], "ohp_knee": []}

Usage:
    python scripts/preprocess_labels.py --config configs/ohp/dataset.yaml
    python scripts/preprocess_labels.py --config configs/squat/dataset.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import load_json, load_yaml, save_json


def preprocess_labels(config_path: str | Path) -> None:
    config = load_yaml(config_path)
    source_dir = REPO_ROOT / config["paths"]["source_labels_dir"]
    output_dir = REPO_ROOT / config["paths"]["labels_dir"]
    label_sources = config["label_sources"]  # {canonical_name: source_filename}

    print(f"Source labels dir : {source_dir}")
    print(f"Output labels dir : {output_dir}")
    print(f"Label mapping     : {label_sources}")

    # Load all source label files
    source_data: dict[str, dict] = {}
    for canonical_name, source_file in label_sources.items():
        path = source_dir / source_file
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping {canonical_name}")
            continue
        source_data[canonical_name] = load_json(path)
        print(f"  Loaded {canonical_name}: {len(source_data[canonical_name])} videos from {source_file}")

    # Collect all video IDs across all label sources
    all_video_ids: set[str] = set()
    for label_name, videos_dict in source_data.items():
        all_video_ids.update(videos_dict.keys())

    print(f"\nTotal unique video IDs across all label files: {len(all_video_ids)}")

    # Merge into per-video JSON files
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for video_id in sorted(all_video_ids):
        merged = {}
        for canonical_name in label_sources:
            if canonical_name in source_data:
                merged[canonical_name] = source_data[canonical_name].get(video_id, [])
            else:
                merged[canonical_name] = []

        save_json(merged, output_dir / f"{video_id}.json")
        written += 1

    print(f"Wrote {written} per-video label files to {output_dir}")

    # Quick stats
    for canonical_name in label_sources:
        if canonical_name not in source_data:
            continue
        n_pos = sum(
            1 for v in source_data[canonical_name].values() if len(v) > 0
        )
        n_total = len(source_data[canonical_name])
        print(f"  {canonical_name}: {n_pos}/{n_total} videos have ≥1 positive segment "
              f"({100 * n_pos / n_total:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess labels for gym-form pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ohp/dataset.yaml",
        help="Path to dataset config YAML (relative to repo root)",
    )
    args = parser.parse_args()
    preprocess_labels(REPO_ROOT / args.config)
