#!/usr/bin/env python3
"""
Validate train/val/test splits for identity leakage and label coverage.

Checks:
  1. No video ID appears in multiple splits (hard overlap).
  2. No *subject ID* (prefix of video_id before underscore) spans multiple splits
     (identity leakage).
  3. Every video ID in the splits has a corresponding label file.
  4. Every label file maps to a split (no orphaned labels).

Usage:
    python scripts/validate_splits.py --config configs/ohp/dataset.yaml
    python scripts/validate_splits.py --config configs/squat/dataset.yaml
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import load_json, load_yaml


def _subject_of(video_id: str) -> str:
    """Extract the subject ID from a video ID like '66089_2'."""
    return video_id.rsplit("_", 1)[0]


def validate_splits(config_path: str | Path) -> bool:
    config = load_yaml(config_path)
    splits_dir = REPO_ROOT / config["paths"]["source_splits_dir"]
    labels_dir = REPO_ROOT / config["paths"]["labels_dir"]

    ok = True

    # ── Load splits ──────────────────────────────────────────────────
    split_names = ["train", "val", "test"]
    splits: dict[str, list[str]] = {}
    for name in split_names:
        path = splits_dir / f"{name}_keys.json"
        if not path.exists():
            print(f"WARNING: split file {path} not found")
            ok = False
            continue
        splits[name] = load_json(path)
        print(f"  {name}: {len(splits[name])} video IDs")

    # ── 1. Hard overlap ──────────────────────────────────────────────
    print("\n[1/4] Checking for duplicate video IDs across splits...")
    video_to_split: dict[str, str] = {}
    for name, ids in splits.items():
        for vid in ids:
            if vid in video_to_split:
                print(f"  FAIL: '{vid}' in both '{video_to_split[vid]}' and '{name}'")
                ok = False
            video_to_split[vid] = name
    if ok:
        print("  PASS — no duplicates.")

    # ── 2. Identity leakage ──────────────────────────────────────────
    print("\n[2/4] Checking for identity (subject) leakage...")
    subject_to_splits: dict[str, set[str]] = defaultdict(set)
    for name, ids in splits.items():
        for vid in ids:
            subject_to_splits[_subject_of(vid)].add(name)

    leaky_subjects = {
        sub: sorted(s) for sub, s in subject_to_splits.items() if len(s) > 1
    }
    if leaky_subjects:
        print(f"  FAIL: {len(leaky_subjects)} subjects appear in multiple splits:")
        for sub, which in sorted(leaky_subjects.items())[:20]:
            print(f"    subject={sub}  splits={which}")
        if len(leaky_subjects) > 20:
            print(f"    ... and {len(leaky_subjects) - 20} more.")
        ok = False
    else:
        print("  PASS — no identity leakage.")

    # ── 3. Label coverage ────────────────────────────────────────────
    print("\n[3/4] Checking that every split video ID has a label file...")
    all_split_ids = set(video_to_split.keys())
    missing_labels = sorted(
        vid for vid in all_split_ids
        if not (labels_dir / f"{vid}.json").exists()
    )
    if missing_labels:
        print(f"  FAIL: {len(missing_labels)} split IDs have no label file in {labels_dir}:")
        for vid in missing_labels[:10]:
            print(f"    {vid}")
        if len(missing_labels) > 10:
            print(f"    ... and {len(missing_labels) - 10} more.")
        ok = False
    else:
        print("  PASS — all split IDs have label files.")

    # ── 4. Orphaned labels ───────────────────────────────────────────
    print("\n[4/4] Checking for orphaned label files (not in any split)...")
    if labels_dir.exists():
        label_ids = {p.stem for p in labels_dir.glob("*.json")}
        orphaned = sorted(label_ids - all_split_ids)
        if orphaned:
            print(f"  INFO: {len(orphaned)} label files have no split assignment:")
            for vid in orphaned[:10]:
                print(f"    {vid}")
            if len(orphaned) > 10:
                print(f"    ... and {len(orphaned) - 10} more.")
        else:
            print("  PASS — no orphaned labels.")
    else:
        print(f"  SKIP — labels_dir {labels_dir} does not exist yet (run preprocess_labels first).")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    if ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED — see details above.")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate splits for gym-form pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ohp/dataset.yaml",
        help="Path to dataset config YAML",
    )
    args = parser.parse_args()
    passed = validate_splits(args.config)
    sys.exit(0 if passed else 1)
