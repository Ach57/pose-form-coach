"""
WindowDataset — Sliding-window dataset for temporal form detection.

Builds contiguous windows of length T from per-video feature + label arrays.
Supports positive-centered sampling and hard-negative mining.

Usage:
    ds = WindowDataset.from_config(config, split="train", features_dir=..., labels_dir=...)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    for features, labels in loader:
        # features: (B, T, F)  labels: (B, T, L)
        ...
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils.io import load_json, load_yaml, rasterize_multilabel


@dataclass
class WindowSample:
    """A single training window."""

    video_id: str
    start_frame: int
    features: np.ndarray   # (T, F)
    labels: np.ndarray     # (T, L) — multi-hot per frame


class WindowDataset:
    """Sliding-window dataset over per-video features and rasterized labels.

    Parameters
    ----------
    window_size : int
        Number of frames per window (T).
    stride : int
        Step between consecutive windows.
    pos_center : bool
        If True, additionally create windows centered on positive segments.
    pos_frac : float
        Target fraction of positive windows per epoch (via oversampling).
    hard_negatives : bool
        If True, oversample windows near positive/negative boundaries.
    """

    def __init__(
        self,
        window_size: int = 64,
        stride: int = 16,
        pos_center: bool = True,
        pos_frac: float = 0.5,
        hard_negatives: bool = True,
    ) -> None:
        self.window_size = window_size
        self.stride = stride
        self.pos_center = pos_center
        self.pos_frac = pos_frac
        self.hard_negatives = hard_negatives
        self._windows: list[WindowSample] = []
        self._pos_indices: list[int] = []
        self._neg_indices: list[int] = []
        self._boundary_indices: list[int] = []

    def build(
        self,
        video_ids: list[str],
        features_dir: Path,
        labels_dir: Path,
        label_names: list[str],
        fps: float = 30.0,
    ) -> None:
        """Populate self._windows from feature/label files for the given video IDs."""
        features_dir = Path(features_dir)
        labels_dir = Path(labels_dir)
        self._windows.clear()
        self._pos_indices.clear()
        self._neg_indices.clear()
        self._boundary_indices.clear()

        skipped = 0
        for video_id in video_ids:
            feat_path = features_dir / f"{video_id}.npy"
            label_path = labels_dir / f"{video_id}.json"

            if not feat_path.exists() or not label_path.exists():
                skipped += 1
                continue

            feats = np.load(feat_path)  # (T_video, F)
            label_dict = load_json(label_path)
            T_video = feats.shape[0]

            if T_video < self.window_size:
                # Pad short videos
                pad = self.window_size - T_video
                feats = np.pad(feats, ((0, pad), (0, 0)), mode="edge")
                # Rasterize with original T then pad
                labels_arr = rasterize_multilabel(label_dict, T_video, fps, label_names)
                labels_arr = np.pad(labels_arr, ((0, pad), (0, 0)), mode="constant")
                T_video = self.window_size
            else:
                labels_arr = rasterize_multilabel(label_dict, T_video, fps, label_names)

            # ── Uniform-stride windows ──
            starts = list(range(0, T_video - self.window_size + 1, self.stride))
            if not starts:
                starts = [0]

            # ── Positive-centered windows ──
            if self.pos_center:
                starts = set(starts)
                for col in range(labels_arr.shape[1]):
                    pos_frames = np.where(labels_arr[:, col] > 0)[0]
                    if len(pos_frames) == 0:
                        continue
                    # Center windows on segment midpoints
                    segments = self._contiguous_segments(pos_frames)
                    for seg_start, seg_end in segments:
                        mid = (seg_start + seg_end) // 2
                        center = mid - self.window_size // 2
                        center = max(0, min(center, T_video - self.window_size))
                        starts.add(center)

                        # Hard negatives: windows that straddle segment boundaries
                        if self.hard_negatives:
                            for offset in [-self.window_size // 2, self.window_size // 2]:
                                boundary_start = seg_start + offset
                                boundary_start = max(0, min(boundary_start, T_video - self.window_size))
                                starts.add(boundary_start)

                starts = sorted(starts)

            for s in starts:
                w_feat = feats[s : s + self.window_size]
                w_labels = labels_arr[s : s + self.window_size]
                idx = len(self._windows)

                self._windows.append(WindowSample(
                    video_id=video_id,
                    start_frame=s,
                    features=w_feat.astype(np.float32),
                    labels=w_labels.astype(np.float32),
                ))

                # Classify window
                has_pos = w_labels.any()
                if has_pos:
                    self._pos_indices.append(idx)
                else:
                    self._neg_indices.append(idx)

        if skipped:
            print(f"  WindowDataset: skipped {skipped}/{len(video_ids)} "
                  f"videos (missing features or labels)")
        print(f"  WindowDataset: {len(self._windows)} windows "
              f"({len(self._pos_indices)} positive, {len(self._neg_indices)} negative)")

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        w = self._windows[idx]
        return w.features, w.labels

    def get_sampler_weights(self) -> np.ndarray:
        """Return per-sample weights to achieve target pos_frac via WeightedRandomSampler."""
        n = len(self._windows)
        if n == 0:
            return np.array([])
        weights = np.ones(n, dtype=np.float64)
        n_pos = max(len(self._pos_indices), 1)
        n_neg = max(len(self._neg_indices), 1)
        # Weight so that sampling yields pos_frac ratio
        w_pos = self.pos_frac / n_pos
        w_neg = (1.0 - self.pos_frac) / n_neg
        for i in self._pos_indices:
            weights[i] = w_pos
        for i in self._neg_indices:
            weights[i] = w_neg
        return weights

    @classmethod
    def from_config(
        cls,
        config: dict,
        split: str,
        features_dir: Path,
        labels_dir: Path,
    ) -> "WindowDataset":
        """Factory: build a WindowDataset from a dataset YAML config and a split name."""
        splits_dir = Path(config["paths"]["source_splits_dir"])
        split_ids = load_json(splits_dir / f"{split}_keys.json")

        win_cfg = config.get("window", {})
        sam_cfg = config.get("sampling", {})

        ds = cls(
            window_size=win_cfg.get("T", 64),
            stride=win_cfg.get("stride", 16),
            pos_center=sam_cfg.get("pos_center", True),
            pos_frac=sam_cfg.get("pos_frac", 0.5),
            hard_negatives=sam_cfg.get("hard_negatives", True),
        )
        ds.build(
            video_ids=split_ids,
            features_dir=features_dir,
            labels_dir=labels_dir,
            label_names=config["labels"],
            fps=config.get("fps", 30.0),
        )
        return ds

    @staticmethod
    def _contiguous_segments(indices: np.ndarray) -> list[tuple[int, int]]:
        """Convert sorted frame indices into (start, end) contiguous segments."""
        if len(indices) == 0:
            return []
        segments = []
        start = indices[0]
        prev = indices[0]
        for i in indices[1:]:
            if i != prev + 1:
                segments.append((start, prev))
                start = i
            prev = i
        segments.append((start, prev))
        return segments
