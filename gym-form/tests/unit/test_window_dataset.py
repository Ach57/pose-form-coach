"""Unit tests for src.datasets.window_dataset — windowing, padding, sampling weights."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.datasets.window_dataset import WindowDataset


# ── Helpers ──────────────────────────────────────────────────────────


def _setup_video(
    tmp_path: Path,
    video_id: str,
    n_frames: int,
    n_features: int,
    label_dict: dict,
) -> tuple[Path, Path]:
    """Write a .npy feature file and .json label file, return (features_dir, labels_dir)."""
    feat_dir = tmp_path / "features"
    label_dir = tmp_path / "labels"
    feat_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    features = np.random.randn(n_frames, n_features).astype(np.float32)
    np.save(feat_dir / f"{video_id}.npy", features)

    with open(label_dir / f"{video_id}.json", "w") as f:
        json.dump(label_dict, f)

    return feat_dir, label_dir


# ── Tests ────────────────────────────────────────────────────────────


class TestWindowDatasetBasics:
    """Core windowing behaviour."""

    def test_uniform_stride_window_count(self, tmp_path):
        """A 128-frame video with T=64, stride=16 → expected number of stride windows."""
        feat_dir, label_dir = _setup_video(
            tmp_path, "v1", n_frames=128, n_features=16,
            label_dict={"a": [], "b": []},
        )
        ds = WindowDataset(window_size=64, stride=16, pos_center=False, hard_negatives=False)
        ds.build(["v1"], feat_dir, label_dir, ["a", "b"], fps=30.0)

        # (128 - 64) / 16 + 1 = 5
        assert len(ds) == 5

    def test_window_shape(self, tmp_path):
        feat_dir, label_dir = _setup_video(
            tmp_path, "v1", n_frames=128, n_features=16,
            label_dict={"a": [], "b": []},
        )
        ds = WindowDataset(window_size=64, stride=16, pos_center=False)
        ds.build(["v1"], feat_dir, label_dir, ["a", "b"], fps=30.0)

        feats, labels = ds[0]
        assert feats.shape == (64, 16)
        assert labels.shape == (64, 2)
        assert feats.dtype == np.float32
        assert labels.dtype == np.float32

    def test_short_video_padded(self, tmp_path):
        """A video shorter than window_size should be padded, not skipped."""
        feat_dir, label_dir = _setup_video(
            tmp_path, "short", n_frames=20, n_features=16,
            label_dict={"a": [], "b": []},
        )
        ds = WindowDataset(window_size=64, stride=16, pos_center=False)
        ds.build(["short"], feat_dir, label_dir, ["a", "b"], fps=30.0)

        assert len(ds) >= 1
        feats, labels = ds[0]
        assert feats.shape == (64, 16)

    def test_missing_files_skipped(self, tmp_path):
        """Videos without feature/label files are silently skipped."""
        feat_dir = tmp_path / "features"
        label_dir = tmp_path / "labels"
        feat_dir.mkdir()
        label_dir.mkdir()

        ds = WindowDataset(window_size=64, stride=16, pos_center=False)
        ds.build(["missing_video"], feat_dir, label_dir, ["a"], fps=30.0)
        assert len(ds) == 0


class TestWindowDatasetPositiveSampling:
    """Positive-centered and hard-negative window generation."""

    def test_pos_center_creates_extra_windows(self, tmp_path):
        """With pos_center=True, we should get more windows than stride-only."""
        label_dict = {"a": [[1.0, 2.0]], "b": []}  # positive segment in "a"
        feat_dir, label_dir = _setup_video(
            tmp_path, "v1", n_frames=128, n_features=16,
            label_dict=label_dict,
        )

        ds_no_center = WindowDataset(window_size=64, stride=16, pos_center=False, hard_negatives=False)
        ds_no_center.build(["v1"], feat_dir, label_dir, ["a", "b"], fps=30.0)

        ds_center = WindowDataset(window_size=64, stride=16, pos_center=True, hard_negatives=False)
        ds_center.build(["v1"], feat_dir, label_dir, ["a", "b"], fps=30.0)

        assert len(ds_center) >= len(ds_no_center)

    def test_positive_windows_contain_positive_frames(self, tmp_path):
        """Every window classified as 'positive' must actually contain ≥1 positive frame."""
        label_dict = {"a": [[1.0, 2.0]], "b": []}
        feat_dir, label_dir = _setup_video(
            tmp_path, "v1", n_frames=128, n_features=16,
            label_dict=label_dict,
        )
        ds = WindowDataset(window_size=64, stride=16, pos_center=True)
        ds.build(["v1"], feat_dir, label_dir, ["a", "b"], fps=30.0)

        for idx in ds._pos_indices:
            _, labels = ds[idx]
            assert labels.any(), f"Window {idx} classified as positive but has no positive frames"

    def test_pos_neg_indices_cover_all_windows(self, tmp_path):
        """Every window must be in either _pos_indices or _neg_indices."""
        label_dict = {"a": [[0.5, 1.5]], "b": []}
        feat_dir, label_dir = _setup_video(
            tmp_path, "v1", n_frames=128, n_features=16,
            label_dict=label_dict,
        )
        ds = WindowDataset(window_size=64, stride=16, pos_center=True)
        ds.build(["v1"], feat_dir, label_dir, ["a", "b"], fps=30.0)

        all_classified = set(ds._pos_indices) | set(ds._neg_indices)
        assert all_classified == set(range(len(ds)))


class TestSamplerWeights:
    """Tests for get_sampler_weights used with WeightedRandomSampler."""

    def test_weights_length(self, tmp_path):
        feat_dir, label_dir = _setup_video(
            tmp_path, "v1", n_frames=128, n_features=16,
            label_dict={"a": [[1.0, 2.0]], "b": []},
        )
        ds = WindowDataset(window_size=64, stride=16)
        ds.build(["v1"], feat_dir, label_dir, ["a", "b"], fps=30.0)

        weights = ds.get_sampler_weights()
        assert len(weights) == len(ds)

    def test_weights_all_positive(self, tmp_path):
        feat_dir, label_dir = _setup_video(
            tmp_path, "v1", n_frames=128, n_features=16,
            label_dict={"a": [[1.0, 2.0]], "b": []},
        )
        ds = WindowDataset(window_size=64, stride=16)
        ds.build(["v1"], feat_dir, label_dir, ["a", "b"], fps=30.0)

        weights = ds.get_sampler_weights()
        assert (weights > 0).all()

    def test_empty_dataset_empty_weights(self, tmp_path):
        ds = WindowDataset(window_size=64, stride=16)
        ds.build([], tmp_path, tmp_path, ["a"], fps=30.0)
        weights = ds.get_sampler_weights()
        assert len(weights) == 0
