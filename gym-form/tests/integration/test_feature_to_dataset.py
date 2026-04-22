"""Integration test: real features + labels → WindowDataset.

Loads one real video's .npy and .json from data/, builds a WindowDataset,
and validates shapes, label alignment, and positive-window correctness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.datasets.window_dataset import WindowDataset

# ── Constants ────────────────────────────────────────────────────────

REPO = Path(__file__).resolve().parent.parent.parent
FEATURES_DIR = REPO / "data" / "features" / "ohp"
LABELS_DIR = REPO / "data" / "labels" / "ohp"

# A video known to have positive elbow segments
VIDEO_ID = "62805_6"
LABEL_NAMES = ["ohp_elbow", "ohp_knee"]
WINDOW_T = 64
STRIDE = 16
FPS = 30.0


# ── Guard: skip if data not extracted yet ────────────────────────────

pytestmark = pytest.mark.skipif(
    not (FEATURES_DIR / f"{VIDEO_ID}.npy").exists()
    or not (LABELS_DIR / f"{VIDEO_ID}.json").exists(),
    reason="Extracted data not present — run scripts/extract_all.py first",
)


# ── Tests ────────────────────────────────────────────────────────────


class TestFeatureThroughDataset:
    """End-to-end: real features file → WindowDataset → valid windows."""

    @pytest.fixture(autouse=True)
    def _build_dataset(self):
        self.ds = WindowDataset(
            window_size=WINDOW_T,
            stride=STRIDE,
            pos_center=True,
            hard_negatives=True,
        )
        self.ds.build(
            [VIDEO_ID], FEATURES_DIR, LABELS_DIR, LABEL_NAMES, fps=FPS
        )

    def test_dataset_not_empty(self):
        assert len(self.ds) > 0, "Dataset should have at least one window"

    def test_feature_window_shape(self):
        feats, labels = self.ds[0]
        assert feats.shape == (WINDOW_T, 16), f"Expected (64, 16), got {feats.shape}"
        assert labels.shape == (WINDOW_T, 2), f"Expected (64, 2), got {labels.shape}"

    def test_features_are_finite(self):
        for i in range(len(self.ds)):
            feats, _ = self.ds[i]
            assert np.all(np.isfinite(feats)), f"Window {i} has non-finite features"

    def test_labels_are_binary(self):
        for i in range(len(self.ds)):
            _, labels = self.ds[i]
            unique = set(np.unique(labels))
            assert unique <= {0.0, 1.0}, f"Window {i} labels contain non-binary values: {unique}"

    def test_has_positive_windows(self):
        """Video 62805_6 has elbow errors → at least one positive window expected."""
        assert len(self.ds._pos_indices) > 0, "Expected positive windows for this video"

    def test_positive_windows_actually_contain_positive_frames(self):
        for idx in self.ds._pos_indices:
            _, labels = self.ds[idx]
            assert labels.any(), f"Positive window {idx} has all-zero labels"

    def test_sampler_weights_match_length(self):
        weights = self.ds.get_sampler_weights()
        assert len(weights) == len(self.ds)
        assert (weights > 0).all()
