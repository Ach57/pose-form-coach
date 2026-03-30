"""Unit tests for src.utils.io — rasterize_segments and rasterize_multilabel."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.io import rasterize_multilabel, rasterize_segments


class TestRasterizeSegments:
    """Tests for rasterize_segments (time ranges → frame-wise bool mask)."""

    def test_empty_segments_all_false(self):
        mask = rasterize_segments([], n_frames=90, fps=30.0)
        assert mask.shape == (90,)
        assert mask.dtype == bool
        assert not mask.any()

    def test_single_segment(self):
        # 1.0s–2.0s at 30 FPS → frames 30..59 active
        mask = rasterize_segments([[1.0, 2.0]], n_frames=90, fps=30.0)
        assert mask[30]
        assert mask[59]
        assert not mask[29]
        assert not mask[60]

    def test_multiple_segments_no_overlap(self):
        segs = [[0.0, 0.5], [2.0, 2.5]]
        mask = rasterize_segments(segs, n_frames=90, fps=30.0)
        # First segment: frames 0..14
        assert mask[0]
        assert mask[14]
        assert not mask[15]
        # Second segment: frames 60..74
        assert mask[60]
        assert mask[74]
        assert not mask[75]
        # Gap between is false
        assert not mask[30]

    def test_segment_clipped_to_n_frames(self):
        # Segment extends beyond video length
        mask = rasterize_segments([[0.0, 100.0]], n_frames=10, fps=30.0)
        assert mask.shape == (10,)
        assert mask.all()  # entire video is active

    def test_segment_before_video_start(self):
        # Negative start gets clamped to 0
        mask = rasterize_segments([[-1.0, 0.5]], n_frames=30, fps=30.0)
        assert mask[0]
        assert mask[14]
        assert not mask[15]

    def test_zero_length_segment(self):
        # [1.0, 1.0] → no frames activated (start == end)
        mask = rasterize_segments([[1.0, 1.0]], n_frames=90, fps=30.0)
        assert not mask.any()

    def test_frame_count_matches(self):
        # A 1-second segment at 30 FPS should activate exactly 30 frames
        mask = rasterize_segments([[0.0, 1.0]], n_frames=100, fps=30.0)
        assert mask.sum() == 30


class TestRasterizeMultilabel:
    """Tests for rasterize_multilabel (dict of segments → multi-hot matrix)."""

    def test_shape_and_dtype(self):
        labels = rasterize_multilabel(
            {"a": [[0.0, 1.0]], "b": []},
            n_frames=30,
            fps=30.0,
            label_order=["a", "b"],
        )
        assert labels.shape == (30, 2)
        assert labels.dtype == np.float32

    def test_column_order_matches_label_order(self):
        label_dict = {"knee": [[0.0, 1.0]], "elbow": []}
        # Order: elbow first, knee second
        labels = rasterize_multilabel(label_dict, 30, 30.0, ["elbow", "knee"])
        # elbow column (0) should be all zeros
        assert labels[:, 0].sum() == 0
        # knee column (1) should be all ones
        assert labels[:, 1].sum() == 30

    def test_missing_label_key_is_all_zero(self):
        # "unknown" is not in label_dict → should be zeros
        labels = rasterize_multilabel(
            {"a": [[0.0, 1.0]]},
            n_frames=30,
            fps=30.0,
            label_order=["a", "unknown"],
        )
        assert labels[:, 1].sum() == 0

    def test_multi_label_overlap(self):
        # Both labels active in same time range
        label_dict = {"a": [[0.0, 0.5]], "b": [[0.0, 0.5]]}
        labels = rasterize_multilabel(label_dict, 30, 30.0, ["a", "b"])
        # First 15 frames should be [1, 1]
        assert (labels[:15] == 1.0).all()
        # Rest should be [0, 0]
        assert (labels[15:] == 0.0).all()
