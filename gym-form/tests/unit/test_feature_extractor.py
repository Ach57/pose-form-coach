"""Unit tests for src.features.feature_extractor — geometry, normalization, OHP features."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.features.feature_extractor import (
    LANDMARK,
    FeatureExtractor,
    OHPFeatureExtractor,
)


# ── Geometry helpers ─────────────────────────────────────────────────


class TestAngleBetween:
    """Tests for FeatureExtractor.angle_between (static method)."""

    def test_right_angle(self):
        # Points forming a 90° angle at B
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[0.0, 0.0, 0.0]])
        c = np.array([[0.0, 1.0, 0.0]])
        angle = FeatureExtractor.angle_between(a, b, c)
        assert angle.shape == (1,)
        assert abs(angle[0] - math.pi / 2) < 1e-5

    def test_straight_line(self):
        # Points in a line → 180°
        a = np.array([[-1.0, 0.0, 0.0]])
        b = np.array([[0.0, 0.0, 0.0]])
        c = np.array([[1.0, 0.0, 0.0]])
        angle = FeatureExtractor.angle_between(a, b, c)
        assert abs(angle[0] - math.pi) < 1e-3

    def test_zero_angle(self):
        # Same direction → 0°
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[0.0, 0.0, 0.0]])
        c = np.array([[2.0, 0.0, 0.0]])
        angle = FeatureExtractor.angle_between(a, b, c)
        assert abs(angle[0]) < 1e-3

    def test_batch_of_frames(self):
        # Multiple frames at once
        T = 10
        a = np.tile([[1.0, 0.0, 0.0]], (T, 1))
        b = np.zeros((T, 3))
        c = np.tile([[0.0, 1.0, 0.0]], (T, 1))
        angles = FeatureExtractor.angle_between(a, b, c)
        assert angles.shape == (T,)
        assert np.allclose(angles, math.pi / 2, atol=1e-5)


# ── Normalization ────────────────────────────────────────────────────


class TestNormalizeLandmarks:
    """Tests for FeatureExtractor.normalize_landmarks."""

    def test_hip_center_is_origin(self, make_landmarks):
        # Place hips at (0.3, 0.6, 0) and (0.7, 0.6, 0)
        lm = make_landmarks(
            n_frames=5,
            overrides={
                LANDMARK["hip_L"]: (0.3, 0.6, 0.0),
                LANDMARK["hip_R"]: (0.7, 0.6, 0.0),
            },
        )
        norm, sw = FeatureExtractor.normalize_landmarks(lm)
        # Mid-hip should be at origin in normalized space
        mid_hip = (norm[:, LANDMARK["hip_L"], :3] + norm[:, LANDMARK["hip_R"], :3]) / 2.0
        assert np.allclose(mid_hip, 0.0, atol=1e-5)

    def test_visibility_preserved(self, make_landmarks):
        lm = make_landmarks(n_frames=5)
        lm[:, 0, 3] = 0.3  # set nose visibility low
        norm, _ = FeatureExtractor.normalize_landmarks(lm)
        assert np.allclose(norm[:, 0, 3], 0.3)

    def test_shoulder_width_positive(self, make_landmarks):
        lm = make_landmarks(
            n_frames=5,
            overrides={
                LANDMARK["shoulder_L"]: (0.3, 0.4, 0.0),
                LANDMARK["shoulder_R"]: (0.7, 0.4, 0.0),
            },
        )
        _, sw = FeatureExtractor.normalize_landmarks(lm)
        assert (sw > 0).all()


# ── Finite diff ──────────────────────────────────────────────────────


class TestFiniteDiff:
    """Tests for FeatureExtractor.finite_diff."""

    def test_constant_signal_zero_velocity(self):
        signal = np.ones((30, 3), dtype=np.float32)
        vel = FeatureExtractor.finite_diff(signal, fps=30.0)
        assert vel.shape == signal.shape
        assert np.allclose(vel, 0.0, atol=1e-5)

    def test_linear_signal_constant_velocity(self):
        # Linear ramp: 0, 1, 2, ... at 30 FPS → velocity = 30/s everywhere
        t = np.arange(30, dtype=np.float32).reshape(-1, 1)
        vel = FeatureExtractor.finite_diff(t, fps=30.0)
        # Interior points should be exactly 30.0
        assert np.allclose(vel[1:-1], 30.0, atol=1e-3)

    def test_shape_preserved(self):
        signal = np.random.randn(50, 8).astype(np.float32)
        vel = FeatureExtractor.finite_diff(signal, fps=30.0)
        assert vel.shape == (50, 8)


# ── OHPFeatureExtractor ─────────────────────────────────────────────


class TestOHPFeatureExtractor:
    """Tests for the full OHP feature pipeline."""

    @pytest.fixture
    def extractor(self):
        return OHPFeatureExtractor(fps=30.0, smooth_window=0)  # no smoothing for tests

    def test_output_shape(self, extractor, make_landmarks):
        lm = make_landmarks(n_frames=50)
        features = extractor.extract(lm)
        assert features.shape == (50, 16)
        assert features.dtype == np.float32

    def test_feature_count_matches_names(self, extractor):
        assert extractor.n_features == 16
        assert len(extractor.feature_names) == 16

    def test_feature_names_order(self, extractor):
        names = extractor.feature_names
        # First 8 are static features
        assert names[0] == "knee_angle_L"
        assert names[4] == "trunk_inclination"
        assert names[7] == "hip_center_y"
        # Next 7 are velocities
        assert names[8] == "d_knee_angle_L"
        assert names[14] == "d_wrist_y_R"
        # Last is scale
        assert names[15] == "shoulder_width"

    def test_straight_limb_gives_pi(self, extractor):
        """When hip-knee-ankle are collinear, knee angle should be ~π."""
        lm = np.full((10, 33, 4), 0.5, dtype=np.float32)
        lm[..., 3] = 1.0
        lm[..., 2] = 0.0

        # Place left leg in a straight line (y-axis)
        lm[:, LANDMARK["hip_L"], :3] = [0.4, 0.3, 0.0]
        lm[:, LANDMARK["knee_L"], :3] = [0.4, 0.5, 0.0]
        lm[:, LANDMARK["ankle_L"], :3] = [0.4, 0.7, 0.0]

        # Need valid hips/shoulders for normalization
        lm[:, LANDMARK["hip_R"], :3] = [0.6, 0.3, 0.0]
        lm[:, LANDMARK["shoulder_L"], :3] = [0.4, 0.2, 0.0]
        lm[:, LANDMARK["shoulder_R"], :3] = [0.6, 0.2, 0.0]

        features = extractor.extract(lm)
        knee_angle_L = features[:, 0]  # first column
        assert np.allclose(knee_angle_L, math.pi, atol=0.05)

    def test_shoulder_width_positive(self, extractor, make_landmarks):
        lm = make_landmarks(
            n_frames=10,
            overrides={
                LANDMARK["shoulder_L"]: (0.3, 0.4, 0.0),
                LANDMARK["shoulder_R"]: (0.7, 0.4, 0.0),
                LANDMARK["hip_L"]: (0.4, 0.6, 0.0),
                LANDMARK["hip_R"]: (0.6, 0.6, 0.0),
            },
        )
        features = extractor.extract(lm)
        shoulder_width = features[:, 15]
        assert (shoulder_width > 0).all()

    def test_static_landmarks_zero_velocity(self, extractor, make_landmarks):
        """If landmarks don't move, all velocity features should be ~0."""
        lm = make_landmarks(n_frames=30)
        features = extractor.extract(lm)
        velocities = features[:, 8:15]  # columns 8..14 are velocities
        assert np.allclose(velocities, 0.0, atol=1e-3)

    def test_smoothing_does_not_change_shape(self, make_landmarks):
        extractor = OHPFeatureExtractor(fps=30.0, smooth_window=5)
        lm = make_landmarks(n_frames=30)
        features = extractor.extract(lm)
        assert features.shape == (30, 16)

    def test_two_frame_no_crash(self, extractor):
        """A 2-frame video (minimum for np.gradient) should not crash."""
        lm = np.full((2, 33, 4), 0.5, dtype=np.float32)
        lm[..., 3] = 1.0
        features = extractor.extract(lm)
        assert features.shape == (2, 16)
