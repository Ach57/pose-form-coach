"""
FeatureExtractor — Converts raw pose landmarks into kinematic feature vectors.

Base class provides shared normalization logic. Subclass per exercise to define
which angles / velocities to compute.

Usage:
    extractor = OHPFeatureExtractor(fps=30.0)
    features = extractor.extract(landmarks)  # (T, 33, 4) → (T, F)
    names = extractor.feature_names           # list of F feature names
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.signal import savgol_filter

# MediaPipe Pose landmark indices (subset we use)
LANDMARK = {
    "nose": 0,
    "shoulder_L": 11,
    "shoulder_R": 12,
    "elbow_L": 13,
    "elbow_R": 14,
    "wrist_L": 15,
    "wrist_R": 16,
    "hip_L": 23,
    "hip_R": 24,
    "knee_L": 25,
    "knee_R": 26,
    "ankle_L": 27,
    "ankle_R": 28,
}


class FeatureExtractor(ABC):
    """Base class for per-exercise feature extraction.

    Subclasses must implement `_compute_frame_features` and `feature_names`.

    Parameters
    ----------
    fps : float
        Frame rate of the source video (for velocity scaling).
    smooth_window : int
        Savitzky-Golay window length for optional landmark smoothing.
        Set to 0 to disable.
    """

    def __init__(self, fps: float = 30.0, smooth_window: int = 5) -> None:
        self.fps = fps
        self.smooth_window = smooth_window

    # ── Public API ──

    def extract(self, landmarks: np.ndarray) -> np.ndarray:
        """Convert raw landmarks (T, 33, 4) → feature matrix (T, F).

        Steps:
        1. Normalize landmarks (center on mid-hip, scale by shoulder width).
        2. Optionally smooth landmarks.
        3. Compute per-frame features (angles, positions).
        4. Compute velocities (first derivative).
        5. Concatenate → (T, F).
        """
        norm, shoulder_w = self.normalize_landmarks(landmarks)

        if self.smooth_window >= 3 and norm.shape[0] >= self.smooth_window:
            norm = self._smooth(norm)

        static = self._compute_frame_features(norm)  # (T, F_static)
        velocities = self.finite_diff(static, self.fps)  # (T, F_static)

        # Select only the velocity columns we want (exclude positional references)
        vel_cols = self._velocity_column_indices()
        vel = velocities[:, vel_cols]  # (T, F_vel)

        # Shoulder width as scale reference
        sw = shoulder_w[:, np.newaxis]  # (T, 1)

        features = np.concatenate([static, vel, sw], axis=1)  # (T, F)
        return features.astype(np.float32)

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Ordered list of feature names matching columns of extract() output."""
        ...

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    # ── Shared geometry helpers (reusable across exercises) ──

    @staticmethod
    def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Compute angle ∠ABC at vertex B for arrays of shape (..., 2 or 3).

        Returns angles in radians, shape (...,).
        """
        ba = a - b
        bc = c - b
        cos_angle = np.sum(ba * bc, axis=-1) / (
            np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8
        )
        return np.arccos(np.clip(cos_angle, -1.0, 1.0))

    @staticmethod
    def normalize_landmarks(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Center on mid-hip, scale by shoulder width.

        Parameters
        ----------
        landmarks : (T, 33, 4)

        Returns
        -------
        normalized : (T, 33, 4) — x,y,z normalized; visibility unchanged
        shoulder_widths : (T,)
        """
        xy = landmarks[..., :3].copy()
        vis = landmarks[..., 3:4]

        mid_hip = (xy[:, LANDMARK["hip_L"]] + xy[:, LANDMARK["hip_R"]]) / 2.0
        xy -= mid_hip[:, np.newaxis, :]

        shoulder_w = np.linalg.norm(
            xy[:, LANDMARK["shoulder_L"]] - xy[:, LANDMARK["shoulder_R"]], axis=-1
        )
        shoulder_w = np.maximum(shoulder_w, 1e-6)  # avoid division by zero

        xy /= shoulder_w[:, np.newaxis, np.newaxis]

        normalized = np.concatenate([xy, vis], axis=-1)
        return normalized, shoulder_w

    @staticmethod
    def finite_diff(signal: np.ndarray, fps: float) -> np.ndarray:
        """Central finite difference for velocity; shape preserved via padding."""
        dt = 1.0 / fps
        vel = np.gradient(signal, dt, axis=0)
        return vel

    def _smooth(self, landmarks: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay filter to landmark coords (preserves visibility)."""
        T, J, C = landmarks.shape
        out = landmarks.copy()
        for j in range(J):
            for c in range(3):  # smooth x, y, z only
                out[:, j, c] = savgol_filter(
                    landmarks[:, j, c],
                    window_length=self.smooth_window,
                    polyorder=2,
                )
        return out

    # ── Subclass hooks ──

    @abstractmethod
    def _compute_frame_features(self, norm_landmarks: np.ndarray) -> np.ndarray:
        """Compute per-frame static features (angles, positions).

        Parameters
        ----------
        norm_landmarks : (T, 33, 4) — already normalized

        Returns
        -------
        features : (T, F_static)
        """
        ...

    @abstractmethod
    def _velocity_column_indices(self) -> list[int]:
        """Return column indices of static features that get velocity derivatives."""
        ...


class OHPFeatureExtractor(FeatureExtractor):
    """Feature extractor for Overhead Press.

    Produces 16 features per frame:
      8 static (5 angles + 3 positions) + 7 velocities + 1 scale.

    Static features (columns 0-7):
      0: knee_angle_L
      1: knee_angle_R
      2: elbow_angle_L
      3: elbow_angle_R
      4: trunk_inclination
      5: wrist_y_L
      6: wrist_y_R
      7: hip_center_y

    Velocities are computed for columns 0-6 (all except hip_center_y).
    Shoulder width is appended as the final column.
    """

    _STATIC_NAMES = [
        "knee_angle_L",
        "knee_angle_R",
        "elbow_angle_L",
        "elbow_angle_R",
        "trunk_inclination",
        "wrist_y_L",
        "wrist_y_R",
        "hip_center_y",
    ]

    _VEL_INDICES = [0, 1, 2, 3, 4, 5, 6]  # all except hip_center_y (7)

    @property
    def feature_names(self) -> list[str]:
        return [
            "knee_angle_L",
            "knee_angle_R",
            "elbow_angle_L",
            "elbow_angle_R",
            "trunk_inclination",
            "wrist_y_L",
            "wrist_y_R",
            "hip_center_y",
            "d_knee_angle_L",
            "d_knee_angle_R",
            "d_elbow_angle_L",
            "d_elbow_angle_R",
            "d_trunk_inclination",
            "d_wrist_y_L",
            "d_wrist_y_R",
            "shoulder_width",
        ]

    def _velocity_column_indices(self) -> list[int]:
        return self._VEL_INDICES

    def _compute_frame_features(self, norm_landmarks: np.ndarray) -> np.ndarray:
        """Compute OHP-specific angles and positions.

        Parameters
        ----------
        norm_landmarks : (T, 33, 4) — normalized coords

        Returns
        -------
        features : (T, 8) — static features before velocity/scale append
        """
        T = norm_landmarks.shape[0]
        xyz = norm_landmarks[..., :3]  # (T, 33, 3)

        def lm(name: str) -> np.ndarray:
            return xyz[:, LANDMARK[name]]  # (T, 3)

        # Joint angles
        knee_angle_L = self.angle_between(lm("hip_L"), lm("knee_L"), lm("ankle_L"))
        knee_angle_R = self.angle_between(lm("hip_R"), lm("knee_R"), lm("ankle_R"))
        elbow_angle_L = self.angle_between(lm("shoulder_L"), lm("elbow_L"), lm("wrist_L"))
        elbow_angle_R = self.angle_between(lm("shoulder_R"), lm("elbow_R"), lm("wrist_R"))

        # Trunk inclination: angle between spine vector and vertical
        mid_hip = (lm("hip_L") + lm("hip_R")) / 2.0
        mid_shoulder = (lm("shoulder_L") + lm("shoulder_R")) / 2.0
        spine = mid_shoulder - mid_hip  # (T, 3)
        vertical = np.zeros_like(spine)
        vertical[:, 1] = -1.0  # MediaPipe y-axis points downward, so up = -y
        cos_trunk = np.sum(spine * vertical, axis=-1) / (
            np.linalg.norm(spine, axis=-1) + 1e-8
        )
        trunk_inclination = np.arccos(np.clip(cos_trunk, -1.0, 1.0))

        # Positional features (y-component, already normalized)
        wrist_y_L = lm("wrist_L")[:, 1]
        wrist_y_R = lm("wrist_R")[:, 1]
        hip_center_y = mid_hip[:, 1]

        features = np.stack(
            [
                knee_angle_L,
                knee_angle_R,
                elbow_angle_L,
                elbow_angle_R,
                trunk_inclination,
                wrist_y_L,
                wrist_y_R,
                hip_center_y,
            ],
            axis=1,
        )  # (T, 8)
        return features


class SquatFeatureExtractor(FeatureExtractor):
    """Feature extractor for Squat.

    Detects two error types:
      - error_knees_forward : knees travel too far forward past the toes
      - error_knees_inward  : knees cave inward (valgus collapse)

    Produces 18 features per frame:
      10 static + 8 velocities

    Static features (columns 0-9):
      0: knee_angle_L          — ∠(hip_L, knee_L, ankle_L)
      1: knee_angle_R          — ∠(hip_R, knee_R, ankle_R)
      2: hip_angle_L           — ∠(shoulder_L, hip_L, knee_L) — squat depth
      3: hip_angle_R           — ∠(shoulder_R, hip_R, knee_R)
      4: trunk_inclination     — angle between spine and vertical
      5: knee_x_L              — lateral knee position (normalized)
      6: knee_x_R              — lateral knee position (normalized)
      7: ankle_x_L             — lateral ankle position (normalized)
      8: ankle_x_R             — lateral ankle position (normalized)
      9: hip_center_y          — vertical hip position (squat depth proxy)

    Velocities (columns 10-17): d/dt of columns 0-7 (angles + lateral positions).
    Column 9 (hip_center_y) is excluded from velocities.
    Shoulder width appended as column 18 (scale reference).

    Total: 10 static + 8 velocities + 1 scale = 19 features.
    """

    _STATIC_NAMES = [
        "knee_angle_L",
        "knee_angle_R",
        "hip_angle_L",
        "hip_angle_R",
        "trunk_inclination",
        "knee_x_L",
        "knee_x_R",
        "ankle_x_L",
        "ankle_x_R",
        "hip_center_y",
    ]

    # Columns 0-8 get velocity derivatives; hip_center_y (9) excluded
    _VEL_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    @property
    def feature_names(self) -> list[str]:
        return [
            "knee_angle_L",
            "knee_angle_R",
            "hip_angle_L",
            "hip_angle_R",
            "trunk_inclination",
            "knee_x_L",
            "knee_x_R",
            "ankle_x_L",
            "ankle_x_R",
            "hip_center_y",
            "d_knee_angle_L",
            "d_knee_angle_R",
            "d_hip_angle_L",
            "d_hip_angle_R",
            "d_trunk_inclination",
            "d_knee_x_L",
            "d_knee_x_R",
            "d_ankle_x_L",
            "d_ankle_x_R",
            "shoulder_width",
        ]  # 20 features total

    def _velocity_column_indices(self) -> list[int]:
        return self._VEL_INDICES

    def _compute_frame_features(self, norm_landmarks: np.ndarray) -> np.ndarray:
        """Compute squat-specific joint angles and lateral positions.

        Parameters
        ----------
        norm_landmarks : (T, 33, 4) — already normalized (centered on mid-hip,
            scaled by shoulder width)

        Returns
        -------
        features : (T, 10)
        """
        xyz = norm_landmarks[..., :3]  # (T, 33, 3)

        def lm(name: str) -> np.ndarray:
            return xyz[:, LANDMARK[name]]  # (T, 3)

        # ── Joint angles ──────────────────────────────────────────────────────
        knee_angle_L = self.angle_between(lm("hip_L"), lm("knee_L"), lm("ankle_L"))
        knee_angle_R = self.angle_between(lm("hip_R"), lm("knee_R"), lm("ankle_R"))

        # Hip (pelvis) angle reflects squat depth
        hip_angle_L = self.angle_between(lm("shoulder_L"), lm("hip_L"), lm("knee_L"))
        hip_angle_R = self.angle_between(lm("shoulder_R"), lm("hip_R"), lm("knee_R"))

        # Trunk inclination: spine vs vertical
        mid_hip = (lm("hip_L") + lm("hip_R")) / 2.0
        mid_shoulder = (lm("shoulder_L") + lm("shoulder_R")) / 2.0
        spine = mid_shoulder - mid_hip
        vertical = np.zeros_like(spine)
        vertical[:, 1] = -1.0  # MediaPipe y↓, so up = -y
        cos_trunk = np.sum(spine * vertical, axis=-1) / (
            np.linalg.norm(spine, axis=-1) + 1e-8
        )
        trunk_inclination = np.arccos(np.clip(cos_trunk, -1.0, 1.0))

        # ── Lateral positions (x-axis) — key for knees-forward & valgus ──────
        # x is normalized by shoulder width; positive = right in image space
        knee_x_L  = lm("knee_L")[:, 0]
        knee_x_R  = lm("knee_R")[:, 0]
        ankle_x_L = lm("ankle_L")[:, 0]
        ankle_x_R = lm("ankle_R")[:, 0]

        # Vertical hip depth (proxy for squat depth)
        hip_center_y = mid_hip[:, 1]

        features = np.stack(
            [
                knee_angle_L,
                knee_angle_R,
                hip_angle_L,
                hip_angle_R,
                trunk_inclination,
                knee_x_L,
                knee_x_R,
                ankle_x_L,
                ankle_x_R,
                hip_center_y,
            ],
            axis=1,
        )  # (T, 10)
        return features
