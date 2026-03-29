"""
PoseExtractor — Extracts MediaPipe Pose landmarks from video files.

Uses the mediapipe.tasks API (v0.10.x+).

Designed to be reused for:
  - Offline batch extraction (full videos → .npz caches)
  - Real-time single-frame extraction (webcam loop)

Usage (offline):
    extractor = PoseExtractor(model_path="artifacts/pose_landmarker_heavy.task")
    result = extractor.extract_video("path/to/video.mp4")
    extractor.save_cache(result, "data/poses/ohp/12345_1.npz")

Usage (real-time):
    extractor = PoseExtractor(model_path="artifacts/pose_landmarker_lite.task")
    extractor.open()
    landmarks = extractor.process_frame(frame)
    extractor.close()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class PoseResult:
    """Container for extracted pose data from a single video."""

    video_id: str
    fps: float
    landmarks: np.ndarray       # shape (T, 33, 4) — x, y, z, visibility
    n_frames: int
    failed_frames: np.ndarray   # boolean mask — True where detection failed

    @property
    def success_rate(self) -> float:
        return 1.0 - self.failed_frames.mean()


class PoseExtractor:
    """MediaPipe PoseLandmarker wrapper (tasks API) for offline and real-time use.

    Parameters
    ----------
    model_path : str | Path
        Path to .task model file. Relative paths resolved from repo root.
    min_detection_confidence : float
        Minimum confidence for initial pose detection.
    min_tracking_confidence : float
        Minimum confidence for frame-to-frame tracking.
    """

    def __init__(
        self,
        model_path: str | Path = "artifacts/pose_landmarker_heavy.task",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        path = Path(model_path)
        if not path.is_absolute():
            path = _REPO_ROOT / path
        self.model_path = str(path)
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._landmarker: PoseLandmarker | None = None

    # ── Offline extraction ───────────────────────────────────────────

    def extract_video(self, video_path: str | Path) -> PoseResult:
        """Extract landmarks for every frame of a video file.

        Returns a PoseResult with landmarks array (T, 33, 4).
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)

        all_landmarks: list[np.ndarray] = []
        failed: list[bool] = []

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        with PoseLandmarker.create_from_options(options) as landmarker:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                lm = self._detect_video(landmarker, frame, frame_idx, fps)
                if lm is not None:
                    all_landmarks.append(lm)
                    failed.append(False)
                else:
                    all_landmarks.append(np.zeros((33, 4), dtype=np.float32))
                    failed.append(True)
                frame_idx += 1

        cap.release()

        landmarks = np.stack(all_landmarks, axis=0)  # (T, 33, 4)
        failed_arr = np.array(failed, dtype=bool)

        # Interpolate short gaps of failed frames (≤3 consecutive)
        landmarks = self._interpolate_gaps(landmarks, failed_arr, max_gap=3)

        video_id = video_path.stem
        return PoseResult(
            video_id=video_id,
            fps=fps,
            landmarks=landmarks,
            n_frames=landmarks.shape[0],
            failed_frames=failed_arr,
        )

    # ── Real-time single-frame ───────────────────────────────────────

    def open(self) -> None:
        """Open a persistent session for real-time (IMAGE mode) use."""
        if self._landmarker is not None:
            self._landmarker.close()
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=self.min_detection_confidence,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

    def close(self) -> None:
        """Close the persistent session."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def process_frame(self, frame: np.ndarray) -> np.ndarray | None:
        """Process a single BGR frame → (33, 4) landmarks or None."""
        if self._landmarker is None:
            self.open()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        return self._result_to_array(result)

    # ── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _detect_video(
        landmarker: PoseLandmarker,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
    ) -> np.ndarray | None:
        """Run landmarker in VIDEO mode on one BGR frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_idx * 1000 / fps)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        return PoseExtractor._result_to_array(result)

    @staticmethod
    def _result_to_array(result) -> np.ndarray | None:
        """Convert PoseLandmarkerResult → (33, 4) float32 or None."""
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None
        lm_list = result.pose_landmarks[0]  # first (only) person
        arr = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in lm_list],
            dtype=np.float32,
        )
        return arr

    @staticmethod
    def _interpolate_gaps(
        landmarks: np.ndarray,
        failed: np.ndarray,
        max_gap: int = 3,
    ) -> np.ndarray:
        """Linearly interpolate short consecutive failed-frame gaps in-place."""
        T = len(failed)
        i = 0
        while i < T:
            if failed[i]:
                j = i
                while j < T and failed[j]:
                    j += 1
                gap = j - i
                if gap <= max_gap and i > 0 and j < T:
                    for k in range(i, j):
                        alpha = (k - i + 1) / (gap + 1)
                        landmarks[k] = (1 - alpha) * landmarks[i - 1] + alpha * landmarks[j]
                i = j
            else:
                i += 1
        return landmarks

    # ── Serialization ────────────────────────────────────────────────

    @staticmethod
    def save_cache(result: PoseResult, path: str | Path) -> None:
        """Save pose result to compressed .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            video_id=result.video_id,
            fps=result.fps,
            landmarks=result.landmarks,
            failed_frames=result.failed_frames,
        )

    @staticmethod
    def load_cache(path: str | Path) -> PoseResult:
        """Load a cached PoseResult from .npz."""
        data = np.load(path, allow_pickle=False)
        return PoseResult(
            video_id=str(data["video_id"]),
            fps=float(data["fps"]),
            landmarks=data["landmarks"],
            n_frames=data["landmarks"].shape[0],
            failed_frames=data["failed_frames"],
        )
