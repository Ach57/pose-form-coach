"""Root conftest — shared fixtures available to all tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ── Paths ────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """pytest-provided temporary directory (auto-cleaned)."""
    return tmp_path


# ── Synthetic landmark factory ───────────────────────────────────────

@pytest.fixture
def make_landmarks():
    """Factory fixture: build (T, 33, 4) landmarks with controllable joint positions.

    Returns a callable(T, overrides) where overrides is a dict of
    {landmark_index: (x, y, z)} that are applied to every frame.
    Unset landmarks default to (0.5, 0.5, 0.0) with visibility=1.0.
    """
    def _factory(
        n_frames: int = 30,
        overrides: dict[int, tuple[float, float, float]] | None = None,
    ) -> np.ndarray:
        lm = np.full((n_frames, 33, 4), 0.5, dtype=np.float32)
        lm[..., 3] = 1.0  # visibility
        lm[..., 2] = 0.0  # z = 0
        if overrides:
            for idx, (x, y, z) in overrides.items():
                lm[:, idx, 0] = x
                lm[:, idx, 1] = y
                lm[:, idx, 2] = z
        return lm

    return _factory


# ── Label file helpers ───────────────────────────────────────────────

@pytest.fixture
def write_label_json(tmp_path: Path):
    """Write a per-video label JSON and return the path."""
    def _write(video_id: str, label_dict: dict, labels_dir: Path | None = None):
        d = labels_dir or tmp_path / "labels"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{video_id}.json"
        path.write_text(json.dumps(label_dict, indent=2))
        return path
    return _write


@pytest.fixture
def write_feature_npy(tmp_path: Path):
    """Write a per-video .npy feature file and return the path."""
    def _write(video_id: str, features: np.ndarray, features_dir: Path | None = None):
        d = features_dir or tmp_path / "features"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{video_id}.npy"
        np.save(path, features)
        return path
    return _write
