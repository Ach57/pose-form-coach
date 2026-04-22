"""Unit tests for src.utils.mediapipe_visualization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.utils.mediapipe_visualization import (
    BONES,
    BONE_COLORS,
    FEATURE_NAMES,
    KEY_FEATURES,
    load_T_frames,
    make_frame_traces,
    build_frames_for_slider,
    build_feature_timeline,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def poses_dir(tmp_path: Path) -> Path:
    return tmp_path / "poses"


@pytest.fixture
def features_dir(tmp_path: Path) -> Path:
    return tmp_path / "features"


@pytest.fixture
def labels_dir(tmp_path: Path) -> Path:
    return tmp_path / "labels"


@pytest.fixture
def dataset_cfg(tmp_path: Path) -> dict:
    (tmp_path / "poses").mkdir()
    (tmp_path / "features").mkdir()
    (tmp_path / "labels").mkdir()
    return {
        "fps": 30.0,
        "paths": {
            "poses_dir": str(tmp_path / "poses"),
            "features_dir": str(tmp_path / "features"),
            "labels_dir": str(tmp_path / "labels"),
        },
    }


def _write_pose(poses_dir: Path, video_id: str, T: int = 10) -> np.ndarray:
    poses_dir.mkdir(parents=True, exist_ok=True)
    lm = np.random.rand(T, 33, 4).astype(np.float32)
    np.savez(poses_dir / f"{video_id}.npz", landmarks=lm, fps=30.0, video_id=video_id)
    return lm


def _write_features(features_dir: Path, video_id: str, T: int = 10) -> np.ndarray:
    features_dir.mkdir(parents=True, exist_ok=True)
    feats = np.random.randn(T, 16).astype(np.float32)
    np.save(features_dir / f"{video_id}.npy", feats)
    return feats


def _write_labels(labels_dir: Path, video_id: str, label_dict: dict | None = None) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    d = label_dict or {"ohp_elbow": [], "ohp_knee": []}
    (labels_dir / f"{video_id}.json").write_text(json.dumps(d))


# ── Tests: load_T_frames ─────────────────────────────────────────────


class TestLoadTFrames:

    def test_returns_landmarks_and_frame_count(self, dataset_cfg):
        poses_dir = Path(dataset_cfg["paths"]["poses_dir"])
        expected = _write_pose(poses_dir, "vid1", T=20)
        lm, T = load_T_frames(dataset_cfg, "vid1")
        assert T == 20
        assert lm.shape == (20, 33, 4)

    def test_landmarks_are_copied(self, dataset_cfg):
        """Returned array must be a copy — not a memory-mapped view into closed file."""
        poses_dir = Path(dataset_cfg["paths"]["poses_dir"])
        _write_pose(poses_dir, "vid1", T=5)
        lm, _ = load_T_frames(dataset_cfg, "vid1")
        # Writing to the returned array must not raise (would fail if file-backed)
        lm[0, 0, 0] = 999.0
        assert lm[0, 0, 0] == 999.0

    def test_raises_if_poses_dir_missing(self, tmp_path):
        cfg = {"paths": {"poses_dir": str(tmp_path / "nonexistent")}}
        with pytest.raises(FileNotFoundError, match="Poses directory"):
            load_T_frames(cfg, "vid1")

    def test_raises_if_file_missing(self, dataset_cfg):
        with pytest.raises(FileNotFoundError, match="Sample video file"):
            load_T_frames(dataset_cfg, "does_not_exist")

    def test_raises_if_landmarks_key_missing(self, dataset_cfg):
        poses_dir = Path(dataset_cfg["paths"]["poses_dir"])
        np.savez(poses_dir / "vid1.npz", other_key=np.zeros((5, 33, 4)))
        with pytest.raises(KeyError, match="landmarks"):
            load_T_frames(dataset_cfg, "vid1")

    def test_raises_if_wrong_shape(self, dataset_cfg):
        poses_dir = Path(dataset_cfg["paths"]["poses_dir"])
        np.savez(poses_dir / "vid1.npz", landmarks=np.zeros((10, 17, 3)))
        with pytest.raises(ValueError, match="Invalid landmarks shape"):
            load_T_frames(dataset_cfg, "vid1")


# ── Tests: make_frame_traces ─────────────────────────────────────────


class TestMakeFrameTraces:

    def test_returns_traces_for_valid_frame(self):
        lm = np.random.rand(33, 4).astype(np.float32)
        traces = make_frame_traces(lm)
        assert len(traces) == 1 + len(BONES)

    def test_returns_empty_for_none(self):
        assert make_frame_traces(None) == []

    def test_returns_empty_for_empty_array(self):
        assert make_frame_traces(np.array([])) == []

    def test_raises_for_wrong_type(self):
        with pytest.raises(TypeError, match="np.ndarray"):
            make_frame_traces([[0.1, 0.2, 0.3, 1.0]] * 33)

    def test_raises_for_wrong_shape(self):
        with pytest.raises(ValueError, match="Invalid lm_frame shape"):
            make_frame_traces(np.zeros((33,)))

    def test_joint_trace_has_33_points(self):
        lm = np.random.rand(33, 4).astype(np.float32)
        traces = make_frame_traces(lm)
        joint_trace = traces[0]
        assert len(joint_trace.x) == 33

    def test_all_bone_regions_present(self):
        lm = np.random.rand(33, 4).astype(np.float32)
        traces = make_frame_traces(lm)
        trace_names = {t.name for t in traces}
        for region in BONES:
            assert region in trace_names


# ── Tests: build_frames_for_slider ───────────────────────────────────


class TestBuildFramesForSlider:

    def test_runs_without_error(self, dataset_cfg):
        poses_dir = Path(dataset_cfg["paths"]["poses_dir"])
        _write_pose(poses_dir, "vid1", T=5)
        # Should not raise
        build_frames_for_slider(dataset_cfg, "vid1")

    def test_propagates_file_not_found(self, dataset_cfg):
        with pytest.raises(FileNotFoundError):
            build_frames_for_slider(dataset_cfg, "missing")


# ── Tests: build_feature_timeline ────────────────────────────────────


class TestBuildFeatureTimeline:

    def test_runs_without_error(self, dataset_cfg):
        features_dir = Path(dataset_cfg["paths"]["features_dir"])
        labels_dir = Path(dataset_cfg["paths"]["labels_dir"])
        _write_features(features_dir, "vid1", T=30)
        _write_labels(labels_dir, "vid1", {"ohp_elbow": [[0.5, 1.0]], "ohp_knee": []})
        build_feature_timeline(dataset_cfg, "vid1")

    def test_raises_if_features_dir_missing(self, tmp_path):
        cfg = {
            "fps": 30.0,
            "paths": {
                "poses_dir": str(tmp_path),
                "features_dir": str(tmp_path / "nope"),
                "labels_dir": str(tmp_path),
            },
        }
        with pytest.raises(FileNotFoundError, match="Features directory"):
            build_feature_timeline(cfg, "vid1")

    def test_raises_if_feature_file_missing(self, dataset_cfg):
        with pytest.raises(FileNotFoundError, match="Feature file"):
            build_feature_timeline(dataset_cfg, "missing")

    def test_raises_if_labels_dir_missing(self, dataset_cfg, tmp_path):
        features_dir = Path(dataset_cfg["paths"]["features_dir"])
        _write_features(features_dir, "vid1", T=10)
        cfg = {
            **dataset_cfg,
            "paths": {**dataset_cfg["paths"], "labels_dir": str(tmp_path / "nope_labels")},
        }
        with pytest.raises(FileNotFoundError, match="Labels directory"):
            build_feature_timeline(cfg, "vid1")

    def test_raises_if_label_file_missing(self, dataset_cfg):
        features_dir = Path(dataset_cfg["paths"]["features_dir"])
        _write_features(features_dir, "vid1", T=10)
        with pytest.raises(FileNotFoundError, match="Label file"):
            build_feature_timeline(dataset_cfg, "vid1")

    def test_raises_if_feature_index_out_of_bounds(self, dataset_cfg, monkeypatch):
        features_dir = Path(dataset_cfg["paths"]["features_dir"])
        labels_dir = Path(dataset_cfg["paths"]["labels_dir"])
        _write_features(features_dir, "vid1", T=5)
        _write_labels(labels_dir, "vid1")
        # Temporarily override KEY_FEATURES with an out-of-bounds index
        import src.utils.mediapipe_visualization as mv
        monkeypatch.setattr(mv, "KEY_FEATURES", [999])
        with pytest.raises(IndexError, match="out of bounds"):
            build_feature_timeline(dataset_cfg, "vid1")


# ── Tests: module-level constants ────────────────────────────────────


class TestConstants:

    def test_feature_names_count(self):
        assert len(FEATURE_NAMES) == 16

    def test_bone_colors_cover_all_regions(self):
        assert set(BONE_COLORS.keys()) == set(BONES.keys())

    def test_key_features_within_bounds(self):
        assert all(0 <= fi < 16 for fi in KEY_FEATURES)