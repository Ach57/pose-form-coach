"""
Shared I/O and data conversion utilities.

Functions here are used across the entire pipeline — preprocessing, training,
evaluation, and real-time inference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml


def load_yaml(path: str | Path) -> dict:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_json(path: str | Path) -> dict | list:
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def save_json(data: dict | list, path: str | Path) -> None:
    """Save data to a JSON file with readable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def rasterize_segments(
    segments: list[list[float]],
    n_frames: int,
    fps: float,
) -> np.ndarray:
    """Convert time-range segments to a frame-wise binary mask.

    Parameters
    ----------
    segments : list of [start_sec, end_sec]
        Temporal segments where the error is active.
    n_frames : int
        Total number of frames in the video.
    fps : float
        Frame rate.

    Returns
    -------
    mask : np.ndarray of shape (n_frames,), dtype bool
        True at frames within any segment.
    """
    mask = np.zeros(n_frames, dtype=bool)
    for start_sec, end_sec in segments:
        start_frame = int(round(start_sec * fps))
        end_frame = int(round(end_sec * fps))
        start_frame = max(0, start_frame)
        end_frame = min(n_frames, end_frame)
        mask[start_frame:end_frame] = True
    return mask


def rasterize_multilabel(
    label_dict: dict[str, list[list[float]]],
    n_frames: int,
    fps: float,
    label_order: list[str],
) -> np.ndarray:
    """Rasterize multi-label segments to a frame-wise multi-hot matrix.

    Parameters
    ----------
    label_dict : {label_name: [[start, end], ...]}
    n_frames : int
    fps : float
    label_order : list of label names (defines column order)

    Returns
    -------
    labels : np.ndarray of shape (n_frames, L), dtype float32
        Multi-hot labels per frame.
    """
    L = len(label_order)
    labels = np.zeros((n_frames, L), dtype=np.float32)
    for i, name in enumerate(label_order):
        if name in label_dict:
            labels[:, i] = rasterize_segments(label_dict[name], n_frames, fps).astype(np.float32)
    return labels
