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

from pathlib import Path
import zipfile


def load_yaml(path: str | Path) -> dict:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def merge_configs(*configs: dict) -> dict:
    """Deep-merge a sequence of dicts, later values overriding earlier ones.

    Only dict values are merged recursively; all other types are replaced.

    Example
    -------
    >>> base  = {"model": {"channels": [64,64], "dropout": 0.1}}
    >>> override = {"model": {"dropout": 0.2, "in_features": 16}}
    >>> merge_configs(base, override)
    {"model": {"channels": [64,64], "dropout": 0.2, "in_features": 16}}
    """
    result: dict = {}
    for cfg in configs:
        for key, val in cfg.items():
            if key in result and isinstance(result[key], dict) and isinstance(val, dict):
                result[key] = merge_configs(result[key], val)
            else:
                result[key] = val
    return result


def load_config(
    *paths: str | Path,
    base_dir: str | Path | None = None,
) -> dict:
    """Load and deep-merge one or more YAML config files.

    Files are merged left-to-right so later files override earlier ones.
    Typical use:

        # Exercise-specific — shared base merged with per-exercise overrides
        cfg = load_config(
            "configs/shared/model.tcn.base.yaml",
            "configs/ohp/model.yaml",
        )

        # Dataset only (no base needed)
        cfg = load_config("configs/ohp/dataset.yaml")

    Parameters
    ----------
    *paths : str | Path
        One or more YAML file paths. Resolved relative to `base_dir` if given.
    base_dir : str | Path | None
        Optional base directory for relative paths. Defaults to cwd.

    Returns
    -------
    dict
        Deep-merged configuration dictionary.
    """
    merged: dict = {}
    for p in paths:
        p = Path(p)
        if base_dir is not None and not p.is_absolute():
            p = Path(base_dir) / p
        merged = merge_configs(merged, load_yaml(p))
    return merged


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


def unzip_file(zip_path: Path, dest_dir: Path, verbose: bool = True) -> None:    
    """
    Unzip a zip file into a destination directory.

    Args:
        zip_path (Path): Path to the .zip file
        dest_dir (Path): Destination directory
    """

    if not zip_path.is_file():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:        
        print(f"Unzipping {zip_path.name} → {dest_dir}")

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest_dir)
        