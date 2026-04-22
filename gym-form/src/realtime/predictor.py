"""
Predictors for live gym form error detection.

Provides ``BasePredictor`` (shared inference logic) and exercise-specific
subclasses ``OHPPredictor`` and ``SquatPredictor``.

Usage
-----
    predictor = OHPPredictor.from_checkpoint(
        checkpoint_path="checkpoints/best.pt",
        model_path="artifacts/pose_landmarker_heavy.task",
        device="cpu",
    )
    with predictor:
        probs = predictor.predict(frame)   # np.ndarray (2,) or None

Label order
-----------
    OHPPredictor:   probs[0] → ohp_elbow,          probs[1] → ohp_knee
    SquatPredictor: probs[0] → error_knees_forward, probs[1] → error_knees_inward
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from src.extract.pose_extractor import PoseExtractor
from src.features.feature_extractor import OHPFeatureExtractor, SquatFeatureExtractor
from src.models.causal_tcn import CausalTCN

# Receptive field of the default model config (1 + 2*(1+2+4)*2 = 29 frames)
_DEFAULT_RF = 29

# Kept for backward compatibility — prefer predictor.labels
LABELS = ["ohp_elbow", "ohp_knee"]


class BasePredictor:
    """Shared inference logic for all exercise predictors.

    Subclasses must set the ``labels`` class variable and implement
    ``from_checkpoint()``.
    """

    labels: list[str] = []

    def __init__(
        self,
        model: CausalTCN,
        pose_extractor: PoseExtractor,
        feature_extractor,
        device: torch.device,
        buffer_size: int = _DEFAULT_RF,
        threshold: float = 0.5,
    ) -> None:
        self.model = model
        self.pose_extractor = pose_extractor
        self.feature_extractor = feature_extractor
        self.device = device
        self.threshold = threshold

        # Rolling buffer of (33, 4) landmark frames
        self._lm_buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._buffer_size = buffer_size

    # ── Session management ────────────────────────────────────────────

    def open(self) -> None:
        """Open MediaPipe session. Call before the camera loop."""
        self.pose_extractor.open()
        self._lm_buffer.clear()

    def close(self) -> None:
        """Close MediaPipe session. Call after the camera loop."""
        self.pose_extractor.close()

    def __enter__(self) -> "BasePredictor":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Per-frame inference ───────────────────────────────────────────

    def predict(self, frame: np.ndarray) -> np.ndarray | None:
        """Run inference on one BGR frame.

        Returns
        -------
        probs : np.ndarray shape (n_labels,) or None
            Sigmoid probabilities for each error label.
            Returns ``None`` until the rolling buffer is full (warm-up phase).
        """
        lm = self.pose_extractor.process_frame(frame)
        if lm is None:
            lm = self._lm_buffer[-1].copy() if self._lm_buffer else np.zeros((33, 4), dtype=np.float32)

        self._lm_buffer.append(lm)

        if len(self._lm_buffer) < self._buffer_size:
            return None

        landmarks_window = np.stack(list(self._lm_buffer), axis=0)
        features = self.feature_extractor.extract(landmarks_window)

        x = torch.from_numpy(features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits)

        return probs[0, -1].cpu().numpy()

    # ── Convenience helpers ───────────────────────────────────────────

    def label_errors(self, probs: np.ndarray) -> list[str]:
        """Return list of active error labels given a probability vector."""
        return [label for label, p in zip(self.labels, probs) if p >= self.threshold]

    @property
    def is_warm(self) -> bool:
        """True once the rolling buffer has filled (warm-up complete)."""
        return len(self._lm_buffer) >= self._buffer_size


class OHPPredictor(BasePredictor):
    """Per-frame OHP form error predictor."""

    labels = ["ohp_elbow", "ohp_knee"]

    # ── Factory ──────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_path: str | Path = "artifacts/pose_landmarker_heavy.task",
        fps: float = 30.0,
        device: str | torch.device = "cpu",
        buffer_size: int = _DEFAULT_RF,
        threshold: float = 0.5,
    ) -> "OHPPredictor":
        """Build an OHPPredictor by loading a saved checkpoint.

        Parameters
        ----------
        checkpoint_path : str | Path
            Path to ``best.pt`` or ``last.pt``.
        model_path : str | Path
            Path to the MediaPipe .task model file.
        fps : float
            Frame rate used during training (default 30).
        device : str | torch.device
            ``"cpu"``, ``"cuda"``, or ``"mps"``.
        buffer_size : int
            Rolling window length in frames.
        threshold : float
            Sigmoid confidence threshold.
        """
        device = torch.device(device)
        checkpoint_path = Path(checkpoint_path)

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

        # Trainer saves: {"model_state_dict": ..., "optimizer_state_dict": ..., ...}
        # Also support: {"model_state": ..., "model_cfg": ...} and bare state_dicts
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model = CausalTCN()
            model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict) and "model_cfg" in ckpt:
            cfg = ckpt["model_cfg"]
            model = CausalTCN(
                in_features=cfg.get("in_features", 16),
                channels=cfg.get("channels", [64, 64, 64]),
                kernel_size=cfg.get("kernel_size", 3),
                dilations=cfg.get("dilations", [1, 2, 4]),
                dropout=cfg.get("dropout", 0.1),
                n_labels=cfg.get("n_labels", 2),
            )
            model.load_state_dict(ckpt.get("model_state", ckpt["model_cfg"]))
        else:
            # Bare state dict
            model = CausalTCN()
            model.load_state_dict(ckpt)

        model.to(device).eval()

        return cls(
            model=model,
            pose_extractor=PoseExtractor(model_path=model_path),
            feature_extractor=OHPFeatureExtractor(fps=fps),
            device=device,
            buffer_size=buffer_size,
            threshold=threshold,
        )


class SquatPredictor(BasePredictor):
    """Per-frame Squat form error predictor."""

    labels = ["knees_forward", "knees_inward"]

    # ── Factory ──────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_path: str | Path = "artifacts/pose_landmarker_heavy.task",
        fps: float = 30.0,
        device: str | torch.device = "cpu",
        buffer_size: int = _DEFAULT_RF,
        threshold: float = 0.5,
    ) -> "SquatPredictor":
        """Build a SquatPredictor by loading a saved checkpoint.

        Parameters
        ----------
        checkpoint_path : str | Path
            Path to ``best.pt`` or ``last.pt``.
        model_path : str | Path
            Path to the MediaPipe .task model file.
        fps : float
            Frame rate used during training (default 30).
        device : str | torch.device
            ``"cpu"``, ``"cuda"``, or ``"mps"``.
        buffer_size : int
            Rolling window length in frames.
        threshold : float
            Sigmoid confidence threshold.
        """
        device = torch.device(device)
        checkpoint_path = Path(checkpoint_path)

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model = CausalTCN(in_features=20, n_labels=2)
            model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict) and "model_cfg" in ckpt:
            cfg = ckpt["model_cfg"]
            model = CausalTCN(
                in_features=cfg.get("in_features", 20),
                channels=cfg.get("channels", [64, 64, 64]),
                kernel_size=cfg.get("kernel_size", 3),
                dilations=cfg.get("dilations", [1, 2, 4]),
                dropout=cfg.get("dropout", 0.1),
                n_labels=cfg.get("n_labels", 2),
            )
            model.load_state_dict(ckpt.get("model_state", ckpt["model_cfg"]))
        else:
            model = CausalTCN(in_features=20, n_labels=2)
            model.load_state_dict(ckpt)

        model.to(device).eval()

        return cls(
            model=model,
            pose_extractor=PoseExtractor(model_path=model_path),
            feature_extractor=SquatFeatureExtractor(fps=fps),
            device=device,
            buffer_size=buffer_size,
            threshold=threshold,
        )
