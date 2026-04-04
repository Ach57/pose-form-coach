"""
OHPPredictor — live inference for the Overhead Press error detector.

Wraps PoseExtractor, OHPFeatureExtractor, and CausalTCN into a single object
that takes one BGR webcam frame and returns per-label probabilities.

Usage
-----
    predictor = OHPPredictor.from_checkpoint(
        checkpoint_path="checkpoints/best.pt",
        model_path="artifacts/pose_landmarker_heavy.task",
        device="cpu",
    )
    predictor.open()
    probs = predictor.predict(frame)   # np.ndarray (2,) or None
    predictor.close()

Label order
-----------
    probs[0]  →  ohp_elbow error probability
    probs[1]  →  ohp_knee  error probability
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from src.extract.pose_extractor import PoseExtractor
from src.features.feature_extractor import OHPFeatureExtractor
from src.models.causal_tcn import CausalTCN

# Labels must match training order
LABELS = ["ohp_elbow", "ohp_knee"]

# Receptive field of the default model config (1 + 2*(1+2+4)*2 = 29 frames)
_DEFAULT_RF = 29


class OHPPredictor:
    """Per-frame OHP form error predictor.

    Parameters
    ----------
    model : CausalTCN
        Loaded, eval-mode model.
    pose_extractor : PoseExtractor
        Configured PoseLandmarker wrapper.
    feature_extractor : OHPFeatureExtractor
        Feature extractor matching the training configuration.
    device : torch.device
        Device to run model inference on.
    buffer_size : int
        Number of frames to keep in the rolling window (should equal model
        receptive field). Predictions are None until the buffer is full.
    threshold : float
        Sigmoid threshold above which a label is considered active.
    """

    def __init__(
        self,
        model: CausalTCN,
        pose_extractor: PoseExtractor,
        feature_extractor: OHPFeatureExtractor,
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

        The checkpoint must be a dict saved by Trainer with keys:
          ``model_state`` and ``model_cfg`` (or bare state_dict).

        Parameters
        ----------
        checkpoint_path : str | Path
            Path to ``best.pt`` or ``last.pt``.
        model_path : str | Path
            Path to the MediaPipe .task model file.
        fps : float
            Frame rate used during training (default 30). Used by feature extractor.
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

        pose_extractor = PoseExtractor(model_path=model_path)
        feature_extractor = OHPFeatureExtractor(fps=fps)

        return cls(
            model=model,
            pose_extractor=pose_extractor,
            feature_extractor=feature_extractor,
            device=device,
            buffer_size=buffer_size,
            threshold=threshold,
        )

    # ── Session management ────────────────────────────────────────────

    def open(self) -> None:
        """Open MediaPipe session. Call before the camera loop."""
        self.pose_extractor.open()
        self._lm_buffer.clear()

    def close(self) -> None:
        """Close MediaPipe session. Call after the camera loop."""
        self.pose_extractor.close()

    def __enter__(self) -> "OHPPredictor":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Per-frame inference ───────────────────────────────────────────

    def predict(self, frame: np.ndarray) -> np.ndarray | None:
        """Run inference on one BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from ``cv2.VideoCapture``, shape (H, W, 3).

        Returns
        -------
        probs : np.ndarray shape (2,) or None
            Sigmoid probabilities [p_elbow_error, p_knee_error].
            Returns ``None`` until the rolling buffer is full (warm-up phase).
        """
        lm = self.pose_extractor.process_frame(frame)
        if lm is None:
            # Detection failed — reuse last known landmarks if buffer non-empty
            lm = self._lm_buffer[-1].copy() if self._lm_buffer else np.zeros((33, 4), dtype=np.float32)

        self._lm_buffer.append(lm)

        if len(self._lm_buffer) < self._buffer_size:
            return None     # still warming up

        # Stack buffer → (T, 33, 4), extract features → (T, 16)
        landmarks_window = np.stack(list(self._lm_buffer), axis=0)
        features = self.feature_extractor.extract(landmarks_window)   # (T, 16)

        # Build tensor (1, T, 16)
        x = torch.from_numpy(features).unsqueeze(0).to(self.device)   # (1, T, 16)

        with torch.no_grad():
            logits = self.model(x)              # (1, T, 2)
            probs = torch.sigmoid(logits)       # (1, T, 2)

        # Return probabilities for the most recent (last) frame
        return probs[0, -1].cpu().numpy()       # (2,)

    # ── Convenience helpers ───────────────────────────────────────────

    def label_errors(self, probs: np.ndarray) -> list[str]:
        """Return list of active error labels given a probability vector."""
        return [label for label, p in zip(LABELS, probs) if p >= self.threshold]

    @property
    def is_warm(self) -> bool:
        """True once the rolling buffer has filled (warm-up complete)."""
        return len(self._lm_buffer) >= self._buffer_size
