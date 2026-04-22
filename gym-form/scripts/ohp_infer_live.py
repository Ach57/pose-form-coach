"""
Live OHP form error detection from webcam.

Usage
-----
    python scripts/infer_live.py --checkpoint checkpoints/best.pt

Optional flags
--------------
    --model-path   Path to MediaPipe .task file (default: artifacts/pose_landmarker_heavy.task)
    --camera       Camera index (default: 0)
    --fps          Frame rate assumed during training (default: 30)
    --threshold    Sigmoid confidence threshold (default: 0.5)
    --device       torch device: cpu | cuda | mps (default: cpu)
    --width        Capture width in pixels (default: 1280)
    --height       Capture height in pixels (default: 720)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/infer_live.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.realtime.predictor import OHPPredictor
from src.utils.openCv import run as opencv_run


# ── Main loop ──────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    print(f"Loading checkpoint: {args.checkpoint}")
    predictor = OHPPredictor.from_checkpoint(
        checkpoint_path=args.checkpoint,
        model_path=args.model_path,
        fps=args.fps,
        device=args.device,
        threshold=args.threshold,
    )
    opencv_run(
        predictor,
        camera=args.camera,
        threshold=args.threshold,
        width=args.width,
        height=args.height,
        window_title="OHP Form Detector",
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live OHP form error detection")
    p.add_argument("--checkpoint",
                   required=True,
                   help="Path to best.pt or last.pt")
    p.add_argument("--model-path",
                   default="artifacts/pose_landmarker_heavy.task",
                   help="MediaPipe .task file (default: pose_landmarker_heavy.task)")
    p.add_argument("--camera",  type=int,   default=0,    help="Camera index (default: 0)")
    p.add_argument("--fps",     type=float, default=30.0, help="Training FPS (default: 30)")
    p.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold (default: 0.5)")
    p.add_argument("--device",  default="cpu",  help="torch device: cpu|cuda|mps (default: cpu)")
    p.add_argument("--width",   type=int,   default=1280, help="Capture width (default: 1280)")
    p.add_argument("--height",  type=int,   default=720,  help="Capture height (default: 720)")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
