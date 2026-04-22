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

import sys
from pathlib import Path

# Allow running as `python scripts/infer_live.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.realtime.predictor import OHPPredictor
from src.utils.openCv import build_parser, run as opencv_run


# ── Main loop ──────────────────────────────────────────────────────────────────

def run(args) -> None:
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

if __name__ == "__main__":
    run(build_parser("Live OHP form error detection").parse_args())
