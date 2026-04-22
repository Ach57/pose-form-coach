"""Entry point for Edith — the voice-controlled gym form coach.

Usage
-----
    # SpeechRecognition fallback (no API key, needs internet)
    python scripts/start_edith.py

    # Pvporcupine wake word (offline, low CPU)
    python scripts/start_edith.py --access-key YOUR_PICOVOICE_KEY
    python scripts/start_edith.py --access-key YOUR_KEY --keyword-path hey-edith_mac.ppn

    # Custom pipeline options
    python scripts/start_edith.py --device mps --camera 1 --threshold 0.4

Interaction examples
--------------------
    Say "Edith"          →  Edith: "Yes?"
    Say "start overhead press"  →  Edith starts the OHP pipeline
    Say "Edith"          →  Edith: "Yes?"
    Say "switch to squat"       →  Edith stops OHP, starts Squat
    Say "Edith"          →  Edith: "Yes?"
    Say "stop"                  →  Edith stops the active pipeline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.control.edith import Edith


def main() -> None:
    p = argparse.ArgumentParser(description="Start Edith — voice-controlled gym form coach")

    # Wake word options
    p.add_argument(
        "--wake-phrase",
        default="edith",
        help="Wake phrase to listen for (default: edith).",
    )

    # Pipeline options
    p.add_argument(
        "--model-path",
        default="artifacts/pose_landmarker_heavy.task",
        help="MediaPipe .task file (default: pose_landmarker_heavy.task)",
    )
    p.add_argument("--camera",    type=int,   default=0,    help="Camera index (default: 0)")
    p.add_argument("--device",    default="cpu",             help="torch device: cpu|cuda|mps (default: cpu)")
    p.add_argument("--fps",       type=float, default=30.0, help="Training FPS (default: 30)")
    p.add_argument("--threshold", type=float, default=0.5,  help="Sigmoid threshold (default: 0.5)")
    p.add_argument("--width",     type=int,   default=1280, help="Capture width (default: 1280)")
    p.add_argument("--height",    type=int,   default=720,  help="Capture height (default: 720)")

    args = p.parse_args()

    edith = Edith(
        wake_phrase=args.wake_phrase,
        model_path=args.model_path,
        camera=args.camera,
        device=args.device,
        fps=args.fps,
        threshold=args.threshold,
        width=args.width,
        height=args.height,
    )
    edith.run()


if __name__ == "__main__":
    main()
