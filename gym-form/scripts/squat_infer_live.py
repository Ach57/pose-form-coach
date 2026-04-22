from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python scripts/infer_live.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.realtime.predictor import SquatPredictor
from src.utils.openCv import build_parser, run as opencv_run

# ── Main loop ──────────────────────────────────────────────────────────────────

def run(args) -> None:
    print(f"Loading checkpoint: {args.checkpoint}")
    predictor = SquatPredictor.from_checkpoint(
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
        window_title="Squat Form Detector",
    )

# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(build_parser("Live Squat form error detection").parse_args())
