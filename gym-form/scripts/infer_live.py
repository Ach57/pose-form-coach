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

import cv2
import numpy as np

# Allow running as `python scripts/infer_live.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.realtime.predictor import LABELS, OHPPredictor
from src.utils.mediapipe_visualization import BONES, BONE_COLORS


# ── Drawing helpers ────────────────────────────────────────────────────────────

# BGR colours
_GREEN  = (50, 220, 50)
_RED    = (30, 30, 230)
_YELLOW = (30, 220, 230)
_WHITE  = (255, 255, 255)
_GREY   = (160, 160, 160)
_BLACK  = (0, 0, 0)

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_BAR_W      = 220   # width of probability bar
_BAR_H      = 22
_PADDING    = 12
_ROW_H      = 36


def _draw_overlay(
    frame: np.ndarray,
    probs: np.ndarray | None,
    is_warm: bool,
    threshold: float,
) -> np.ndarray:
    """Draw status panel on top-left corner of `frame` (in-place)."""

    h, w = frame.shape[:2]
    panel_w = _BAR_W + 200
    panel_h = _PADDING * 2 + _ROW_H * (len(LABELS) + 1)

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), _BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = _PADDING

    # ── Header ──
    if not is_warm:
        status_text = "Warming up..."
        status_color = _YELLOW
    elif probs is None:
        status_text = "No pose detected"
        status_color = _GREY
    else:
        errors = [lbl.split("_")[1].upper() for lbl, p in zip(LABELS, probs) if p >= threshold]
        if errors:
            status_text = "ERROR: " + " + ".join(errors)
            status_color = _RED
        else:
            status_text = "Good form"
            status_color = _GREEN

    cv2.putText(frame, status_text, (_PADDING, y + 18), _FONT, 0.65, status_color, 2, cv2.LINE_AA)
    y += _ROW_H

    # ── Probability bars ──
    if probs is not None:
        for label, p in zip(LABELS, probs):
            short = label.split("_")[1]   # "elbow" / "knee"
            bar_fill = int(p * _BAR_W)
            bar_color = _RED if p >= threshold else _GREEN

            # Label text
            cv2.putText(frame, f"{short}:", (_PADDING, y + 15), _FONT, 0.5, _WHITE, 1, cv2.LINE_AA)
            # Background bar
            cv2.rectangle(frame, (80, y), (80 + _BAR_W, y + _BAR_H), _GREY, -1)
            # Filled portion
            if bar_fill > 0:
                cv2.rectangle(frame, (80, y), (80 + bar_fill, y + _BAR_H), bar_color, -1)
            # Threshold line
            thr_x = 80 + int(threshold * _BAR_W)
            cv2.line(frame, (thr_x, y), (thr_x, y + _BAR_H), _WHITE, 1)
            # Probability value
            cv2.putText(frame, f"{p:.2f}", (80 + _BAR_W + 6, y + 15), _FONT, 0.45, _WHITE, 1, cv2.LINE_AA)

            y += _ROW_H

    return frame


def draw_skeleton(frame: np.ndarray, lm: np.ndarray, vis_thr: float = 0.3) -> None:
    """Draw pose skeleton on `frame` from normalized MediaPipe landmarks `lm`.

    `lm` is expected in MediaPipe format (33, 4): x,y in [0,1], z, visibility.
    This function converts normalized coords to pixel coords and draws bones
    and joints. It skips landmarks with visibility < `vis_thr`.
    """
    if lm is None or lm.size == 0:
        return

    h, w = frame.shape[:2]

    # Bones
    for region, pairs in BONES.items():
        col_hex = BONE_COLORS.get(region, "#FFFFFF").lstrip("#")
        # hex -> BGR tuple
        r = int(col_hex[0:2], 16)
        g = int(col_hex[2:4], 16)
        b = int(col_hex[4:6], 16)
        color = (b, g, r)
        for i, j in pairs:
            vi = float(lm[i, 3])
            vj = float(lm[j, 3])
            if vi > vis_thr and vj > vis_thr:
                p1 = (int(lm[i, 0] * w), int(lm[i, 1] * h))
                p2 = (int(lm[j, 0] * w), int(lm[j, 1] * h))
                cv2.line(frame, p1, p2, color, 2, lineType=cv2.LINE_AA)

    # Joints
    for i in range(lm.shape[0]):
        v = float(lm[i, 3])
        if v > vis_thr:
            pt = (int(lm[i, 0] * w), int(lm[i, 1] * h))
            cv2.circle(frame, pt, 3, (255, 200, 0), -1, lineType=cv2.LINE_AA)


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

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {args.camera}", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("Press  q  or  Esc  to quit.")

    with predictor:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Empty frame, retrying...")
                continue

            probs = predictor.predict(frame)
            # Draw last detected landmarks onto the frame (if available)
            if predictor._lm_buffer:
                last_lm = predictor._lm_buffer[-1]
                draw_skeleton(frame, last_lm, vis_thr=0.3)

            _draw_overlay(frame, probs, predictor.is_warm, args.threshold)

            cv2.imshow("OHP Form Detector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):   # q or Esc
                break

    cap.release()
    cv2.destroyAllWindows()


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
