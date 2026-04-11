from __future__ import annotations

import argparse
import queue
import sys
import threading

import cv2
import numpy as np

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
    labels: list[str]
) -> np.ndarray:
    """Draw status panel on top-left corner of `frame` (in-place)."""

    h, w = frame.shape[:2]
    panel_w = _BAR_W + 200
    panel_h = _PADDING * 2 + _ROW_H * (len(labels) + 1)

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
        errors = [lbl.split("_")[1].upper() for lbl, p in zip(labels, probs) if p >= threshold]
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
        for label, p in zip(labels, probs):
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

def run(
    predictor,
    camera: int = 0,
    threshold: float = 0.5,
    width: int = 1280,
    height: int = 720,
    window_title: str = "Form Detector",
    stop_event: threading.Event | None = None,
) -> None:
    """Generic camera loop that works with any BasePredictor subclass.

    Parameters
    ----------
    predictor : BasePredictor
        An already-constructed (but not yet opened) predictor instance.
    camera : int
        Camera index passed to ``cv2.VideoCapture``.
    threshold : float
        Sigmoid confidence threshold forwarded to ``_draw_overlay``.
    width, height : int
        Capture resolution.
    window_title : str
        Title of the ``cv2.imshow`` window.
    stop_event : threading.Event | None
        When set, the loop exits cleanly.  Used by PipelineManager to stop
        the background thread from Edith's voice loop.
    """
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {camera}", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print("Press  q  or  Esc  to quit.")

    with predictor:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Empty frame, retrying...")
                continue

            probs = predictor.predict(frame)
            if predictor._lm_buffer:
                draw_skeleton(frame, predictor._lm_buffer[-1], vis_thr=0.3)

            _draw_overlay(frame, probs, predictor.is_warm, threshold, predictor.labels)

            cv2.imshow(window_title, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):   # q or Esc
                break
            if stop_event is not None and stop_event.is_set():
                break

    cap.release()
    cv2.destroyAllWindows()


def capture_and_infer(
    predictor,
    frame_queue: "queue.Queue[np.ndarray]",
    camera: int = 0,
    threshold: float = 0.5,
    width: int = 1280,
    height: int = 720,
    stop_event: threading.Event | None = None,
) -> None:
    """Background-thread version of the camera loop.

    Unlike ``run()``, this function never calls ``cv2.imshow``.
    Instead it pushes every annotated frame into *frame_queue* so the
    **main thread** can display it.  This is required on macOS where
    AppKit forbids GUI calls off the main thread.

    Used by :class:`~src.control.pipeline_manager.PipelineManager`.
    """
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {camera}", file=sys.stderr)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    with predictor:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            probs = predictor.predict(frame)
            if predictor._lm_buffer:
                draw_skeleton(frame, predictor._lm_buffer[-1], vis_thr=0.3)
            _draw_overlay(frame, probs, predictor.is_warm, threshold, predictor.labels)

            # Drop oldest frame if the main thread is falling behind
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                frame_queue.put_nowait(frame)

            if stop_event is not None and stop_event.is_set():
                break

    cap.release()


# ── Shared CLI parser ──────────────────────────────────────────────────────────

def build_parser(description: str = "Live form error detection") -> argparse.ArgumentParser:
    """Return an ArgumentParser with all standard inference flags.

    Scripts call ``build_parser("Live OHP …").parse_args()`` to get a consistent
    CLI across exercises without duplicating argument definitions.
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--checkpoint",
                   required=True,
                   help="Path to best.pt or last.pt")
    p.add_argument("--model-path",
                   default="artifacts/pose_landmarker_heavy.task",
                   help="MediaPipe .task file (default: pose_landmarker_heavy.task)")
    p.add_argument("--camera",    type=int,   default=0,    help="Camera index (default: 0)")
    p.add_argument("--fps",       type=float, default=30.0, help="Training FPS (default: 30)")
    p.add_argument("--threshold", type=float, default=0.5,  help="Sigmoid threshold (default: 0.5)")
    p.add_argument("--device",    default="cpu",             help="torch device: cpu|cuda|mps (default: cpu)")
    p.add_argument("--width",     type=int,   default=1280, help="Capture width (default: 1280)")
    p.add_argument("--height",    type=int,   default=720,  help="Capture height (default: 720)")
    return p