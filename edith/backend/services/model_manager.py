"""
ModelManager
============
Drives OHPPredictor / SquatPredictor from gym-form using the same registry
as edith.py.  Exercise IDs are registry keys: "ohp", "squat", etc.

Each launch() call:
  1. Resolves the predictor class + checkpoint from gym-form's model_registry.
  2. Loads the predictor (blocking, done in a thread-pool executor).
  3. Starts a background daemon thread running the camera loop.
  4. An asyncio drain task forwards frames + real stats to the WebSocket.
"""

import asyncio
import base64
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Awaitable, Optional

import cv2
import numpy as np

# ── Resolve gym-form on the Python path ───────────────────────────────────────
_GYM_FORM = Path(__file__).resolve().parents[3] / "gym-form"
if str(_GYM_FORM) not in sys.path:
    sys.path.insert(0, str(_GYM_FORM))

from src.control.model_registry import resolve, known_exercises  # noqa: E402
from src.utils.openCv import draw_skeleton, _draw_overlay         # noqa: E402


class ModelManager:
    def __init__(self) -> None:
        self.active_exercise: Optional[str] = None   # registry key e.g. "ohp"
        self._stop_event  = threading.Event()
        self._thread:  Optional[threading.Thread]           = None
        self._drain:   Optional[asyncio.Task]               = None
        self._loop:    Optional[asyncio.AbstractEventLoop]  = None
        self._q:       Optional[asyncio.Queue]              = None

    # ── Public API ────────────────────────────────────────────────────────────

    async def launch(
        self,
        exercise_key: str,
        on_log:   Callable[[str, str], Awaitable[None]],
        on_stats: Callable[[dict],     Awaitable[None]],
        on_state: Callable[[str, Optional[str]], Awaitable[None]],
        on_frame: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> None:
        """Boot a model and start streaming annotated frames + stats."""
        if self.active_exercise == exercise_key:
            entry = resolve(exercise_key)
            name  = entry.display_name if entry else exercise_key
            await on_log(f"We're already doing {name}.", "warn")
            return

        if self.active_exercise:
            await self._stop_inference()

        entry = resolve(exercise_key)
        if entry is None:
            available = ", ".join(known_exercises())
            await on_log(
                f"Unknown exercise '{exercise_key}'. Available: {available}", "error"
            )
            return

        self._loop = asyncio.get_event_loop()
        self._q    = asyncio.Queue()

        await on_state("booting", exercise_key)
        await on_log(f"Initiating boot sequence — {entry.display_name}", "info")
        await on_log("Loading neural weights...", "info")

        # Resolve absolute checkpoint path
        ckpt = _GYM_FORM / entry.checkpoint
        model_path = _GYM_FORM / "artifacts" / "pose_landmarker_heavy.task"

        try:
            predictor = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: entry.predictor_cls.from_checkpoint(
                    checkpoint_path=ckpt,
                    model_path=str(model_path),
                ),
            )
        except Exception as exc:
            await on_log(f"Failed to load model: {exc}", "error")
            await on_state("idle", None)
            return

        await on_log("Calibrating pose skeleton...", "info")
        await on_log("Connecting inference pipeline...", "info")
        await on_log("Running self-diagnostics...", "info")

        self.active_exercise = exercise_key
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._camera_thread,
            args=(predictor,),
            daemon=True,
        )
        self._thread.start()

        self._drain = asyncio.create_task(
            self._drain_loop(on_stats, on_frame)
        )

        await on_state("active", exercise_key)
        await on_log(f"{entry.display_name} ONLINE — ready for inference.", "success")

    async def shutdown(
        self,
        on_log:   Callable[[str, str], Awaitable[None]],
        on_state: Callable[[str, Optional[str]], Awaitable[None]],
    ) -> None:
        await on_log("Shutdown command received. Terminating inference...", "warn")
        await on_state("shutdown", None)
        await self._stop_inference()
        await on_state("idle", None)
        await on_log("All models offline. System in standby.", "info")

    async def switch(
        self,
        next_key: Optional[str],
        on_log:   Callable[[str, str], Awaitable[None]],
        on_stats: Callable[[dict],     Awaitable[None]],
        on_state: Callable[[str, Optional[str]], Awaitable[None]],
        on_frame: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> None:
        """Switch to next_key, or toggle between exercises if next_key is None."""
        if next_key is None:
            # Toggle: pick the first exercise that isn't active
            all_keys = known_exercises()
            other = next((k for k in all_keys if k != self.active_exercise), None)
            if other is None:
                await on_log("Only one exercise registered — nothing to switch to.", "warn")
                return
            next_key = other

        entry = resolve(next_key)
        name  = entry.display_name if entry else next_key
        await on_log(f"Switching to {name}...", "warn")
        await self.launch(next_key, on_log, on_stats, on_state, on_frame)

    def known(self) -> list[str]:
        return known_exercises()

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _stop_inference(self) -> None:
        self._stop_event.set()

        if self._drain:
            self._drain.cancel()
            try:
                await self._drain
            except asyncio.CancelledError:
                pass
            self._drain = None

        if self._thread:
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._thread.join(timeout=5)
            )
            self._thread = None

        self.active_exercise = None

    def _camera_thread(self, predictor) -> None:
        """Daemon thread: capture → infer → annotate → enqueue (b64 frame + stats)."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self._enqueue(("log", ("Camera not available.", "error")))
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        with predictor:
            while not self._stop_event.is_set():
                t0 = time.perf_counter()

                ret, frame = cap.read()
                if not ret:
                    continue

                probs = predictor.predict(frame)
                if predictor._lm_buffer:
                    draw_skeleton(frame, predictor._lm_buffer[-1], vis_thr=0.3)
                _draw_overlay(
                    frame, probs, predictor.is_warm,
                    predictor.threshold, predictor.labels,
                )

                elapsed    = time.perf_counter() - t0
                latency_ms = int(elapsed * 1000)
                fps        = min(int(1.0 / elapsed), 60) if elapsed > 0 else 30
                confidence = (
                    int(float(np.max(probs)) * 100)
                    if probs is not None else 0
                )

                ok, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                )
                if ok:
                    b64 = base64.b64encode(buf).decode("utf-8")
                    self._enqueue(("frame", b64))

                self._enqueue(("stats", {
                    "confidence": confidence,
                    "fps":        fps,
                    "latency":    latency_ms,
                }))

        cap.release()

    def _enqueue(self, item: tuple) -> None:
        if self._loop is None or self._q is None:
            return
        kind = item[0]
        if kind == "frame" and self._q.qsize() >= 2:
            return
        self._loop.call_soon_threadsafe(self._q.put_nowait, item)

    async def _drain_loop(
        self,
        on_stats: Callable[[dict], Awaitable[None]],
        on_frame: Optional[Callable[[str], Awaitable[None]]],
    ) -> None:
        while True:
            kind, payload = await self._q.get()
            if kind == "frame" and on_frame:
                await on_frame(payload)
            elif kind == "stats":
                await on_stats(payload)


import asyncio
import base64
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Awaitable, Optional

import cv2
import numpy as np

# ── Resolve gym-form on the Python path ───────────────────────────────────────
_GYM_FORM = Path(__file__).resolve().parents[3] / "gym-form"
if str(_GYM_FORM) not in sys.path:
    sys.path.insert(0, str(_GYM_FORM))

from src.realtime.predictor import OHPPredictor, SquatPredictor  # noqa: E402
from src.utils.openCv import draw_skeleton, _draw_overlay          # noqa: E402

# ── Model registry ────────────────────────────────────────────────────────────
_ARTIFACTS   = _GYM_FORM / "artifacts" / "pose_landmarker_heavy.task"
_CHECKPOINTS = _GYM_FORM / "checkpoints"

_REGISTRY = {
    "alpha": {
        "cls":   SquatPredictor,
        "ckpt":  _CHECKPOINTS / "squat" / "best.pt",
        "label": "Squat Analyzer",
    },
    "beta": {
        "cls":   OHPPredictor,
        "ckpt":  _CHECKPOINTS / "ohp" / "best.pt",
        "label": "OHP Analyzer",
    },
}


class ModelManager:
    def __init__(self) -> None:
        self.active_model: Optional[str] = None
        self._stop_event  = threading.Event()
        self._thread:  Optional[threading.Thread]              = None
        self._drain:   Optional[asyncio.Task]                  = None
        self._loop:    Optional[asyncio.AbstractEventLoop]     = None
        self._q:       Optional[asyncio.Queue]                 = None

    # ── Public API ────────────────────────────────────────────────────────────

    async def launch(
        self,
        model_id: str,
        on_log:   Callable[[str, str], Awaitable[None]],
        on_stats: Callable[[dict],     Awaitable[None]],
        on_state: Callable[[str, Optional[str]], Awaitable[None]],
        on_frame: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> None:
        """Boot a model and start streaming annotated frames + stats."""
        if self.active_model == model_id:
            await on_log(f"Model {model_id.upper()} is already active.", "warn")
            return

        if self.active_model:
            await self._stop_inference(on_log, on_state)

        self._loop = asyncio.get_event_loop()
        self._q    = asyncio.Queue(maxsize=0)  # unbounded; we drop in the thread

        await on_state("booting", model_id)
        entry = _REGISTRY[model_id]
        await on_log(f"Initiating boot sequence — Model {model_id.upper()}", "info")
        await on_log("Loading neural weights...", "info")

        try:
            predictor = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: entry["cls"].from_checkpoint(
                    checkpoint_path=entry["ckpt"],
                    model_path=str(_ARTIFACTS),
                ),
            )
        except Exception as exc:
            await on_log(f"Failed to load model: {exc}", "error")
            await on_state("idle", None)
            return

        await on_log("Calibrating pose skeleton...", "info")
        await on_log("Connecting inference pipeline...", "info")
        await on_log("Running self-diagnostics...", "info")

        self.active_model = model_id
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._camera_thread,
            args=(predictor,),
            daemon=True,
        )
        self._thread.start()

        self._drain = asyncio.create_task(
            self._drain_loop(on_stats, on_frame)
        )

        await on_state("active", model_id)
        await on_log(f"{entry['label']} ONLINE — ready for inference.", "success")

    async def shutdown(
        self,
        on_log:   Callable[[str, str], Awaitable[None]],
        on_state: Callable[[str, Optional[str]], Awaitable[None]],
    ) -> None:
        await on_log("Shutdown command received. Terminating inference...", "warn")
        await on_state("shutdown", None)
        await self._stop_inference(on_log, on_state)
        await on_state("idle", None)
        await on_log("All models offline. System in standby.", "info")

    async def switch(
        self,
        on_log:   Callable[[str, str], Awaitable[None]],
        on_stats: Callable[[dict],     Awaitable[None]],
        on_state: Callable[[str, Optional[str]], Awaitable[None]],
        on_frame: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> None:
        next_model = "beta" if self.active_model == "alpha" else "alpha"
        await on_log(f"Switching to Model {next_model.upper()}...", "warn")
        await self.shutdown(on_log, on_state)
        await self.launch(next_model, on_log, on_stats, on_state, on_frame)

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _stop_inference(self, on_log, on_state) -> None:
        self._stop_event.set()

        if self._drain:
            self._drain.cancel()
            try:
                await self._drain
            except asyncio.CancelledError:
                pass
            self._drain = None

        if self._thread:
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._thread.join(timeout=5)
            )
            self._thread = None

        self.active_model = None

    def _camera_thread(self, predictor) -> None:
        """Daemon thread: capture → infer → annotate → enqueue (b64 frame + stats)."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self._enqueue(("log", ("Camera not available.", "error")))
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        with predictor:
            while not self._stop_event.is_set():
                t0 = time.perf_counter()

                ret, frame = cap.read()
                if not ret:
                    continue

                probs = predictor.predict(frame)
                if predictor._lm_buffer:
                    draw_skeleton(frame, predictor._lm_buffer[-1], vis_thr=0.3)
                _draw_overlay(
                    frame, probs, predictor.is_warm,
                    predictor.threshold, predictor.labels,
                )

                elapsed    = time.perf_counter() - t0
                latency_ms = int(elapsed * 1000)
                fps        = min(int(1.0 / elapsed), 60) if elapsed > 0 else 30
                confidence = (
                    int(float(np.max(probs)) * 100)
                    if probs is not None else 0
                )

                # Encode frame as JPEG and base64 it
                ok, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                )
                if ok:
                    b64 = base64.b64encode(buf).decode("utf-8")
                    self._enqueue(("frame", b64))

                self._enqueue(("stats", {
                    "confidence": confidence,
                    "fps":        fps,
                    "latency":    latency_ms,
                }))

        cap.release()

    def _enqueue(self, item: tuple) -> None:
        """Thread-safe put into the asyncio queue via the event loop."""
        if self._loop is None or self._q is None:
            return
        kind = item[0]
        # Drop frames when the drain task is falling behind
        if kind == "frame" and self._q.qsize() >= 2:
            return
        self._loop.call_soon_threadsafe(self._q.put_nowait, item)

    async def _drain_loop(
        self,
        on_stats: Callable[[dict], Awaitable[None]],
        on_frame: Optional[Callable[[str], Awaitable[None]]],
    ) -> None:
        """Async task: forward queued data to WebSocket callbacks."""
        while True:
            kind, payload = await self._q.get()
            if kind == "frame" and on_frame:
                await on_frame(payload)
            elif kind == "stats":
                await on_stats(payload)
            # "log" items from the camera thread are silently dropped here;
            # errors during launch() are surfaced directly via on_log.
