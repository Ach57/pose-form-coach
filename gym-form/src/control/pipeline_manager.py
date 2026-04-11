"""Manages the lifecycle of the active form-detection pipeline.

The inference loop (camera + model) runs in a background daemon thread so
Edith's voice loop stays fully responsive while analysis is running.

Usage
-----
    pm = PipelineManager(model_path="artifacts/pose_landmarker_heavy.task")
    pm.start("ohp")       # starts camera loop in background thread
    pm.stop()             # gracefully shuts it down
    pm.start("squat")     # switch exercise — stop is called automatically
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import cv2
import numpy as np

from src.control.model_registry import resolve
from src.utils.openCv import capture_and_infer


class PipelineManager:
    """Start, stop, and switch the live inference pipeline.

    Parameters
    ----------
    model_path : str
        Path to the MediaPipe .task file.
    camera : int
        Camera index (default 0).
    device : str
        Torch device string: ``"cpu"``, ``"cuda"``, or ``"mps"``.
    fps : float
        Frame rate assumed during model training.
    threshold : float
        Sigmoid confidence threshold for error display.
    width, height : int
        Capture resolution.
    """

    def __init__(
        self,
        model_path: str = "artifacts/pose_landmarker_heavy.task",
        camera: int = 0,
        device: str = "cpu",
        fps: float = 30.0,
        threshold: float = 0.5,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self.model_path = model_path
        self.camera = camera
        self.device = device
        self.fps = fps
        self.threshold = threshold
        self.width = width
        self.height = height

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._current_exercise: str | None = None
        self._frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self._window_title: str = ""

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def current_exercise(self) -> str | None:
        return self._current_exercise

    @property
    def window_title(self) -> str:
        return self._window_title

    def get_frame(self) -> np.ndarray | None:
        """Return the latest annotated frame, or *None* if none is ready."""
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None

    def start(self, exercise: str) -> bool:
        """Start inference for *exercise*.

        If a pipeline is already running it is stopped first (i.e. this method
        handles switching automatically).

        Returns
        -------
        bool
            *True* on success, *False* if the exercise name is not recognised.

        Raises
        ------
        FileNotFoundError
            When the checkpoint for the exercise does not exist on disk.
        """
        entry = resolve(exercise)
        if entry is None:
            return False

        checkpoint = entry.checkpoint
        if not Path(checkpoint).exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}\n"
                "Train the model first or point the registry to the correct path."
            )

        if self.is_running:
            self.stop()

        predictor = entry.predictor_cls.from_checkpoint(
            checkpoint_path=checkpoint,
            model_path=self.model_path,
            fps=self.fps,
            device=self.device,
            threshold=self.threshold,
        )

        # Drain stale frames from a previous session
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

        self._stop_event.clear()
        self._current_exercise = exercise.lower()
        self._window_title = f"{entry.display_name} — Form Detector"
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(predictor, entry.display_name),
            daemon=True,
            name=f"pipeline-{exercise}",
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        """Gracefully stop the active pipeline and wait for the thread to exit."""
        if not self.is_running:
            return
        self._stop_event.set()
        cv2.destroyAllWindows()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._current_exercise = None

    # ── Internal ───────────────────────────────────────────────────────────────

    def _run_loop(self, predictor, display_name: str) -> None:
        capture_and_infer(
            predictor,
            frame_queue=self._frame_queue,
            camera=self.camera,
            threshold=self.threshold,
            width=self.width,
            height=self.height,
            stop_event=self._stop_event,
        )
    