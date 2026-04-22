"""Edith — voice-controlled gym form coach.

Flow
----
1.  WakeWordDetector listens in the background (always on).
2.  When the wake word fires, Edith speaks "Yes?" and listens for one command.
3.  The command is parsed for intent (start / stop / switch / status).
4.  PipelineManager starts/stops/switches the live inference model.
5.  Edith speaks a confirmation and returns to listening.

Usage
-----
    # SR fallback (no API key needed, requires internet)
    edith = Edith()
    edith.run()

    # Pvporcupine mode (offline, low CPU)
    edith = Edith(access_key="YOUR_PICOVOICE_KEY", keyword_path="hey-edith.ppn")
    edith.run()

    # With custom pipeline options
    edith = Edith(device="mps", model_path="artifacts/pose_landmarker_heavy.task")
    edith.run()
"""

from __future__ import annotations

import queue
import re
import sys
import threading
import time

from src.control.model_registry import known_exercises, resolve
from src.control.pipeline_manager import PipelineManager
from src.interfaces.voice_input import listen_once
from src.interfaces.voice_output import speak
from src.interfaces.wake_word import WakeWordDetector

_STOP_WORDS     = {"stop", "done", "finish", "quit", "end", "pause", "halt"}
_START_WORDS    = {"start", "begin", "do", "run", "launch", "switch", "change", "load"}
_STATUS_WORDS   = {"status", "what", "which", "current", "running"}
_SHUTDOWN_WORDS = {"shutdown", "exit", "terminate", "goodbye", "bye", "close", "kill"}

class Edith:
    """Voice-controlled controller for gym form analysis.

    Parameters
    ----------
    wake_phrase : str
        Word/phrase that activates Edith (default ``"edith"``).
    **pipeline_kwargs
        Forwarded to :class:`PipelineManager`
        (``model_path``, ``device``, ``camera``, ``fps``, ``threshold``, …).
    """

    def __init__(
        self,
        wake_phrase: str = "edith",
        **pipeline_kwargs,
    ) -> None:
        self._pipeline = PipelineManager(**pipeline_kwargs)
        self._detector = WakeWordDetector(
            on_wake=self._on_wake,
            wake_phrase=wake_phrase,
        )
        self._listening_for_command = False
        self._shutdown_event = threading.Event()
        # Speech queue: background threads enqueue text; main thread plays it.
        # Each item is (text, done_event) so callers can wait for completion.
        self._speech_queue: queue.Queue[tuple[str, threading.Event]] = queue.Queue()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start Edith.  Blocks until Ctrl-C or 'q' / Esc in the video window."""
        import cv2

        exercises = ", ".join(known_exercises())
        # First speak is on main thread directly — safe
        speak(f"Edith is online. Call my name. Available exercises: {exercises}.")
        self._detector.start()
        active_window: str = ""
        try:
            while not self._shutdown_event.is_set():
                # ── Drain speech queue (main-thread TTS) ──
                try:
                    text, done = self._speech_queue.get_nowait()
                    speak(text)
                    done.set()
                except queue.Empty:
                    pass

                # ── Display frames (main-thread imshow) ──
                frame = self._pipeline.get_frame()
                current_title = self._pipeline.window_title

                if frame is not None:
                    # Window title changed (exercise switched) — destroy old window first
                    if current_title != active_window:
                        if active_window:
                            cv2.destroyWindow(active_window)
                        active_window = current_title
                    cv2.imshow(current_title, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        self._pipeline.stop()
                        cv2.destroyAllWindows()
                        active_window = ""
                else:
                    # Pipeline stopped — clean up its window if still open
                    if active_window and not self._pipeline.is_running:
                        cv2.destroyAllWindows()
                        active_window = ""
                    time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            self._do_shutdown()

    def _shutdown(self) -> None:
        """Signal the main thread to shut down (safe to call from any thread)."""
        self._shutdown_event.set()

    def _do_shutdown(self) -> None:
        """Actual cleanup — always runs on the main thread."""
        import cv2
        self._pipeline.stop()
        cv2.destroyAllWindows()
        speak("Shutting down.")
        self._detector.stop()
        sys.exit(0)

    def _say(self, text: str) -> None:
        """Speak *text* from any thread, always executing on the main thread."""
        done = threading.Event()
        self._speech_queue.put((text, done))
        done.wait()  # block the caller until main thread finishes speaking

    # ── Wake word callback ─────────────────────────────────────────────────────

    def _on_wake(self) -> None:
        # Debounce — ignore a second trigger while already handling a command
        if self._listening_for_command:
            return
        self._listening_for_command = True
        try:
            self._say("Yes?")
            command = listen_once()
            if command:
                self._handle(command)
            else:
                self._say("I didn't catch that.")
        finally:
            self._listening_for_command = False

    # ── Intent handling ────────────────────────────────────────────────────────

    def _handle(self, text: str) -> None:
        words = set(re.sub(r"[^\w\s]", "", text.lower()).split())

        # ── Status query ──
        if words & _STATUS_WORDS:
            ex = self._pipeline.current_exercise
            if ex:
                entry = resolve(ex)
                name = entry.display_name if entry else ex
                self._say(f"Currently analysing {name}.")
            else:
                self._say("No exercise is running.")
            return

        # ── Stop command ──
        if (words & _STOP_WORDS) and not (words & _START_WORDS):
            if self._pipeline.is_running:
                ex = self._pipeline.current_exercise
                entry = resolve(ex) if ex else None
                name = entry.display_name if entry else ex
                self._pipeline.stop()
                self._say(f"Stopped {name}. Great work.")
            else:
                self._say("Nothing is running.")
            return    
        
        # ── Start / switch command ──
        matched = self._match_exercise(text)
        if matched:
            entry = resolve(matched)
            name = entry.display_name if entry else matched
            # Already running the same exercise — nothing to do
            if self._pipeline.is_running and self._pipeline.current_exercise == matched:
                self._say(f"We're already doing {name}.")
                return
            if self._pipeline.is_running:
                current_entry = resolve(self._pipeline.current_exercise or "")
                current_name = current_entry.display_name if current_entry else self._pipeline.current_exercise
                self._say(f"Switching from {current_name} to {name}.")
            else:
                self._say(f"Starting {name}. Get into position.")
            try:
                ok = self._pipeline.start(matched)
                if not ok:
                    self._say(f"Sorry, I don't recognise {matched} as an exercise.")
            except FileNotFoundError:
                self._say(
                    f"Checkpoint for {name} not found. "
                    "Please train the model first."
                )
            return

        # ── Shutdown command ──
        if words & _SHUTDOWN_WORDS:            
            self._shutdown()
            return

        # ── Unknown ──
        exercises = ", ".join(known_exercises())
        self._say(
            f"I didn't understand that. "
            f"Try: start overhead press, start squat, stop, or status. "
            f"Available exercises are: {exercises}."
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _match_exercise(self, text: str) -> str | None:
        """Return the registry key of the first exercise mentioned in *text*."""
        text_lower = text.lower()
        for key in known_exercises():
            entry = resolve(key)
            if entry is None:
                continue
            candidates = [key] + [a.lower() for a in entry.aliases]
            if any(c in text_lower for c in candidates):
                return key
        return None
