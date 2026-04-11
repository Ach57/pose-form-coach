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

import re
import time

from src.control.model_registry import known_exercises, resolve
from src.control.pipeline_manager import PipelineManager
from src.interfaces.voice_input import listen_once
from src.interfaces.voice_output import speak
from src.interfaces.wake_word import WakeWordDetector

_STOP_WORDS  = {"stop", "done", "finish", "quit", "end", "pause", "halt"}
_START_WORDS = {"start", "begin", "do", "run", "launch", "switch", "change", "load"}
_STATUS_WORDS = {"status", "what", "which", "current", "running"}


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

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start Edith.  Blocks until Ctrl-C or 'q' / Esc in the video window."""
        import cv2

        exercises = ", ".join(known_exercises())
        speak(f"Edith is online. Say my name to give a command. Available exercises: {exercises}.")
        self._detector.start()
        try:
            while True:
                # cv2.imshow MUST run on the main thread (macOS AppKit requirement).
                # The pipeline pushes annotated frames via a queue; we display them here.
                frame = self._pipeline.get_frame()
                if frame is not None:
                    cv2.imshow(self._pipeline.window_title, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):   # q or Esc closes the window
                        self._pipeline.stop()
                        cv2.destroyAllWindows()
                else:
                    # No active pipeline — yield so the voice thread stays responsive
                    time.sleep(0.02)
        except KeyboardInterrupt:
            self._shutdown()

    def _shutdown(self) -> None:
        self._detector.stop()
        self._pipeline.stop()
        speak("Edith shutting down. Good workout.")

    # ── Wake word callback ─────────────────────────────────────────────────────

    def _on_wake(self) -> None:
        # Debounce — ignore a second trigger while already handling a command
        if self._listening_for_command:
            return
        self._listening_for_command = True
        try:
            speak("Yes?")
            command = listen_once()
            if command:
                self._handle(command)
            else:
                speak("I didn't catch that.")
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
                speak(f"Currently analysing {name}.")
            else:
                speak("No exercise is running.")
            return

        # ── Stop command ──
        if (words & _STOP_WORDS) and not (words & _START_WORDS):
            if self._pipeline.is_running:
                ex = self._pipeline.current_exercise
                entry = resolve(ex) if ex else None
                name = entry.display_name if entry else ex
                self._pipeline.stop()
                speak(f"Stopped {name}. Great work.")
            else:
                speak("Nothing is running.")
            return

        # ── Start / switch command ──
        matched = self._match_exercise(text)
        if matched:
            entry = resolve(matched)
            name = entry.display_name if entry else matched
            if self._pipeline.is_running:
                current_entry = resolve(self._pipeline.current_exercise or "")
                current_name = current_entry.display_name if current_entry else self._pipeline.current_exercise
                speak(f"Switching from {current_name} to {name}.")
            else:
                speak(f"Starting {name}. Get into position.")
            try:
                ok = self._pipeline.start(matched)
                if not ok:
                    speak(f"Sorry, I don't recognise {matched} as an exercise.")
            except FileNotFoundError:
                speak(
                    f"Checkpoint for {name} not found. "
                    "Please train the model first."
                )
            return

        # ── Unknown ──
        exercises = ", ".join(known_exercises())
        speak(
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
