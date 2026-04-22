"""Wake-word detection for Edith using SpeechRecognition.

Listens in short bursts via Google STT and checks whether the
*wake_phrase* (default "edith") appears in the transcript.
Requires an internet connection.

Usage
-----
    detector = WakeWordDetector(on_wake=my_callback)
    detector.start()
    ...
    detector.stop()
"""

from __future__ import annotations

import threading
from typing import Callable


class WakeWordDetector:
    """Listens in a background daemon thread and calls *on_wake* when the
    wake phrase is detected.

    Parameters
    ----------
    on_wake : Callable[[], None]
        Zero-argument callback invoked on the detector thread.
    wake_phrase : str
        Phrase to match in the transcript (default ``"edith"``).
    """

    def __init__(
        self,
        on_wake: Callable[[], None],
        wake_phrase: str = "edith",
    ) -> None:
        self._on_wake = on_wake
        self._wake_phrase = wake_phrase.lower()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start listening in a background daemon thread."""
        self._stop_event.clear()
        print(f"[WakeWord] Starting  (phrase: '{self._wake_phrase}')")
        self._thread = threading.Thread(target=self._listen_sr, daemon=True, name="wake-word")
        self._thread.start()

    def stop(self) -> None:
        """Signal the listener to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    # ── SpeechRecognition listener ────────────────────────────────────────────

    def _listen_sr(self) -> None:
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True

        # Calibrate once on startup
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)

        while not self._stop_event.is_set():
            try:
                with sr.Microphone() as source:
                    # Short bursts — sufficient to catch a wake phrase
                    audio = recognizer.listen(source, timeout=3, phrase_time_limit=4)
                text = recognizer.recognize_google(audio).lower()
                if self._wake_phrase in text:
                    self._on_wake()
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                pass
