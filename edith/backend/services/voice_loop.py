"""
Voice Loop Service
==================
Backend wake-word + command listener that mirrors start_edith.py's voice flow.

Flow
----
1. Phase 1 (always-on): run listen_once() in an executor.
   If "edith" is in the result → advance to phase 2.

2. Phase 2 (command): speak "Yes?", send wake event to frontend,
   run listen_once() again, parse intent, dispatch command.

All blocking I/O (mic, TTS) runs in an asyncio executor so the event loop
stays free for WebSocket traffic and frame streaming.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import sys
from typing import Awaitable, Callable

_GYM_FORM = Path(__file__).resolve().parents[3] / "gym-form"
if str(_GYM_FORM) not in sys.path:
    sys.path.insert(0, str(_GYM_FORM))

from src.interfaces.voice_input  import listen_once
from src.interfaces.voice_output import speak_async

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # edith/backend
from constants.state import WakeState, Logstate

log = logging.getLogger(__name__)

WAKE_PHRASE = "edith"


class VoiceLoop:
    """Single instance per WebSocket connection; drives the wake/command cycle."""

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._active = False

    # ── Public API ────────────────────────────────────────────────────────────

    def start(
        self,
        send_wake,           # async (state, transcript=None) → None
        send_log,            # async (text, type) → None
        dispatch_command,    # async (text) → None
    ) -> None:
        """Launch the background voice loop (idempotent)."""
        if self._task and not self._task.done():
            return
        self._active = True
        self._task = asyncio.create_task(
            self._loop(send_wake, send_log, dispatch_command),
            name="voice-loop",
        )

    def stop(self) -> None:
        """Cancel the background task."""
        self._active = False
        if self._task:
            self._task.cancel()
            self._task = None

    # ── Internal loop ─────────────────────────────────────────────────────────

    async def _loop(self, send_wake, send_log, dispatch_command) -> None:
        loop = asyncio.get_event_loop()

        while self._active:
            # ── Phase 1: listen for wake word ─────────────────────────────
            try:
                text: str = await loop.run_in_executor(None, listen_once)
            except Exception as exc:
                log.warning("[VoiceLoop] listen error: %s", exc)
                await asyncio.sleep(1)
                continue

            if not text or WAKE_PHRASE not in text.lower():
                continue

            # ── Wake word detected ─────────────────────────────────────────
            await send_wake(WakeState.AWAKE, "Yes?")
            await send_log("Yes?", Logstate.VOICE)

            # Speak "Yes?" directly (it's an async coroutine)
            try:
                await speak_async("Yes?")
            except Exception as exc:
                log.warning("[VoiceLoop] TTS error: %s", exc)

            # ── Phase 2: capture command ───────────────────────────────────
            await send_wake(WakeState.LISTENING)

            try:
                command: str = await loop.run_in_executor(None, listen_once)
            except Exception as exc:
                log.warning("[VoiceLoop] command listen error: %s", exc)
                await send_wake(WakeState.IDLE)
                continue

            await send_wake(WakeState.IDLE)

            if command:
                await send_log(f'\u25b6 "{command}"', Logstate.VOICE)
                await dispatch_command(command)
