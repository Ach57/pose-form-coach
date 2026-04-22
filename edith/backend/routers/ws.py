"""
WebSocket Router
================
Single persistent connection per client.

Voice flow (backend-driven):
  VoiceLoop runs on the server — listens for "edith", speaks "Yes?",
  captures the command, parses intent, dispatches — all logged to the
  frontend via WakeEvent + LogEvent.

Quick-command buttons in the HUD also send CommandMessage frames which
bypass the voice loop and go straight to dispatch_command().
"""

import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pathlib import Path
import sys

from services.intent      import parse as parse_intent
from services.model_manager import ModelManager
from services.voice_loop  import VoiceLoop
from models.schemas import (
    CommandMessage,
    StateChangeEvent,
    StatsEvent,
    LogEvent,
    FrameEvent,
    WakeEvent,
)

_GYM_FORM = Path(__file__).resolve().parents[3] / "gym-form"
if str(_GYM_FORM) not in sys.path:
    sys.path.insert(0, str(_GYM_FORM))

router = APIRouter()

# Shared across the connection lifetime.
manager    = ModelManager()
voice_loop = VoiceLoop()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[EDITH] Client connected")

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def send_state(state: str, model=None):
        await websocket.send_text(
            StateChangeEvent(state=state, model=model).model_dump_json()
        )

    async def send_log(text: str, log_type: str = "info"):
        await websocket.send_text(
            LogEvent(text=text, type=log_type).model_dump_json()
        )
                
        if log_type != "voice":
            from src.interfaces.voice_output import speak_async
            await speak_async(text)

    async def send_stats(stats: dict):
        await websocket.send_text(StatsEvent(**stats).model_dump_json())

    async def send_frame(b64: str):
        await websocket.send_text(FrameEvent(data=b64).model_dump_json())

    async def send_wake(state: str, transcript: str | None = None):
        await websocket.send_text(
            WakeEvent(state=state, transcript=transcript).model_dump_json()
        )

    # ── Intent dispatch (shared by voice loop + button commands) ─────────────

    async def dispatch_command(text: str):
        intent = parse_intent(text)

        if intent.intent == "start":
            await manager.launch(
                intent.exercise, send_log, send_stats, send_state, send_frame
            )

        elif intent.intent == "switch":
            await manager.switch(
                intent.exercise, send_log, send_stats, send_state, send_frame
            )

        elif intent.intent == "stop":
            if manager.active_exercise:
                await manager.shutdown(send_log, send_state)
            else:
                await send_log("No exercise is running.", "warn")

        elif intent.intent == "shutdown":
            await manager.shutdown(send_log, send_state)

        elif intent.intent == "status":
            ex = manager.active_exercise
            if ex:
                from src.control.model_registry import resolve
                entry = resolve(ex)
                name  = entry.display_name if entry else ex
                await send_log(f"Currently analysing {name}.", "info")
            else:
                await send_log("No exercise is running.", "info")

        else:
            exercises = ", ".join(manager.known())
            await send_log(
                f"I didn't understand that. Try: 'start ohp', 'start squat', "
                f"'stop', 'switch', or 'status'. Available: {exercises}",
                "warn",
            )

    # ── Announce + start voice loop ───────────────────────────────────────────

    from src.control.model_registry import known_exercises
    exercises = ", ".join(known_exercises())
    await send_log("EDITH is online. Available exercises: " + exercises, "success")
    await send_state("idle")

    voice_loop.start(send_wake, send_log, dispatch_command)

    # ── Main receive loop (button commands from HUD) ──────────────────────────

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                data = json.loads(raw)
                msg  = CommandMessage(**data)
            except Exception:
                await send_log(f"Unrecognised message: {raw}", "error")
                continue

            text = msg.command.strip()
            await send_log(f'\u25b6 "{text}"', "voice")
            await dispatch_command(text)

    except WebSocketDisconnect:
        print("[EDITH] Client disconnected")
        voice_loop.stop()
        if manager.active_exercise:
            await manager._stop_inference()
