"""
WebSocket Router
================
Single persistent connection per client.
Receives natural-language commands from the React HUD, parses them with the
same intent engine as edith.py, and dispatches to ModelManager.
"""

import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services.intent import parse as parse_intent
from services.model_manager import ModelManager
from models.schemas import (
    CommandMessage,
    StateChangeEvent,
    StatsEvent,
    LogEvent,
    FrameEvent,
)

router = APIRouter()

# One ModelManager shared across the connection lifetime.
manager = ModelManager()


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

    async def send_stats(stats: dict):
        await websocket.send_text(
            StatsEvent(**stats).model_dump_json()
        )

    async def send_frame(b64: str):
        await websocket.send_text(
            FrameEvent(data=b64).model_dump_json()
        )

    # ── Announce connection ───────────────────────────────────────────────────
    from src.control.model_registry import known_exercises
    exercises = ", ".join(known_exercises())
    await send_log("EDITH online. Available exercises: " + exercises, "success")
    await send_state("idle")

    # ── Main receive loop ─────────────────────────────────────────────────────
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
            await send_log(f'▶ "{text}"', "voice")

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

    except WebSocketDisconnect:
        print("[EDITH] Client disconnected")
        if manager.active_exercise:
            await manager._stop_inference()
