# Voice Pipeline

## Overview

EDITH uses a **two-phase, backend-driven voice loop**. All microphone access and text-to-speech happen on the Python server — the browser has no audio permissions requirement and is purely a display surface for voice state.

```
                    ┌────────────────────────────────────┐
                    │        BACKEND  (Python)            │
                    │                                     │
  🎤 System Mic ──► │  Phase 1: Wake-word detection       │
                    │    listen_once() [blocking, thread] │
                    │    "edith" detected?                │
                    │          │                          │
                    │          ▼                          │
                    │  send WakeEvent(AWAKE, "Yes?") ─────┼──► WebSocket ──► HUD label changes
                    │  send_log("Yes?", VOICE)       ─────┼──► WebSocket ──► Log entry
                    │  speak_async("Yes?")                │   (edge-tts → afplay)
                    │          │                          │
                    │          ▼                          │
                    │  Phase 2: Command capture           │
                    │    send WakeEvent(LISTENING)   ─────┼──► WebSocket ──► HUD label changes
                    │    listen_once() [blocking, thread] │
                    │    command text returned            │
                    │          │                          │
                    │          ▼                          │
                    │  send WakeEvent(IDLE)          ─────┼──► WebSocket ──► HUD resets
                    │  send_log(▶ "command", VOICE)  ─────┼──► WebSocket ──► Log entry
                    │  dispatch_command(text)             │
                    │          │                          │
                    │          ▼                          │
                    │  Intent parser → ModelManager       │
                    └────────────────────────────────────┘
```

---

## Phase 1 — Wake-Word Detection

**Library:** `speech_recognition` (Google Web Speech API backend)  
**Function:** `listen_once()` from `gym-form/src/interfaces/voice_input.py`

`listen_once()` blocks the calling thread until it receives a complete utterance. The `VoiceLoop` runs this inside `asyncio.get_event_loop().run_in_executor(None, listen_once)` so the event loop is never blocked — the WebSocket remain fully responsive during mic listening.

The detected text is lowercased and checked for the string `"edith"` anywhere in the utterance. This makes the trigger robust — "hey edith", "ok edith", "yo edith" all work.

---

## Phase 2 — Command Capture

After the wake word is detected:

1. Backend sends `WakeEvent(state=AWAKE, transcript="Yes?")` over WebSocket → HUD label glows green
2. Backend logs `"Yes?"` as a `VOICE` log entry → appears in RightPanel log
3. `speak_async("Yes?")` synthesises and plays audio immediately (blocking but inside the task, not the event loop — TTS is awaited directly)
4. Backend sends `WakeEvent(state=LISTENING)` → HUD label changes to cyan
5. Second `listen_once()` call captures the user's command
6. Backend sends `WakeEvent(state=IDLE)` → HUD resets
7. Command is logged and dispatched

---

## Text-to-Speech (TTS)

**Library:** `edge-tts` (Microsoft Edge Neural TTS, offline-capable)  
**Voice:** `en-US-JennyNeural` (default)  
**Playback:** `afplay` (macOS built-in audio player)

```python
async def speak_async(text: str) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save("/tmp/edith_tts.mp3")
    subprocess.run(["afplay", "/tmp/edith_tts.mp3"], check=False)
```

TTS is only triggered from the `VoiceLoop` ("Yes?"). Log messages sent by `send_log` from the rest of the system are **not** spoken aloud — this avoids Edith narrating her own internal status messages.

---

## Concurrency Model

```
asyncio event loop (main thread)
  │
  ├── WebSocket receive loop       (awaits receive_text)
  ├── VoiceLoop._loop task         (awaits run_in_executor)
  └── ModelManager._drain_loop     (awaits queue.get)
           ▲
           │  thread-safe via call_soon_threadsafe
           │
  [Camera thread]  (daemon, blocking cv2 loop)
```

The critical constraint: `listen_once()` and `afplay` are both **blocking**. They run in thread-pool executors so they do not stall the event loop. The camera thread and drain loop continue to operate independently while the voice loop is waiting for mic input.

---

## WakeEvent Protocol

The frontend reflects backend voice state via `WakeEvent` messages:

```json
{ "event": "wake_state", "state": "awake",     "transcript": "Yes?" }
{ "event": "wake_state", "state": "listening",  "transcript": null   }
{ "event": "wake_state", "state": "idle",       "transcript": null   }
```

`App.jsx` handles these by updating `wakeState` and `voiceTranscript` state, which flow down to `BottomPanel` for display. The frontend never drives these states — it only reflects them.

---

## Quick-Command Button Bypass

HUD buttons do **not** go through the voice pipeline. They send a `CommandMessage` directly over WebSocket → the main receive loop → `dispatch_command()`. The `VoiceLoop` is unaware of button commands. This means both input paths (voice + buttons) produce identical side effects via the shared `dispatch_command` function.

---

## Error Handling

| Error | Behaviour |
|---|---|
| Mic not available / no audio | `listen_once()` raises → `VoiceLoop` logs warning, sleeps 1s, retries |
| TTS fails | `speak_async` exception caught → warning logged, loop continues |
| Command capture returns empty string | Silently ignored, loop restarts phase 1 |
| Wake word not in utterance | Silently ignored, loop restarts phase 1 |
