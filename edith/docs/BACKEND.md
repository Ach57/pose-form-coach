# Backend

## Directory Structure

```
edith/backend/
├── main.py                  ← FastAPI app factory, CORS, router mounting
├── requirements.txt
├── .env                     ← Environment variables (word sets, origins, title)
│
├── config/
│   └── config.py            ← Typed env-var loading (dotenv → Python objects)
│
├── constants/
│   ├── constants.py         ← Env-var key names
│   ├── state.py             ← All enums (SystemState, WakeState, Logstate, …)
│   └── text.py              ← Regex patterns, string constants
│
├── models/
│   └── schemas.py           ← Pydantic message models (in/out over WebSocket)
│
├── routers/
│   ├── ws.py                ← /ws WebSocket endpoint (main entry point)
│   └── health.py            ← GET /health
│
└── services/
    ├── intent.py            ← Natural-language → IntentResult parser
    ├── model_manager.py     ← ML lifecycle: load, run, stream, stop
    └── voice_loop.py        ← Wake-word detection + TTS response loop
```

---

## Application Entry Point — `main.py`

```
FastAPI app
  │
  ├── CORSMiddleware  (allows localhost:5173)
  ├── /ws             (routers/ws.py)
  └── /health         (routers/health.py)
```

FastAPI is run via `uvicorn` (ASGI). The app is stateless at the HTTP layer — all persistent state lives inside the module-level `ModelManager` and `VoiceLoop` instances in `ws.py`.

---

## WebSocket Router — `routers/ws.py`

The single `/ws` endpoint is the heart of the backend.  
On connection it:

1. Sends an announce `log` event (available exercises)
2. Sends `state_change → idle`
3. Starts `VoiceLoop` (background task)
4. Enters the main `receive_text` loop (button commands from the HUD)

```
Client connects
     │
     ▼
 send_log("EDITH is online...")  ──► LogEvent over WS
 send_state(SystemState.IDLE)    ──► StateChangeEvent over WS
 voice_loop.start(...)           ──► asyncio.Task launched
     │
     ▼
 loop: receive_text()
     │
     ├─ parse JSON → CommandMessage
     ├─ send_log(▶ "text", VOICE)
     └─ dispatch_command(text)
              │
              ├─ intent == START    → manager.launch(...)
              ├─ intent == STOP     → manager.shutdown(...)
              ├─ intent == SWITCH   → manager.switch(...)
              ├─ intent == STATUS   → send_log(active exercise)
              ├─ intent == SHUTDOWN → manager.shutdown(...)
              └─ intent == UNKNOWN  → send_log(help text, WARN)
```

`dispatch_command` is shared between the button command loop and the `VoiceLoop` — both paths produce identical outcomes.

---

## Services

### `services/intent.py` — Intent Parser

Parses free-text natural language into a structured `IntentResult`.  
Direct port of `edith.py`'s `_handle()` / `_match_exercise()` logic.

**Input:** `"start overhead press"`  
**Output:** `IntentResult(intent=IntentType.START, exercise="ohp")`

```
text
  │
  ▼
tokenise (lowercase, strip punctuation)
  │
  ├─ words ∩ SHUTDOWN_WORDS  → IntentType.SHUTDOWN
  ├─ words ∩ STATUS_WORDS    → IntentType.STATUS
  ├─ words ∩ STOP_WORDS      → IntentType.STOP
  └─ words ∩ START_WORDS
           │
           ├─ _match_exercise(text) found  → IntentType.START + exercise key
           ├─ words ∩ SWITCH_WORDS         → IntentType.SWITCH
           └─ otherwise                    → IntentType.UNKNOWN
```

Word sets (`START_WORDS`, `STOP_WORDS`, etc.) are loaded from `.env` via `config.py`.  
Exercise matching checks gym-form's registry keys **and** all declared aliases.

---

### `services/model_manager.py` — ML Lifecycle Manager

Manages the full lifecycle of an exercise predictor: load → run → stream → stop.

#### Class: `ModelManager`

```
ModelManager
  ├── active_exercise: Optional[str]       registry key ("ohp", "squat", …)
  ├── _q: asyncio.Queue                    shared frame+stats queue
  ├── _thread: threading.Thread            camera daemon thread
  └── _drain: asyncio.Task                 async consumer of the queue
```

#### `launch(exercise_key, ...)` sequence

```
launch("ohp")
  │
  ├─ send state_change(BOOTING)
  ├─ resolve entry from model_registry
  ├─ run_in_executor → predictor.from_checkpoint(ckpt, mediapipe_task)
  ├─ send state_change(ACTIVE)
  ├─ start _camera_thread(predictor)   [daemon thread]
  └─ start _drain_loop(...)            [asyncio Task]
```

#### Camera Thread → asyncio Queue → WebSocket

```
[thread]                     [asyncio event loop]
cv2.VideoCapture
    │
    ├─ predictor.infer(frame)   (MediaPipe + CausalTCN)
    ├─ draw_skeleton(frame)
    ├─ draw_overlay(frame, stats)
    ├─ cv2.imencode → base64
    │
    ▼
_enqueue(("frame", b64))  ──call_soon_threadsafe──►  asyncio.Queue
_enqueue(("stats", {...}))                                │
                                                          ▼
                                                    _drain_loop
                                                          │
                                                    ├─ on_frame(b64)  → FrameEvent → WS
                                                    └─ on_stats(dict) → StatsEvent → WS
```

Frame drops: if the queue has ≥ 2 items the new frame is silently dropped to prevent backpressure.

---

### `services/voice_loop.py` — Wake-Word Loop

See [VOICE-PIPELINE.md](./VOICE-PIPELINE.md) for the full flow.  
Runs as a persistent `asyncio.Task` alongside the WebSocket receive loop.

---

## Data Models — `models/schemas.py`

All WebSocket messages are Pydantic models serialised as JSON.

| Class | Direction | Discriminator (`event`) |
|---|---|---|
| `CommandMessage` | Client → Server | *(none, always command)* |
| `StateChangeEvent` | Server → Client | `"state_change"` |
| `StatsEvent` | Server → Client | `"stats"` |
| `LogEvent` | Server → Client | `"log"` |
| `FrameEvent` | Server → Client | `"frame"` |
| `WakeEvent` | Server → Client | `"wake_state"` |

---

## Constants & Enums — `constants/state.py`

```python
SystemState   →  idle | booting | active | shutdown
WakeState     →  idle | awake | listening
Logstate      →  info | warn | error | success | voice
IntentType    →  start | stop | switch | status | shutdown | unknown
EventType     →  state_change
WakeType      →  wake_state
ModelStatsEvent → stats
LogEventType  →  log
FrameEventType → frame
```

All enums extend `str` so they serialise directly to JSON strings without custom encoders.

---

## gym-form Integration

The backend has no ML code of its own. All inference is delegated to the `gym-form` library, which is imported at runtime via `sys.path` injection:

```python
_GYM_FORM = Path(__file__).resolve().parents[3] / "gym-form"
sys.path.insert(0, str(_GYM_FORM))

from src.control.model_registry import resolve, known_exercises
from src.realtime.predictor import OHPPredictor, SquatPredictor
from src.interfaces.voice_input  import listen_once
from src.interfaces.voice_output import speak_async
from src.utils.openCv import draw_skeleton, _draw_overlay
```

The `model_registry` is the single source of truth for which exercises exist, what their display names are, and where their checkpoints live.
