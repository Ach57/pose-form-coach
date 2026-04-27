# WebSocket Data Flow

## Protocol Summary

All communication between frontend and backend uses a single WebSocket connection at `ws://localhost:8000/ws`. Messages are JSON objects. The `event` field acts as a discriminator.

### Message Directions

```
Client (React) ──────────────────────────────────► Server (FastAPI)
  { "command": "start ohp" }                         CommandMessage

Server (FastAPI) ────────────────────────────────► Client (React)
  { "event": "state_change", ... }                   StateChangeEvent
  { "event": "log",          ... }                   LogEvent
  { "event": "stats",        ... }                   StatsEvent
  { "event": "frame",        ... }                   FrameEvent
  { "event": "wake_state",   ... }                   WakeEvent
```

---

## Message Schemas

### `CommandMessage` (Client → Server)

```json
{
  "command": "start overhead press"
}
```

Free-text natural language. Parsed by `services/intent.py` on the backend.

---

### `StateChangeEvent` (Server → Client)

```json
{
  "event": "state_change",
  "state": "booting",
  "model": "ohp"
}
```

| Field | Type | Values |
|---|---|---|
| `state` | `SystemState` | `idle` · `booting` · `active` · `shutdown` |
| `model` | `string \| null` | Registry key of active exercise, or null |

Drives `useEdith`'s state machine in the frontend.

---

### `LogEvent` (Server → Client)

```json
{
  "event": "log",
  "text": "Initiating boot sequence — Overhead Press",
  "type": "info"
}
```

| Field | Type | Values |
|---|---|---|
| `type` | `Logstate` | `info` · `warn` · `error` · `success` · `voice` |

`voice` type is used for transcribed user speech (shown differently in the log panel).

---

### `StatsEvent` (Server → Client)

```json
{
  "event": "stats",
  "confidence": 87,
  "fps": 24,
  "latency": 42
}
```

Emitted by `ModelManager._drain_loop` at every inference cycle while a model is active. Updates `StatBar` components in `CenterPanel`.

---

### `FrameEvent` (Server → Client)

```json
{
  "event": "frame",
  "data": "/9j/4AAQSkZJRgABAQAA..."
}
```

Base64-encoded JPEG of the annotated camera frame. In `App.jsx`:

```js
setFrameSrc(`data:image/jpeg;base64,${data.data}`)
```

This string is set directly as the `src` of an `<img>` in `CenterPanel`.

---

### `WakeEvent` (Server → Client)

```json
{
  "event": "wake_state",
  "state": "awake",
  "transcript": "Yes?"
}
```

| Field | Type | Values |
|---|---|---|
| `state` | `WakeState` | `idle` · `awake` · `listening` |
| `transcript` | `string \| null` | Short text to show in the voice display area |

---

## Connection Lifecycle Sequence

```
Browser loads           Backend starts
     │                       │
     ├── new WebSocket() ─────►
     │                       ├── accept()
     │                       ├── send LogEvent("EDITH is online...", success)
     │                       ├── send StateChangeEvent(idle)
     │                       └── VoiceLoop.start()
     │                                │
     │◄── LogEvent ──────────────────┤
     │◄── StateChangeEvent ──────────┤
     │                               │
     │     [user says "edith"]       │
     │                       ├── send WakeEvent(awake, "Yes?")
     │                       ├── send LogEvent("Yes?", voice)
     │                       │   [speaks "Yes?" via TTS]
     │                       └── send WakeEvent(listening)
     │◄── WakeEvent(awake) ─────────┤
     │◄── LogEvent("Yes?") ─────────┤
     │◄── WakeEvent(listening) ─────┤
     │                               │
     │     [user says "start ohp"]   │
     │                       ├── send WakeEvent(idle)
     │                       ├── send LogEvent(▶ "start ohp", voice)
     │                       ├── [dispatch_command → manager.launch]
     │                       ├── send StateChangeEvent(booting, "ohp")
     │                       ├── send LogEvent("Initiating boot sequence...", info)
     │                       ├── send LogEvent("Loading neural weights...", info)
     │                       │   [model loads in executor thread]
     │                       ├── send StateChangeEvent(active, "ohp")
     │                       └── [camera thread starts]
     │◄── WakeEvent(idle) ──────────┤
     │◄── LogEvent ─────────────────┤  × multiple
     │◄── StateChangeEvent(booting)─┤
     │◄── StateChangeEvent(active) ─┤
     │                               │
     │   [every ~33ms while active]  │
     │◄── FrameEvent ───────────────┤  (b64 JPEG)
     │◄── StatsEvent ───────────────┤  (confidence, fps, latency)
     │                               │
     │   [user clicks "stop" button] │
     ├── CommandMessage("stop") ─────►
     │                       ├── send LogEvent(▶ "stop", voice)
     │                       ├── [manager.shutdown()]
     │                       └── send StateChangeEvent(idle)
     │◄── LogEvent ─────────────────┤
     │◄── StateChangeEvent(idle) ───┤
     │                               │
     │   [browser closes]            │
     ├── WebSocket close ────────────►
                             ├── VoiceLoop.stop()
                             └── manager._stop_inference()
```

---

## Frontend Event Routing — `App.jsx`

```js
useWebSocket((data) => {
  if      (data.event === "log")         addLog(data.text, data.type)
  else if (data.event === "frame")       setFrameSrc(`data:image/jpeg;base64,${data.data}`)
  else if (data.event === "wake_state")  { setWakeState(data.state); setVoiceTranscript(data.transcript) }
  else                                   handleServerEvent(data)  // state_change, stats
})
```

`handleServerEvent` in `useEdith` handles `state_change` and `stats`.  
`log` and `frame` and `wake_state` are handled directly in `App.jsx` because they require state not owned by `useEdith`.

---

## Reconnect Strategy

If the WebSocket closes (server restart, network interruption), the frontend retries after 3 seconds:

```js
ws.onclose = () => {
  reconnectRef.current = setTimeout(connect, 3000);
};
```

`connect` has stable identity (`[]` deps) so there is no risk of multiple reconnect timers racing.
