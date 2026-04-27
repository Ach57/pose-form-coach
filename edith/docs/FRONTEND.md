# Frontend

## Directory Structure

```
edith/frontend/src/
├── main.jsx                   ← React entry point
├── App.jsx                    ← Root component — wires all hooks + panels
│
├── hooks/
│   ├── useWebSocket.js        ← WS connection manager + sendCommand()
│   ├── useEdith.js            ← System state machine (idle/booting/active/shutdown)
│   ├── useVoice.js            ← Button simulation (typewriter effect)
│   └── useLogs.js             ← Log entry list with timestamps
│
├── components/
│   ├── panels/
│   │   ├── hudHeader.jsx      ← Top bar: system name, state badge, clock
│   │   ├── leftPanel.jsx      ← Inference module list
│   │   ├── centerPanel.jsx    ← Hero: orb / camera feed, boot bar, stats
│   │   ├── rightPanel.jsx     ← Live log feed
│   │   ├── bottomPanel.jsx    ← Wake-state label, voice wave, quick commands
│   │   └── modelCard.jsx      ← Card atom used inside leftPanel
│   └── ui/
│       ├── effects.jsx        ← HexGrid, ScanLine, PulseRing, VoiceWave
│       └── atoms.jsx          ← StatBar, ProgressBar, etc.
│
├── constants/
│   ├── index.js               ← Barrel re-export
│   ├── commands.js            ← Quick-command button labels
│   ├── models.js              ← Exercise metadata (id, name, label, color)
│   └── logColors.js           ← Log type → CSS color mapping
│
└── styles/
    └── hud.css                ← All styling (HUD panels, glow, animations)
```

---

## Component Tree

```
App
├── HexGrid          (background SVG lattice)
├── ScanLine         (CRT scanline effect)
│
├── HudHeader        systemState, activeModel
├── LeftPanel        systemState, activeModel
├── CenterPanel      systemState, activeModel, bootProgress, stats, frameSrc
├── RightPanel       systemState, activeModel, stats, logs, logRef
└── BottomPanel      wakeState, listening, transcript, onCommand(simulateVoice)
```

---

## Hooks

### `useWebSocket(onMessage)` → `{ sendCommand }`

Manages the single persistent WebSocket to `ws://localhost:8000/ws`.

**Key design:** the `onMessage` callback is stored in a `ref` and updated on every render. The `connect` function itself has no deps (`[]`), so it is created once and the WebSocket is opened exactly once — no reconnect loops caused by re-renders.

```
mount
  └─ connect()
       ├─ new WebSocket(WS_URL)
       ├─ ws.onmessage → onMessageRef.current(data)
       └─ ws.onclose   → setTimeout(connect, 3000)   [auto-reconnect]

sendCommand(text)
  └─ ws.send(JSON.stringify({ command: text }))
```

---

### `useEdith()` → `{ systemState, activeModel, bootProgress, stats, handleServerEvent }`

Pure state machine. No timers except the boot-progress animation. All transitions are driven by `state_change` events from the backend.

```
State transitions:

  idle
   │  state_change(booting, model)
   ▼
 booting  ──── progress animation creeps 0→95%
   │  state_change(active)
   ▼
 active   ──── stats updates, camera frames
   │  state_change(idle)  or  state_change(shutdown)
   ▼
  idle / shutdown
```

Boot progress bar: animates locally to 95%, then snaps to 100% when the server sends `active`. This gives instant visual feedback without blocking on the model load time.

---

### `useVoice(onCommand, wakeState)` → `{ transcript, simulateVoice }`

The backend owns all real voice I/O (microphone + TTS). This hook handles only the **quick-command button simulation**: a typewriter character-by-character animation followed by `onCommand(cmd)`.

```
simulateVoice("start ohp")
  ├─ for each char: setTimeout → setTranscript(partial)
  └─ final timeout → onCommand("start ohp") → sendCommand via WS
```

`wakeState` is passed in from `App.jsx` (set by WS `wake_state` events) so the simulation is blocked while the backend voice loop is active.

---

### `useLogs(initial)` → `{ logs, logRef, addLog }`

Maintains the ordered log list shown in `RightPanel`. `logRef` is a DOM ref used for auto-scrolling to the latest entry.

---

## State Ownership

| State | Owned by | Set by |
|---|---|---|
| `systemState` | `useEdith` | `state_change` WS event |
| `activeModel` | `useEdith` | `state_change` WS event |
| `bootProgress` | `useEdith` | local timer + `state_change(active)` |
| `stats` | `useEdith` | `stats` WS event |
| `frameSrc` | `App.jsx` | `frame` WS event (base64 JPEG → img src) |
| `wakeState` | `App.jsx` | `wake_state` WS event |
| `voiceTranscript` | `App.jsx` | `wake_state.transcript` field |
| `logs` | `useLogs` | `log` WS event |

The frontend has **zero simulation logic** — every meaningful state change originates from the backend.

---

## Panel Descriptions

### `HudHeader`
Top bar showing the system name "EDITH", the current `systemState` as a badge, and a live clock. Accent colour shifts with active model.

### `LeftPanel`
Lists all registered inference modules (OHP, Squat) as `ModelCard` components. Cards highlight when the corresponding model is active.

### `CenterPanel`
The hero of the HUD.

- **Idle / Booting:** displays a `PulseRing` orb with animated rings
- **Booting:** shows a loading progress bar below the orb
- **Active + frame received:** replaces the orb with a live `<img>` fed by the base64 JPEG stream from the backend
- **Stats:** `StatBar` components show confidence %, FPS, and latency ms

### `RightPanel`
Scrollable event log. Log entries are colour-coded by type (`info`, `warn`, `error`, `success`, `voice`). Auto-scrolls to latest entry via `logRef`.

### `BottomPanel`
Voice status zone. The label and `VoiceWave` animation reflect the backend's wake state:

| `wakeState` | Label | Style |
|---|---|---|
| `idle` | `EDITH STANDBY — Say "Edith"` | Dim cyan |
| `awake` | `EDITH AWAKE — Yes?` | Green pulse glow |
| `listening` | `EDITH LISTENING — Speak your command...` | Cyan glow |

Quick-command buttons trigger `simulateVoice()` which sends the command directly over WebSocket, bypassing the voice loop.

---

## Styling Approach

All styles live in a single `hud.css`. The design language is a sci-fi terminal HUD:

- **Palette:** `#00f0ff` (cyan primary), `#00ff88` (green accent), dark backgrounds
- **Effects:** `text-shadow` glow, `box-shadow` panel borders, CSS keyframe animations for pulse rings, scanline overlay, noise texture
- **Layout:** CSS Grid (`hud-layout`) for the panel grid; flexbox inside each panel
- **Dynamic:** accent colours are passed as inline `style` props from the active model's `color` field in `constants/models.js`
