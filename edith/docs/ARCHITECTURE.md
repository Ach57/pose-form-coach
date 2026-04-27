# Architecture Overview

## High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER ENVIRONMENT                           │
│                                                                     │
│   🎤 Microphone ──────────────────────────────────────────────┐    │
│   🖥  Browser (localhost:5173)                                 │    │
│   │                                                            │    │
│   │   ┌─────────────────────────────────────────┐             │    │
│   │   │           REACT HUD (Frontend)           │             │    │
│   │   │                                          │             │    │
│   │   │  HudHeader  LeftPanel  CenterPanel       │             │    │
│   │   │  RightPanel  BottomPanel                 │             │    │
│   │   │                                          │             │    │
│   │   │  useWebSocket ◄──── WS events ────────┐  │             │    │
│   │   │  useEdith    (state machine)           │  │             │    │
│   │   │  useVoice    (button simulation)       │  │             │    │
│   │   └────────────────┬───────────────────────┘  │             │    │
│   │                    │ WebSocket                 │             │    │
│   │                    │ ws://localhost:8000/ws    │             │    │
│   └────────────────────▼───────────────────────────┘             │    │
│                                                                     │    │
│   ┌─────────────────────────────────────────────────┐             │    │
│   │            FASTAPI BACKEND (localhost:8000)      │◄────────────┘    │
│   │                                                  │                   │
│   │  ┌──────────┐  ┌───────────┐  ┌──────────────┐ │                   │
│   │  │VoiceLoop │  │  Intent   │  │ ModelManager │ │                   │
│   │  │(wake word│  │  Parser   │  │ (ML engine)  │ │                   │
│   │  │ + TTS)   │  │           │  │              │ │                   │
│   │  └────┬─────┘  └─────┬─────┘  └──────┬───────┘ │                   │
│   │       │              │               │           │                   │
│   │       └──────────────┴───────────────┘           │                   │
│   │                      │                           │                   │
│   │              ┌───────▼────────┐                  │                   │
│   │              │  gym-form lib  │                  │                   │
│   │              │  OHPPredictor  │                  │                   │
│   │              │  SquatPredictor│                  │                   │
│   │              │  MediaPipe     │                  │                   │
│   │              │  CausalTCN     │                  │                   │
│   │              └────────────────┘                  │                   │
│   └─────────────────────────────────────────────────┘                   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### Backend

| Technology            | Version         | Role                                   |
| --------------------- | --------------- | -------------------------------------- |
| **Python**            | 3.14 (Homebrew) | Runtime                                |
| **FastAPI**           | latest          | HTTP + WebSocket server                |
| **uvicorn**           | latest          | ASGI server                            |
| **Pydantic v2**       | latest          | Message schema validation              |
| **OpenCV**            | `opencv-python` | Camera capture, frame annotation       |
| **MediaPipe**         | (via gym-form)  | Pose landmark detection                |
| **PyTorch**           | (via gym-form)  | CausalTCN inference                    |
| **SpeechRecognition** | latest          | Microphone input (wake word + command) |
| **edge-tts**          | latest          | Neural TTS ("Yes?" voice response)     |
| **python-dotenv**     | latest          | Environment variable loading           |

### Frontend

| Technology       | Version | Role                                  |
| ---------------- | ------- | ------------------------------------- |
| **React**        | 18      | UI framework                          |
| **Vite**         | latest  | Dev server + bundler                  |
| **CSS (custom)** | —       | HUD styling, animations, glow effects |

---

## Connectivity

```
Frontend  ──── WebSocket (ws://localhost:8000/ws) ────  Backend
              (persistent, full-duplex, JSON frames)

Frontend  ──── GET /health ───────────────────────────  Backend
              (REST, optional health check)
```

### Why WebSocket (not REST)?

The system requires **three concurrent, asynchronous data streams** from backend → frontend:

1. **Annotated video frames** at ~15–30 fps (base64 JPEG)
2. **Inference statistics** (confidence, fps, latency) every ~500 ms
3. **Log events** and **state change notifications** on demand

REST polling cannot achieve the latency or throughput needed. A single persistent WebSocket connection handles all three streams efficiently with a shared `asyncio.Queue`.

---

## Monorepo Layout

```
Fitness/
├── gym-form/               ← ML library (predictors, registry, voice I/O)
│   ├── src/
│   │   ├── control/
│   │   │   └── model_registry.py
│   │   ├── realtime/
│   │   │   └── predictor.py   (OHPPredictor, SquatPredictor, BasePredictor)
│   │   ├── interfaces/
│   │   │   ├── voice_input.py   (listen_once via SpeechRecognition)
│   │   │   └── voice_output.py  (speak_async via edge-tts)
│   │   └── utils/
│   │       └── openCv.py
│   ├── checkpoints/
│   │   ├── ohp/best.pt
│   │   └── squat/best.pt
│   └── artifacts/
│       └── pose_landmarker_heavy.task
│
└── edith/                  ← UI project (this repo)
    ├── backend/            ← FastAPI + services
    └── frontend/           ← React HUD
```

The backend references `gym-form` via `sys.path` injection at runtime — no packaging required.

---

## Deployment Model

Both processes run locally on the same machine:

```
uvicorn main:app --reload --port 8000   (backend)
npm run dev                              (frontend, port 5173)
```

CORS is configured to allow `localhost:5173` → `localhost:8000`. The microphone and camera are accessed directly by the backend process (no browser permissions required for audio input).
