# E.D.I.T.H — Pose Detection System

> Even Dead I'm The Hero

Voice-controlled gym pose detection UI with a React HUD frontend and FastAPI backend.

---

## Project Structure

```
edith/
├── frontend/               # React + Vite app
│   └── src/
│       ├── components/
│       │   ├── ui/         # Primitive, reusable UI atoms
│       │   └── panels/     # HUD layout panels (Left, Center, Right, Bottom)
│       ├── hooks/          # Custom React hooks (useEdith, useWebSocket, useLogs)
│       ├── constants/      # Models config, commands list
│       ├── styles/         # Global CSS
│       └── App.jsx         # Root — composes panels
│
└── backend/                # FastAPI Python app
    ├── main.py             # App entry, WebSocket endpoint
    ├── routers/
    │   └── ws.py           # WebSocket route handler
    ├── services/
    │   └── model_manager.py  # Your ML model launch/switch/shutdown logic
    └── models/
        └── schemas.py      # Pydantic event schemas
```

---

## Quick Start

### Backend

```bash
cd backend
pip install fastapi uvicorn websockets
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` — it connects to the backend at `ws://localhost:8000/ws`.

---

## WebSocket Protocol

**Frontend → Backend**

```json
{ "command": "LAUNCH MODEL ALPHA" }
{ "command": "SWITCH MODEL" }
{ "command": "SHUTDOWN" }
```

**Backend → Frontend**

```json
{ "event": "state_change", "state": "booting", "model": "alpha" }
{ "event": "state_change", "state": "active",  "model": "alpha" }
{ "event": "stats",        "confidence": 92, "fps": 30, "latency": 14 }
{ "event": "log",          "text": "Model online", "type": "success" }
{ "event": "state_change", "state": "idle",   "model": null }
```
