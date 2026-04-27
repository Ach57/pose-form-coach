# EDITH — Documentation Index

> **E**xercise **D**iagnostic and **I**nference **T**raining **H**ub  
> A real-time AI gym-form coach with a voice-first interface.

---

## What is EDITH?

EDITH is a browser-based HUD (Heads-Up Display) that wraps the `gym-form` machine-learning pipeline and exposes it through a conversational, voice-activated interface. You speak to it — "Edith, start overhead press" — it loads the correct neural network, opens the camera, and streams annotated form-feedback frames back to a React dashboard in real time.

It has two physical layers:

| Layer | Technology | Role |
|---|---|---|
| **Backend** | Python · FastAPI · WebSocket | ML inference, voice I/O, intent parsing, frame streaming |
| **Frontend** | React · Vite · CSS | HUD display, real-time stats, log panel, quick commands |

The two layers talk exclusively over a **single persistent WebSocket connection**.

---

## Documentation Map

| Document | What it covers |
|---|---|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | System-level diagram, component map, technology choices |
| [BACKEND.md](./BACKEND.md) | FastAPI app structure, services, data models, gym-form integration |
| [FRONTEND.md](./FRONTEND.md) | React component tree, hooks, state machine, panel layout |
| [VOICE-PIPELINE.md](./VOICE-PIPELINE.md) | Wake-word detection, TTS, command dispatch flow |
| [DATA-FLOW.md](./DATA-FLOW.md) | WebSocket message protocol, event types, sequencing |

---

## Quick Start

See [LAUNCH-STACK.md](./LAUNCH-STACK.md) for how to run the dev stack.
