# Executive Summary — Real‑Time Gym Form Assistant

**One‑sentence summary:** A camera‑based assistant that spots common form errors during strength exercises (starting with the Overhead Press) and gives instant, friendly feedback — like having a coach by your side, 24/7.

---

## Why this matters

- **Safer, smarter training:** Reduces injury risk by catching form mistakes early.
- **Coaching at scale:** Delivers consistent guidance without needing a trainer present for every session.
- **Motivating feedback:** Clear cues (“soft knee dip”, “elbow path off”) help users improve rep by rep.

---

## What it does (in plain language)

- You start the app and face the camera.
- The assistant “reads” your body position (no special sensors needed).
- As you lift, it highlights form issues in real time and suggests quick fixes.

_Initial focus:_ **Overhead Press** with two error types — **knee dip** and **elbow path**. The same foundation extends to **Squat** and **Barbell Row** next.

---

## Who it’s for

- **Everyday lifters** who want safer, cleaner technique.
- **Coaches/gyms** to support members between sessions.
- **Content creators/rehab** for quick checks when filming technique.

---

## What’s required to launch (practical, non‑technical)

**People & time**

- 1 project lead (coordination) + 1 builder (model/app) + light support from a coach for sign‑off on cues.
- 4–6 weeks for a polished MVP of the Overhead Press assistant.

**Data**

- A set of short exercise videos already labeled where form is **correct** or **has a mistake** (we have this for OHP, Squat, Row).
- No faces or personal info needed in outputs.

**Equipment**

- A laptop with a webcam (or a phone, in a later phase).

**Software**

- Pose‑tracking and a lightweight model packaged into a simple desktop app. No cloud dependency required.

---

## How it works (at a glance)

1. **See**: The camera captures your movement.
2. **Understand**: The system turns body points into simple motion signals.
3. **Coach**: It recognizes known mistake patterns and gives clear, on‑screen tips.

Everything runs locally for fast response and privacy.

---

## What we will deliver in the first release (MVP)

- **Exercise**: Overhead Press.
- **Detections**: “Knee dip” and “Elbow path off”.
- **Experience**: Live view with green/amber indicators and short text tips.
- **Performance targets**:
  - Accuracy you can trust (aiming ≥ **85%** correct detections on a held‑out test set).
  - Smooth, low‑lag feedback (**<150 ms** end‑to‑end on a laptop).

---

## What’s not included (yet)

- Barbell object tracking, full workout logging, and advanced analytics.
- Mobile app distribution and multi‑person scenes.
- Medical/physio advice (this is a technique assistant, not a diagnosis tool).

---

## How we’ll measure success

- **Detection quality**: High precision/recall on form errors; minimal false alarms.
- **User experience**: Feedback feels timely (no flicker), helpful, and not distracting.
- **Adoption**: Users choose to keep it on during sets; positive qualitative feedback.

---

## Timeline (indicative)

- **Weeks 0–1**: Confirm scope, prepare sample videos, align on feedback wording.
- **Weeks 2–3**: Build & test the Overhead Press model; tune thresholds for reliable signals.
- **Weeks 4–5**: Real‑time app polish; pilot with a small group and gather feedback.
- **Week 6**: MVP sign‑off; plan expansion to Squat and Barbell Row.

---

## Risks & how we’ll handle them

- **Varied camera angles/lighting** → Use a diverse set of videos; keep guidance generalized.
- **False positives** → Calibrate thresholds; add gentle hysteresis (turn on at a higher score than it turns off) to avoid flicker.
- **User privacy** → Process on device; store only anonymized motion points if any data is saved.

---

## Budget & tooling

- Uses existing laptops and webcams.
- Built on reliable, widely used open components.
- No paid cloud services are required for MVP.

---

## Decisions we need upfront

1. **MVP scope**: Start with Overhead Press (knee + elbow) — confirm.
2. **Pilot group**: Identify 3–5 users to try the MVP and give feedback.
3. **Feedback style**: Approve the tone and wording of on‑screen cues (short, supportive).

---

## Next steps

- Green‑light the MVP scope and pilot group.
- Provide the initial labeled videos for calibration.
- We deliver a first demo build in **~3 weeks** with live OHP feedback.

---

**Contact:** Achraf — Project Lead, Real‑Time Gym Form Assistant
