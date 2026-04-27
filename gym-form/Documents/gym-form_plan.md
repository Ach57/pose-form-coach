# Gym Form Real‑Time Detection — Plan & Architecture

A concise blueprint to re‑engineer your real‑time exercise form checker (MediaPipe @ 30 FPS) into a unified, multi‑exercise, multi‑label temporal model with stable, low‑latency inference.

---

## 1) Goals

- **Unify** separate models (OHP, Squat, Barbell Row) into one **multi‑label, multi‑exercise** temporal model.
- **Improve accuracy** via engineered kinematic features + temporal modeling (TCN).
- **Real‑time** feedback with **<100–150 ms** end‑to‑end latency and stable, flicker‑free signals (hysteresis).

---

## 2) Repository / Folder Structure

```text
gym-form/
├── data/
│   ├── raw_videos/                # (Optional) original videos
│   ├── poses/                     # pose caches: *.npz or *.parquet
│   └── labels/                    # per-video JSON with time segments
├── src/
│   ├── extract/                   # MediaPipe → pose cache (offline)
│   │   └── extract_pose.py
│   ├── features/                  # normalization, angles, velocities
│   │   └── ohp_features.py
│   ├── datasets/                  # windowing, pos/neg sampling
│   │   └── window_dataset.py
│   ├── models/                    # TCN backbone + multi-label head
│   │   └── unified_tcn.py
│   ├── train/                     # training loop, metrics, thresholds
│   │   ├── train.py
│   │   └── calibrate_thresholds.py
│   ├── eval/                      # segment mAP, PR-AUC, reports
│   │   └── eval_segments.py
│   ├── realtime/                  # webcam loop, hysteresis, overlay
│   │   └── app.py
│   └── utils/                     # IO, label rasterize, smoothing
│       └── io.py
├── configs/
│   ├── dataset.ohp.yaml           # paths, labels, window T/stride
│   ├── model.tcn.yaml             # channels, kernel, dilations
│   ├── train.ohp.yaml             # lr, loss weights, aug, epochs
│   └── thresholds.ohp.yaml        # per-label ON/OFF + hysteresis
├── scripts/                       # CLI wrappers (bash/py)
│   ├── run_extract.sh
│   ├── run_train.sh
│   └── run_realtime.sh
├── artifacts/
│   ├── checkpoints/               # model.ckpt / onnx
│   ├── metrics/                   # json/csv of results
│   └── calibration/               # chosen thresholds
├── deploy/
│   ├── onnx/                      # exported models
│   └── docker/                    # Dockerfile for runtime
├── tests/                         # unit tests for features/windowing
└── notebooks/
    └── 01_ohp_baseline.ipynb
```

---

## 3) Data Contracts

**Pose cache — `poses/{video_id}.npz`**

- `fps: float` (e.g., `30.0`)
- `exercise: str` (`"overhead_press" | "squat" | "barbell_row"`)
- `landmarks: (T, 33, 3 or 4)` → `(x, y, z)` **or** `(x, y, visibility[, z])`

**Labels — `labels/{video_id}.json`**

```json
{
  "ohp_knee": [
    [2.5, 3.93],
    [6.05, 6.93]
  ],
  "ohp_elbow": []
}
```

> Keep timestamps in **seconds**. During preprocessing, rasterize time ranges to **frame‑wise multi‑hot** labels at 30 FPS.

---

## 4) Feature Extraction (per frame)

- Normalize landmarks per frame: **center = mid‑hips**, **scale = shoulder width**; optional smoothing (Savitzky–Golay or One‑Euro filter).
- Kinematic features (strong baseline):
  - **Angles**: knees ∠(hip,knee,ankle), elbows ∠(shoulder,elbow,wrist), **trunk inclination** vs vertical
  - **Velocities**: first derivatives of angles; wrist vertical velocity
  - **Path proxy**: **wrist y** (bar path proxy for OHP/Row)
- Concatenate optional **exercise one‑hot** per frame for conditioning.

---

## 5) Model Architecture (Unified, Streaming‑Ready)

**Backbone: Causal Temporal Convolutional Network (TCN)**

- 3–4 temporal blocks, channels `[64, 64, 64]`, kernel `3`, dilations `[1, 2, 4, (8)]`
- Receptive field ≈ **64 frames** (~2.1 s @ 30 FPS). Use **causal** padding (no future frames).

**Heads:**

- **Error head (multi‑label, per frame)** → sigmoid outputs for labels, e.g., `ohp_knee`, `ohp_elbow` (extendable to Squat/Row labels)
- **(Optional) Exercise head (per sequence)** → auxiliary cross‑entropy loss; helps regularization

**Loss:**

- `BCEWithLogitsLoss` with per‑label `pos_weight` (class imbalance)
- Total: `L_error + λ * L_exercise` (e.g., λ = 0.3)

**Post‑processing:**

- Per‑label **hysteresis thresholds** (e.g., ON ≥ 0.6, OFF ≤ 0.4)
- Merge micro‑segments (<100 ms) & bridge short gaps (<150 ms)

---

## 6) Training & Evaluation Pipeline

**Flow**

1. **Extract** pose for each video (full length, 30 FPS) → write pose caches.
2. **Rasterize** label segments → frame‑wise multi‑hot labels (per error type).
3. **Features**: compute angles/velocities + normalization.
4. **Windowing**: build contiguous windows (e.g., **T = 64**, stride **= 16**). Oversample positives; include hard negatives near boundaries.
5. **Train**: AdamW, early stopping on **segment‑mAP** (tIoU 0.1/0.25/0.5).
6. **Calibrate** thresholds per label to maximize F1 / mAP on validation.
7. **Evaluate**: frame PR‑AUC; **segment‑mAP**; rep‑wise accuracy; stability (flip rate); latency.

**Splits**

- **By subject** (avoid identity leakage); maintain separate train/val/test.

**Augmentations**

- Time stretch ±20%, small feature noise, time masking/drops, optional left/right mirroring where semantics allow.

---

## 7) Real‑Time Inference (Webcam)

**Runtime loop**

1. **MediaPipe Pose** @ ~30 FPS → landmarks.
2. **Features** per frame → append to rolling buffer of length **T = 64**.
3. Every 2–3 frames, run **causal TCN** on the buffer → latest per‑label probs.
4. Apply **hysteresis** + segment cleanup (merge/bridge) → stable flags.
5. UI overlay with **explanations** (e.g., "Knee dip: knee angle ≈ 155°; target ≥ 170°").

**Latency target**

- Pose: 10–30 ms (CPU)
- Features + model: <5–10 ms
- E2E: **<100–150 ms**

---

## 8) Infrastructure & Deployment

**Environments**

- **Colab** for experiments & training.
- **Local dev** for realtime app and quick iterations.
- (Optional) **GPU box** for scale training.

**Packaging**

- Export to **ONNX**; run with **ONNX Runtime** (CPU/GPU) for low‑latency inference.
- **Docker** image with MediaPipe + ONNX Runtime for portable demos.

**Tracking & Artifacts**

- Use **TensorBoard** / **Weights & Biases** for metrics.
- Store in `artifacts/` (checkpoints, metrics, calibration).
- Consider **DVC** for data/model versioning (optional).

---

## 9) Milestones (OHP‑First)

1. **Setup** repo + configs + pose extractor.
2. **OHP Baseline**: features → TCN → thresholds → realtime demo.
3. **Hardening**: calibrate, add hysteresis, finalize metrics.
4. **Extend**: add Squat & Row labels → same pipeline (expand label space or add exercise‑conditioned heads).
5. **Deploy**: ONNX + Docker demo; gather feedback & iterate.

---

## 10) Config Stubs

**`configs/dataset.ohp.yaml`**

```yaml
paths:
  poses_dir: data/poses
  labels_dir: data/labels
labels:
  - ohp_knee
  - ohp_elbow
window:
  T: 64 # frames (~2.1 s)
  stride: 16 # frames
sampling:
  pos_center: true
  pos_frac: 0.5 # target fraction positives per batch
  hard_negatives: true
fps: 30.0
exercise: overhead_press
```

**`configs/model.tcn.yaml`**

```yaml
model:
  type: unified_tcn
  in_features: 16 # adjust to your feature count
  channels: [64, 64, 64]
  kernel: 3
  dilations: [1, 2, 4]
  dropout: 0.1
  n_labels: 2 # ohp_knee, ohp_elbow
  n_exercises: 3 # ohp, squat, row
```

**`configs/train.ohp.yaml`**

```yaml
train:
  batch_size: 64
  epochs: 50
  optimizer: adamw
  lr: 3e-4
  weight_decay: 1e-4
  grad_clip: 1.0
  pos_weight: [w_knee, w_elbow] # fill from class stats
  aux_exercise_loss: 0.3 # lambda
  early_stop_patience: 8
augs:
  time_stretch: 0.2
  feature_jitter: 0.02
  time_mask_prob: 0.1
val:
  metric: segment_map
  tiou: [0.1, 0.25, 0.5]
```

**`configs/thresholds.ohp.yaml`**

```yaml
thresholds:
  ohp_knee:
    on: 0.60
    off: 0.40
    min_dur_ms: 100
    bridge_gap_ms: 150
  ohp_elbow:
    on: 0.55
    off: 0.35
    min_dur_ms: 100
    bridge_gap_ms: 150
```

---

## 11) Notes & Best Practices

- Extract **full‑video** pose once; cache it. Storage is tiny and gives you negatives + context.
- Build datasets from **contiguous windows** (e.g., 64 frames). Shuffle **windows** across videos, not frames inside windows.
- Use **subject‑wise splits** to avoid identity leakage.
- Calibrate **per‑label thresholds** on a validation set; apply **hysteresis** for real‑time stability.
- Prefer **MediaPipe world landmarks** (metric 3D) if available; otherwise use normalized 2D and be consistent train↔infer.

---
