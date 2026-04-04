# gym-form — Real-Time Gym Form Detection

A full pipeline that detects Overhead Press (OHP) form errors in real time using MediaPipe pose estimation and a Causal Temporal Convolutional Network (Causal TCN).

## Project Status

| Phase | Name | Status |
|---|---|---|
| 0 | Data Foundation | ✅ Complete |
| 1 | Pose + Feature Extraction | ✅ Complete |
| 2 | Model + Training | ✅ Complete |
| 3 | Real-Time Inference | ✅ Complete |

**109 tests passing.** All 2,260 OHP videos processed and cached.

---

## Quick Start

```bash
cd gym-form
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Train (Colab recommended for GPU)

Open `notebooks/train_ohp.ipynb` in Google Colab. It handles:
- GPU detection and setup
- Config loading
- Model build (`CausalTCN`)
- Training with early stopping
- Checkpoint export to Google Drive

### Run live inference (after training)

```bash
python scripts/infer_live.py --checkpoint checkpoints/best.pt
```

Optional flags:

```bash
python scripts/infer_live.py \
  --checkpoint checkpoints/best.pt \
  --model-path artifacts/pose_landmarker_heavy.task \
  --device mps \        # cpu | cuda | mps
  --threshold 0.4 \
  --camera 0
```

Press `q` or `Esc` to quit.

### Re-run data pipeline (already cached)

```bash
# Batch pose + feature extraction (2260 videos → .npz + .npy)
python scripts/extract_all.py --config configs/dataset.ohp.yaml

# Validate splits (no identity leakage)
python scripts/validate_splits.py --config configs/dataset.ohp.yaml

# Audit class balance
python scripts/audit_data.py --config configs/dataset.ohp.yaml
```

### Run tests

```bash
pytest tests/ -v
```

---

## Repository Layout

```
gym-form/
├── configs/                         # YAML configs — single source of truth
│   ├── dataset.ohp.yaml             #   paths, splits, windowing, sampling
│   ├── features.ohp.yaml            #   16 feature definitions
│   ├── model.tcn.yaml               #   TCN architecture
│   ├── train.ohp.yaml               #   hyperparams, pos_weight, augmentations
│   └── thresholds.ohp.yaml          #   hysteresis thresholds for inference
│
├── data/                            # Generated data (gitignored)
│   ├── features/ohp/                #   2260 .npy files — (T, 16) feature matrices
│   ├── labels/ohp/                  #   2260 .json files — per-video merged labels
│   └── poses/ohp/                   #   2260 .npz files — (T, 33, 4) raw landmarks
│
├── src/                             # Python package
│   ├── extract/
│   │   └── pose_extractor.py        #   ✅ PoseExtractor (offline + real-time)
│   ├── features/
│   │   └── feature_extractor.py     #   ✅ OHPFeatureExtractor (16 features)
│   ├── datasets/
│   │   └── window_dataset.py        #   ✅ WindowDataset (sliding windows)
│   ├── models/
│   │   └── causal_tcn.py            #   ✅ CausalTCN (PyTorch nn.Module)
│   ├── train/
│   │   └── trainer.py               #   ✅ Trainer (BCELoss, early stopping)
│   ├── eval/
│   │   └── metrics.py               #   ✅ segment-mAP, hysteresis_segments
│   ├── realtime/
│   │   └── predictor.py             #   ✅ OHPPredictor (rolling buffer + live predict)
│   └── utils/
│       ├── io.py                     #   ✅ YAML/JSON I/O, rasterize helpers
│       ├── mediapipe_visualization.py#   ✅ 3D skeleton + feature timeline (Plotly)
│       ├── checkpoints.py            #   ✅ Checkpoint save/copy to Drive
│       ├── hardware.py               #   ✅ Device detection (CPU/CUDA/MPS)
│       └── runtime.py               #   ✅ Colab detection helpers
│
├── scripts/                         # Executable entry points
│   ├── preprocess_labels.py         #   ✅ Convert Fitness-AQA → per-video JSONs
│   ├── validate_splits.py           #   ✅ Check identity leakage + coverage
│   ├── audit_data.py                #   ✅ Compute class balance + pos_weight
│   ├── extract_all.py               #   ✅ Batch pose + feature extraction
│   └── infer_live.py                #   ✅ Webcam inference with OpenCV overlay
│
├── notebooks/
│   └── train_ohp.ipynb              #   ✅ Colab training notebook (GPU)
│
├── tests/
│   └── unit/                        #   ✅ 109 tests across all modules
│
├── artifacts/
│   ├── pose_landmarker_heavy.task   # MediaPipe model file (29 MB, offline)
│   └── pose_landmarker_lite.task    # MediaPipe model file (5.6 MB, real-time)
│
└── pyproject.toml                   # Dependencies + tool config
```

`✅` = implemented and tested

---

## Architecture

### Causal TCN

The model uses a Causal Temporal Convolutional Network — causal convolutions mean no future frames are ever looked at, making it suitable for real-time use.

```
Input  (B, T, 16)
   │
   ▼
TemporalBlock  dilation=1   →  64 channels
TemporalBlock  dilation=2   →  64 channels
TemporalBlock  dilation=4   →  64 channels
   │
   ▼
Conv1d head  →  (B, T, 2) logits
   │
   ▼
sigmoid  →  (B, T, 2) probabilities
```

| Property | Value |
|---|---|
| Input features | 16 per frame |
| Channels | [64, 64, 64] |
| Kernel size | 3 |
| Dilations | [1, 2, 4] |
| Receptive field | 29 frames (~1 s at 30 fps) |
| Parameters | ~67 K |
| Output labels | `ohp_elbow`, `ohp_knee` |

### Features (16 per frame)

| # | Name | Type |
|---|---|---|
| 0 | `knee_angle_L` | angle (rad) |
| 1 | `knee_angle_R` | angle (rad) |
| 2 | `elbow_angle_L` | angle (rad) |
| 3 | `elbow_angle_R` | angle (rad) |
| 4 | `trunk_inclination` | angle (rad) |
| 5 | `wrist_y_L` | position (normalized) |
| 6 | `wrist_y_R` | position (normalized) |
| 7 | `hip_center_y` | position (normalized) |
| 8–14 | `d_*` | first derivative (velocity) of cols 0–6 |
| 15 | `shoulder_width` | scale reference |

Landmarks are normalized per frame: centered on mid-hip, scaled by shoulder width.

---

## Real-Time Inference Pipeline

```
Webcam frame  (BGR, H×W×3)
      │
      ▼
PoseExtractor.process_frame()
      │  → (33, 4)  [x, y, z, visibility]
      ▼
Rolling buffer  (last 29 frames)
      │  → (29, 33, 4)
      ▼
OHPFeatureExtractor.extract()
      │  → (29, 16)  float32
      ▼
CausalTCN  (loaded from best.pt)
      │  → (1, 29, 2) logits  →  sigmoid  →  probs
      ▼
Take last frame probs  →  [p_elbow, p_knee]
      │
      ▼
OpenCV overlay: probability bars + status banner
```

`OHPPredictor` (in `src/realtime/predictor.py`) encapsulates everything. The script in `scripts/infer_live.py` is a thin camera loop on top of it.

---

## Training

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Batch size | 64 |
| Max epochs | 50 |
| Early stopping | patience=10, monitor=segment-mAP |
| Loss | BCEWithLogitsLoss with pos_weight |
| Sampler | WeightedRandomSampler (balanced) |
| Window size | 64 frames |
| Window stride | 16 frames |
| pos_weight (elbow) | 3.70 |
| pos_weight (knee) | 3.28 |

---

## Key Numbers

| Metric | Value |
|---|---|
| Total videos | 2,260 |
| Train / Val / Test | 1,582 / 339 / 339 |
| Features per frame | 16 |
| Labels | `ohp_elbow`, `ohp_knee` |
| ohp_elbow positive | 576 videos (25.5%) |
| ohp_knee positive | 777 videos (34.4%) |
| Identity leakage | None |
| Median segment duration | 0.62 s (elbow), 0.77 s (knee) |
| Unit tests | 109 passing |

---

## Source Module Reference

### `src/extract/pose_extractor.py` — `PoseExtractor`

Wraps the MediaPipe PoseLandmarker (tasks API v0.10+).

| Method | Description |
|---|---|
| `extract_video(path)` | Offline — processes every frame, returns `PoseResult` with `(T, 33, 4)` landmarks. Interpolates short detection gaps (≤3 frames). |
| `open()` | Opens a persistent IMAGE-mode session for real-time use. |
| `process_frame(frame)` | Single BGR frame → `(33, 4)` or `None` if detection fails. |
| `close()` | Closes the MediaPipe session. |

### `src/features/feature_extractor.py` — `OHPFeatureExtractor`

| Method | Description |
|---|---|
| `extract(landmarks)` | `(T, 33, 4)` → `(T, 16)` float32. Normalizes, smooths (Savitzky-Golay), computes angles, velocities, and scale. |
| `feature_names` | Ordered list of 16 feature name strings. |

### `src/models/causal_tcn.py` — `CausalTCN`

| Method | Description |
|---|---|
| `forward(x)` | `(B, T, F)` → `(B, T, L)` logits (pre-sigmoid). |
| `receptive_field` | Total frames the model can see (29 with default config). |
| `from_config(cfg)` | Factory: builds model from a config dict. |

### `src/train/trainer.py` — `Trainer`

| Method | Description |
|---|---|
| `fit(train_loader, val_loader)` | Full training loop with early stopping. Saves `best.pt` and `last.pt`. |
| `evaluate(loader)` | Returns segment-mAP on val/test split. |
| `load_checkpoint(path)` | Restores model + optimizer state. |

### `src/eval/metrics.py`

| Function | Description |
|---|---|
| `hysteresis_segments(probs, on, off)` | Converts frame-level probabilities to stable segments using a two-threshold hysteresis. |
| `segment_ap(pred_segs, gt_segs, iou_thresh)` | Average precision for a single label at a given IoU threshold. |
| `segment_map(pred_segs, gt_segs, iou_thresholds)` | Mean AP across IoU thresholds [0.1, 0.25, 0.5]. |

### `src/realtime/predictor.py` — `OHPPredictor`

| Method | Description |
|---|---|
| `from_checkpoint(path, ...)` | Factory: loads `best.pt`/`last.pt`, builds model, returns ready-to-use predictor. |
| `open()` / `close()` | Start/stop MediaPipe session. Supports `with` statement. |
| `predict(frame)` | BGR frame → `np.ndarray (2,)` probs or `None` during warm-up. |
| `label_errors(probs)` | Returns list of active error label strings above threshold. |
| `is_warm` | `True` once the 29-frame rolling buffer has filled. |
