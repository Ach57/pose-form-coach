# gym-form — Real-Time Gym Form Detection

## Where Are We?

**Phase 0 (Data Foundation)** — COMPLETE  
**Phase 1 (Pose + Feature Extraction)** — COMPLETE  
**Phase 2 (Model + Training)** — NOT STARTED

All 2,260 OHP videos have been processed. The next step is implementing the Causal TCN model and training loop.

---

## Quick Start

```bash
cd gym-form
source .venv/bin/activate

# Re-run extraction (already done — 2260 videos cached)
python scripts/extract_all.py --config configs/dataset.ohp.yaml

# Validate splits (passed: no identity leakage)
python scripts/validate_splits.py --config configs/dataset.ohp.yaml

# Audit class balance (pos_weight already in train config)
python scripts/audit_data.py --config configs/dataset.ohp.yaml
```

---

## Repository Layout

```
gym-form/
├── configs/                    # YAML configs — single source of truth
│   ├── dataset.ohp.yaml        #   paths, labels, splits, windowing, sampling
│   ├── features.ohp.yaml       #   16 feature definitions (angles, velocities, scale)
│   ├── model.tcn.yaml          #   TCN architecture (channels, kernel, dilations)
│   ├── train.ohp.yaml          #   training hyperparams, pos_weight, augmentations
│   └── thresholds.ohp.yaml     #   hysteresis thresholds for real-time inference
│
├── data/                       # Generated data (gitignored)
│   ├── features/ohp/           #   2260 .npy files — (T, 16) feature matrices
│   ├── labels/ohp/             #   2260 .json files — per-video merged labels
│   └── poses/ohp/              #   2260 .npz files — (T, 33, 4) raw landmarks
│
├── src/                        # Python package — reusable classes
│   ├── extract/
│   │   └── pose_extractor.py   #   ✅ PoseExtractor (mediapipe tasks API)
│   ├── features/
│   │   └── feature_extractor.py#   ✅ OHPFeatureExtractor (16 features)
│   ├── datasets/
│   │   └── window_dataset.py   #   ✅ WindowDataset (sliding windows + sampling)
│   ├── models/
│   │   └── causal_tcn.py       #   ⬜ CausalTCN (stub — needs PyTorch impl)
│   ├── train/                  #   ⬜ Training loop (not started)
│   ├── eval/                   #   ⬜ Segment-mAP evaluation (not started)
│   ├── realtime/               #   ⬜ Live inference pipeline (not started)
│   └── utils/
│       └── io.py               #   ✅ YAML/JSON I/O, rasterize_segments/multilabel
│
├── scripts/                    # Executable scripts
│   ├── preprocess_labels.py    #   ✅ Convert per-error-type → per-video labels
│   ├── validate_splits.py      #   ✅ Check identity leakage + coverage
│   ├── audit_data.py           #   ✅ Compute class balance + pos_weight
│   └── extract_all.py          #   ✅ Batch pose + feature extraction
│
├── artifacts/                  # Models, checkpoints, metrics
│   ├── pose_landmarker_heavy.task   # MediaPipe model (offline, 29 MB)
│   ├── pose_landmarker_lite.task    # MediaPipe model (real-time, 5.6 MB)
│   ├── checkpoints/            #   ⬜ Saved model weights (empty)
│   ├── metrics/                #   ⬜ Training/eval logs (empty)
│   └── calibration/            #   ⬜ Threshold calibration data (empty)
│
├── tests/                      #   ⬜ Unit tests (not started)
├── notebooks/                  #   ⬜ EDA / visualization (not started)
├── deploy/                     #   ⬜ ONNX export, Docker (not started)
└── pyproject.toml              # Dependencies + tool config
```

`✅` = implemented `⬜` = stub / empty

---

## What Each File Does

### Configs (read these first)

| File                  | Purpose                                                                                                                                                                                       |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset.ohp.yaml`    | Maps the raw Fitness-AQA dataset to our pipeline. Defines source paths, label mapping (`error_elbows.json` → `ohp_elbow`), split paths, window size (T=64, stride=16), and sampling strategy. |
| `features.ohp.yaml`   | Enumerates all 16 features by name, type (angle/position/velocity/scale), and which landmarks they use. This is the canonical feature list.                                                   |
| `model.tcn.yaml`      | TCN architecture: 16 input features, 3 blocks of 64 channels, kernel=3, dilations=[1,2,4], 2 output labels.                                                                                   |
| `train.ohp.yaml`      | Training: batch=64, epochs=50, AdamW lr=3e-4, **pos_weight=[3.70, 3.28]** (computed from audit), early stopping on segment-mAP, augmentation params.                                          |
| `thresholds.ohp.yaml` | Hysteresis thresholds for real-time: ohp_elbow on=0.60/off=0.40, ohp_knee on=0.55/off=0.35.                                                                                                   |

### Source Modules

| Module                          | Class                         | Status  | What It Does                                                                                                                                                                      |
| ------------------------------- | ----------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `extract/pose_extractor.py`     | `PoseExtractor`               | ✅ Done | Wraps MediaPipe PoseLandmarker (tasks API v0.10+). `extract_video()` for offline batch, `process_frame()` for real-time. Caches to `.npz`. Interpolates short detection gaps.     |
| `features/feature_extractor.py` | `FeatureExtractor` (abstract) | ✅ Done | Base class: normalize landmarks (center on mid-hip, scale by shoulder width), Savitzky-Golay smoothing, `angle_between()` geometry helper, `finite_diff()` for velocities.        |
|                                 | `OHPFeatureExtractor`         | ✅ Done | 16 features: 5 joint angles (knees, elbows, trunk), 3 positions (wrist y, hip center y), 7 velocities, 1 scale (shoulder width).                                                  |
| `datasets/window_dataset.py`    | `WindowDataset`               | ✅ Done | Sliding windows from per-video features + labels. Positive-centered and hard-negative (boundary) windows. `get_sampler_weights()` for balanced sampling. `from_config()` factory. |
| `models/causal_tcn.py`          | `CausalTCN`                   | ⬜ Stub | Architecture defined but no PyTorch layers yet. Will be: Input(B,T,16) → 3 TemporalBlocks → per-frame logits(B,T,2).                                                              |
| `utils/io.py`                   | —                             | ✅ Done | `load_yaml()`, `load_json()`, `save_json()`, `rasterize_segments()` (time ranges → frame mask), `rasterize_multilabel()` (multi-label → frame matrix).                            |

### Scripts

| Script                 | What It Does                                                                                                        | Already Run? |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------- | :----------: |
| `preprocess_labels.py` | Reads `error_elbows.json` + `error_knees.json` from Fitness-AQA → writes 2260 per-video JSONs to `data/labels/ohp/` |      ✅      |
| `validate_splits.py`   | Checks: no duplicate IDs, no subject leakage, label coverage, no orphans. **All passed.**                           |      ✅      |
| `audit_data.py`        | Computes class balance: ohp_elbow 25.5% positive (pw=3.70), ohp_knee 34.4% positive (pw=3.28). Per-split breakdown. |      ✅      |
| `extract_all.py`       | Runs PoseExtractor → OHPFeatureExtractor on all videos. Writes `.npz` (poses) + `.npy` (features).                  |      ✅      |

---

## Key Numbers

| Metric                    | Value                          |
| ------------------------- | ------------------------------ |
| Total videos              | 2,260                          |
| Train / Val / Test        | 1,582 / 339 / 339              |
| Features per frame        | 16                             |
| Window size               | 64 frames (~2.1s at 30 FPS)    |
| Labels                    | `ohp_elbow`, `ohp_knee`        |
| ohp_elbow positive videos | 576 (25.5%), pos_weight = 3.70 |
| ohp_knee positive videos  | 777 (34.4%), pos_weight = 3.28 |
| Identity leakage          | None                           |
| Median segment duration   | 0.62s (elbow), 0.77s (knee)    |

---

## Pipeline Overview

```
Video (.mp4)
    │
    ▼
PoseExtractor (MediaPipe PoseLandmarker)
    │  → landmarks (T, 33, 4) cached as .npz
    ▼
OHPFeatureExtractor (angles, velocities, normalization)
    │  → features (T, 16) cached as .npy
    ▼
WindowDataset (sliding windows T=64, stride=16)
    │  → (B, T=64, F=16) features + (B, T=64, L=2) labels
    ▼
CausalTCN  ⬜ NOT YET
    │  → per-frame logits (B, T, 2)
    ▼
BCEWithLogitsLoss + pos_weight  ⬜ NOT YET
    │
    ▼
Hysteresis Post-Processing  ⬜ NOT YET
    │  → stable error segment predictions
    ▼
Segment-mAP Evaluation  ⬜ NOT YET
```

---

## What's Next (Phase 2)

1. **Implement CausalTCN** — PyTorch `nn.Module` with causal convolutions
2. **Training loop** — BCEWithLogitsLoss, WeightedRandomSampler, early stopping
3. **Segment-mAP evaluation** — temporal IoU at [0.1, 0.25, 0.5]
4. **Hysteresis post-processing** — smooth raw sigmoid outputs into stable segments
