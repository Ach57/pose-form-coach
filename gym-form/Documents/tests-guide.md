# Test Suite — Navigation Guide

**81 tests** | pytest | `tests/unit/` (74) · `tests/integration/` (7)

```
.venv/bin/pytest tests/ -v          # run all
.venv/bin/pytest tests/unit/ -v     # unit only
.venv/bin/pytest tests/integration/ # integration only
```

---

## Unit Tests (74 tests)

### `test_io.py` — Label rasterization utilities
Source: `src/utils/io.py`

| Class | Test | What it verifies |
|-------|------|------------------|
| **TestRasterizeSegments** | `test_empty_segments_all_false` | Empty segment list → all-False mask |
| | `test_single_segment` | One `[start, end]` → correct frames set to True |
| | `test_multiple_segments_no_overlap` | Two non-overlapping segments rasterized correctly |
| | `test_segment_clipped_to_n_frames` | Segment extending past video end is clipped |
| | `test_segment_before_video_start` | Negative start time doesn't crash |
| | `test_zero_length_segment` | `[t, t]` zero-length segment handled |
| | `test_frame_count_matches` | Output length equals `n_frames` |
| **TestRasterizeMultilabel** | `test_shape_and_dtype` | Output shape `(T, L)`, dtype `float32` |
| | `test_column_order_matches_label_order` | Columns follow `label_names` ordering |
| | `test_missing_label_key_is_all_zero` | Label not in JSON → all-zero column |
| | `test_multi_label_overlap` | Two labels active on same frames both rasterized |

---

### `test_feature_extractor.py` — Feature extraction pipeline
Source: `src/features/feature_extractor.py`

| Class | Test | What it verifies |
|-------|------|------------------|
| **TestAngleBetween** | `test_right_angle` | 90° → π/2 |
| | `test_straight_line` | 180° → π (tolerance 1e-3 due to ε in denominator) |
| | `test_zero_angle` | Same direction → 0 |
| | `test_batch_of_frames` | Vectorized over T frames |
| **TestNormalizeLandmarks** | `test_hip_center_is_origin` | Mid-hip centered to (0,0,0) |
| | `test_visibility_preserved` | 4th channel (visibility) unchanged |
| | `test_shoulder_width_positive` | Shoulder width > 0 for all frames |
| **TestFiniteDiff** | `test_constant_signal_zero_velocity` | Constant input → zero derivative |
| | `test_linear_signal_constant_velocity` | Linear input → constant derivative |
| | `test_shape_preserved` | Output shape matches input |
| **TestOHPFeatureExtractor** | `test_output_shape` | `(T, 16)` output from `(T, 33, 4)` landmarks |
| | `test_feature_count_matches_names` | 16 features = len(feature_names) |
| | `test_feature_names_order` | Feature name list matches expected order |
| | `test_straight_limb_gives_pi` | Collinear joints → angle ≈ π |
| | `test_shoulder_width_positive` | Scale feature > 0 |
| | `test_static_landmarks_zero_velocity` | No movement → zero velocity features |
| | `test_smoothing_does_not_change_shape` | Savitzky-Golay preserves dimensions |
| | `test_two_frame_no_crash` | Minimum viable input (np.gradient needs ≥2) |

---

### `test_window_dataset.py` — Sliding-window dataset
Source: `src/datasets/window_dataset.py`

| Class | Test | What it verifies |
|-------|------|------------------|
| **TestWindowDatasetBasics** | `test_uniform_stride_window_count` | 128 frames, T=64, stride=16 → 5 windows |
| | `test_window_shape` | Each window is `(64, 16)` features + `(64, 2)` labels |
| | `test_short_video_padded` | Video < T gets edge-padded, not skipped |
| | `test_missing_files_skipped` | Missing .npy/.json → 0 windows, no crash |
| **TestWindowDatasetPositiveSampling** | `test_pos_center_creates_extra_windows` | `pos_center=True` adds windows vs stride-only |
| | `test_positive_windows_contain_positive_frames` | Every positive-indexed window has ≥1 positive frame |
| | `test_pos_neg_indices_cover_all_windows` | `_pos_indices ∪ _neg_indices = range(len)` |
| **TestSamplerWeights** | `test_weights_length` | Weights array length == dataset length |
| | `test_weights_all_positive` | All weights > 0 |
| | `test_empty_dataset_empty_weights` | Empty dataset → empty array |

---

### `test_causal_tcn.py` — CausalTCN model
Source: `src/models/causal_tcn.py`

| Class | Test | What it verifies |
|-------|------|------------------|
| **TestCausalConv1d** | `test_output_length_matches_input` | Causal padding preserves temporal dimension |
| | `test_causality` | Changing future input doesn't affect past output |
| **TestTemporalBlock** | `test_shape_preserved` | `(B, C_in, T)` → `(B, C_out, T)` |
| | `test_residual_same_channels` | Identity residual when `C_in == C_out` |
| **TestCausalTCN** | `test_output_shape` | `(4, 64, 16)` → `(4, 64, 2)` |
| | `test_causality_full_model` | Full model is causal (zeroing future doesn't change past) |
| | `test_receptive_field` | RF = 29 for `dilations=[1,2,4]`, `k=3` |
| | `test_variable_length_input` | Works with T=32, 100, 200 |
| | `test_from_config` | `from_config("configs/model.tcn.yaml")` loads correctly |
| | `test_backward_pass` | All parameters receive gradients |

---

### `test_metrics.py` — Evaluation metrics
Source: `src/eval/metrics.py`

| Class | Test | What it verifies |
|-------|------|------------------|
| **TestTemporalIoU** | `test_perfect_overlap` | Same segment → IoU = 1.0 |
| | `test_no_overlap` | Disjoint segments → IoU = 0.0 |
| | `test_partial_overlap` | Correct intersection/union ratio |
| | `test_contained` | One segment inside another |
| **TestHysteresisSegments** | `test_basic` | Rising above `on`, falling below `off` |
| | `test_empty_when_below_threshold` | Never crosses `on` → no segments |
| | `test_min_dur_filters_short` | Short blip filtered by `min_dur` |
| | `test_bridge_gap_merges` | Close segments merged by `bridge_gap` |
| | `test_segment_at_end` | Active segment at array end closed properly |
| **TestFrameMetrics** | `test_perfect_prediction` | P=R=F1 ≈ 1.0 |
| | `test_all_wrong` | All false positives → P ≈ 0 |
| **TestSegmentAP** | `test_perfect_match` | Perfect prediction → AP = 1.0 |
| | `test_no_predictions_with_gt` | Missing predictions → AP = 0.0 |
| | `test_no_gt_no_preds` | No GT, no preds → AP = 1.0 |
| | `test_no_gt_with_preds` | False positives only → AP = 0.0 |
| **TestSegmentMAP** | `test_basic` | Single-video mAP aggregation |

---

### `test_trainer.py` — Training loop components
Source: `src/train/trainer.py`

| Class | Test | What it verifies |
|-------|------|------------------|
| **TestWindowAugmentor** | `test_jitter_changes_features` | Gaussian noise modifies features |
| | `test_no_aug_identity` | Zero config → features unchanged |
| | `test_time_mask_zeros_some_frames` | Time masking zeros a contiguous span |
| **TestBinaryToSegments** | `test_single_segment` | `[0,0,1,1,1,0,0]` → `[(2,4)]` |
| | `test_multiple_segments` | Two runs detected |
| | `test_empty` | All zeros → empty list |
| | `test_all_positive` | All ones → `[(0, T-1)]` |
| **TestTrainerInit** | `test_creates_with_defaults` | Trainer initializes, creates checkpoint dir |
| | `test_save_and_load_checkpoint` | Round-trip: save → load → weights match |

---

## Integration Tests (7 tests)

### `test_feature_to_dataset.py` — End-to-end data pipeline
Sources: `src/datasets/window_dataset.py` + real data from `data/features/ohp/` and `data/labels/ohp/`

Uses video **62805_6** (known to have positive elbow error segments).
Skipped automatically if extracted data is not present.

| Class | Test | What it verifies |
|-------|------|------------------|
| **TestFeatureThroughDataset** | `test_dataset_not_empty` | Real video produces ≥1 window |
| | `test_feature_window_shape` | Windows are `(64, 16)` features + `(64, 2)` labels |
| | `test_features_are_finite` | No NaN/Inf in any window |
| | `test_labels_are_binary` | Labels contain only 0.0 and 1.0 |
| | `test_has_positive_windows` | Video with known errors → positive windows exist |
| | `test_positive_windows_actually_contain_positive_frames` | Positive-indexed windows really have positive frames |
| | `test_sampler_weights_match_length` | Sampler weights cover all windows |
