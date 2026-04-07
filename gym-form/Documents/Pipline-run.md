NOTE:
To run the full pipeline for squats:

```bash
# 1. Preprocess labels
python scripts/preprocess_labels.py --config configs/dataset.squat.yaml

# 2. Extract poses + features
python scripts/extract_all.py --config configs/dataset.squat.yaml

# 3. Validate splits
python scripts/validate_splits.py --config configs/dataset.squat.yaml
```

```bash
# Use MPS on Apple Silicon, lower threshold
python scripts/ohp_infer_live.py \
  --checkpoint checkpoints/best.pt \
  --device mps \
  --threshold 0.4 \
  --camera 0
```

```bash
python scripts/squat_infer_live.py \
  --checkpoint checkpoints/best_squat.pt \
  --device mps \
  --threshold 0.5 \
  --camera 0
```
