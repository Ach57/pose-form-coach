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
