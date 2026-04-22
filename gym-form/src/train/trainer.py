"""Training loop for CausalTCN form-error detection.

Handles:
- Dataset construction from configs
- BCEWithLogitsLoss with pos_weight
- WeightedRandomSampler for class balance
- Data augmentation (jitter, time-stretch, time-mask)
- Validation with segment-mAP early stopping
- Checkpoint save/load
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.datasets.window_dataset import WindowDataset
from src.eval.metrics import frame_metrics, hysteresis_segments, segment_map
from src.models.causal_tcn import CausalTCN
from src.utils.io import load_json, load_yaml, rasterize_multilabel


# ── Augmentations ────────────────────────────────────────────────────


class WindowAugmentor:
    """Simple numpy augmentations applied to (T, F) feature windows."""

    def __init__(
        self,
        feature_jitter: float = 0.0,
        time_mask_prob: float = 0.0,
        time_mask_max: int = 10,
    ) -> None:
        self.feature_jitter = feature_jitter
        self.time_mask_prob = time_mask_prob
        self.time_mask_max = time_mask_max

    def __call__(
        self, features: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.feature_jitter > 0:
            features = features + np.random.randn(*features.shape).astype(
                np.float32
            ) * self.feature_jitter

        if self.time_mask_prob > 0 and np.random.rand() < self.time_mask_prob:
            T = features.shape[0]
            mask_len = np.random.randint(1, min(self.time_mask_max, T) + 1)
            start = np.random.randint(0, T - mask_len + 1)
            features[start : start + mask_len] = 0.0

        return features, labels


# ── Collate with augmentation ────────────────────────────────────────


def _make_collate(augmentor: WindowAugmentor | None = None):
    """Return a collate function that optionally applies augmentations."""

    def collate_fn(batch):
        feats_list, labels_list = [], []
        for feats, labels in batch:
            if augmentor is not None:
                feats, labels = augmentor(feats.copy(), labels.copy())
            feats_list.append(torch.from_numpy(feats))
            labels_list.append(torch.from_numpy(labels))
        return torch.stack(feats_list), torch.stack(labels_list)

    return collate_fn


# ── Trainer ──────────────────────────────────────────────────────────


class Trainer:
    """Trains a CausalTCN on windowed features with early stopping.

    Parameters
    ----------
    model : CausalTCN
    train_cfg : dict — the 'train' section of train.ohp.yaml
    dataset_cfg : dict — full dataset.ohp.yaml
    aug_cfg : dict | None — the 'augmentations' section
    val_cfg : dict | None — the 'validation' section
    device : str — 'cpu', 'cuda', or 'mps'
    checkpoint_dir : Path — where to save checkpoints
    """

    def __init__(
        self,
        model: CausalTCN,
        train_cfg: dict,
        dataset_cfg: dict,
        aug_cfg: dict | None = None,
        val_cfg: dict | None = None,
        device: str = "cpu",
        checkpoint_dir: Path | str = "checkpoints",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_cfg = train_cfg
        self.dataset_cfg = dataset_cfg
        self.val_cfg = val_cfg or {}
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss
        pos_weight = torch.tensor(
            train_cfg.get("pos_weight", [1.0, 1.0]), dtype=torch.float32
        ).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg.get("lr", 3e-4),
            weight_decay=train_cfg.get("weight_decay", 1e-4),
        )

        # Augmentor
        self.augmentor = None
        if aug_cfg:
            self.augmentor = WindowAugmentor(
                feature_jitter=aug_cfg.get("feature_jitter", 0.0),
                time_mask_prob=aug_cfg.get("time_mask_prob", 0.0),
            )

        # Early stopping
        es = train_cfg.get("early_stop", {})
        self.patience = es.get("patience", 8)
        self.best_metric = -float("inf")
        self.wait = 0
        self.best_epoch = 0

    def _build_loader(
        self, split: str, shuffle: bool = True
    ) -> DataLoader:
        """Build a DataLoader for the given split."""
        ds = WindowDataset.from_config(
            self.dataset_cfg,
            split=split,
            features_dir=Path(self.dataset_cfg["paths"].get("features_dir", "data/features/ohp")),
            labels_dir=Path(self.dataset_cfg["paths"]["labels_dir"]),
        )

        sampler = None
        if shuffle and len(ds) > 0:
            weights = ds.get_sampler_weights()
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(weights).double(),
                num_samples=len(ds),
                replacement=True,
            )

        return DataLoader(
            ds,
            batch_size=self.train_cfg.get("batch_size", 64),
            sampler=sampler,
            shuffle=False,  # sampler handles shuffling
            collate_fn=_make_collate(self.augmentor if shuffle else None),
            num_workers=0,
            pin_memory=(self.device.type != "cpu"),
        )

    def _train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch, return mean loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        grad_clip = self.train_cfg.get("grad_clip", 0.0)

        for feats, labels in loader:
            feats = feats.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(feats)
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict[str, float]:
        """Run validation, return loss + frame metrics + segment-mAP."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_probs: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for feats, labels in loader:
            feats = feats.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(feats)
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

        results: dict[str, float] = {
            "val_loss": total_loss / max(n_batches, 1)
        }

        if not all_probs:
            return results

        probs_cat = np.concatenate(all_probs, axis=0)  # (N, T, L)
        labels_cat = np.concatenate(all_labels, axis=0)

        # Frame-level metrics (averaged across labels)
        fm = frame_metrics(probs_cat, labels_cat)
        results.update({f"frame_{k}": v for k, v in fm.items()})

        # Segment-mAP per label
        tiou = self.val_cfg.get("tiou", [0.1, 0.25, 0.5])
        n_labels = probs_cat.shape[-1]
        label_maps: list[float] = []

        for label_idx in range(n_labels):
            all_preds_segs: list[list[tuple[int, int]]] = []
            all_scores: list[list[float]] = []
            all_gt_segs: list[list[tuple[int, int]]] = []

            for i in range(probs_cat.shape[0]):
                p = probs_cat[i, :, label_idx]
                g = labels_cat[i, :, label_idx]

                pred_segs = hysteresis_segments(p, on=0.5, off=0.3, min_dur=3)
                pred_scores_list = [float(p[s:e + 1].mean()) for s, e in pred_segs]

                gt_segs = _binary_to_segments(g)

                all_preds_segs.append(pred_segs)
                all_scores.append(pred_scores_list)
                all_gt_segs.append(gt_segs)

            smap = segment_map(all_preds_segs, all_scores, all_gt_segs, tiou)
            label_maps.append(smap["mAP"])
            for k, v in smap.items():
                results[f"label{label_idx}_{k}"] = v

        results["segment_map"] = float(np.mean(label_maps))
        return results

    def fit(self, epochs: int | None = None) -> dict[str, list]:
        """Full training loop with early stopping.

        Returns
        -------
        history : dict with keys 'train_loss', 'val_loss', 'segment_map', etc.
        """
        epochs = epochs or self.train_cfg.get("epochs", 50)
        train_loader = self._build_loader("train", shuffle=True)
        val_loader = self._build_loader("val", shuffle=False)

        history: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "segment_map": [],
        }

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_results = self._validate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_results["val_loss"])
            seg_map = val_results.get("segment_map", 0.0)
            history["segment_map"].append(seg_map)

            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_results['val_loss']:.4f} | "
                f"seg-mAP={seg_map:.4f}"
            )

            # Early stopping
            if seg_map > self.best_metric:
                self.best_metric = seg_map
                self.best_epoch = epoch
                self.wait = 0
                self.save_checkpoint("best.pt")
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(best={self.best_metric:.4f} at epoch {self.best_epoch})"
                    )
                    break

        self.save_checkpoint("last.pt")
        return history

    def save_checkpoint(self, filename: str) -> Path:
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_metric": self.best_metric,
                "best_epoch": self.best_epoch,
            },
            path,
        )
        return path

    def load_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_metric = ckpt.get("best_metric", -float("inf"))
        self.best_epoch = ckpt.get("best_epoch", 0)


# ── Helpers ──────────────────────────────────────────────────────────


def _binary_to_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Convert a binary (T,) mask to a list of (start, end) inclusive segments."""
    segments = []
    in_seg = False
    start = 0
    for t in range(len(mask)):
        if mask[t] > 0.5 and not in_seg:
            in_seg = True
            start = t
        elif mask[t] <= 0.5 and in_seg:
            in_seg = False
            segments.append((start, t - 1))
    if in_seg:
        segments.append((start, len(mask) - 1))
    return segments
