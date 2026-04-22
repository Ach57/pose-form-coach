"""Unit tests for src.train.trainer — augmentations and trainer init."""

from __future__ import annotations

import numpy as np
import torch
import pytest

from src.train.trainer import WindowAugmentor, Trainer, _binary_to_segments
from src.models.causal_tcn import CausalTCN


class TestWindowAugmentor:

    def test_jitter_changes_features(self):
        aug = WindowAugmentor(feature_jitter=0.1)
        feats = np.ones((64, 16), dtype=np.float32)
        labels = np.zeros((64, 2), dtype=np.float32)
        out_feats, out_labels = aug(feats.copy(), labels.copy())
        assert not np.array_equal(out_feats, feats)
        assert np.array_equal(out_labels, labels)

    def test_no_aug_identity(self):
        aug = WindowAugmentor()
        feats = np.ones((64, 16), dtype=np.float32)
        labels = np.zeros((64, 2), dtype=np.float32)
        out_feats, _ = aug(feats.copy(), labels.copy())
        assert np.array_equal(out_feats, feats)

    def test_time_mask_zeros_some_frames(self):
        np.random.seed(42)
        aug = WindowAugmentor(time_mask_prob=1.0, time_mask_max=10)
        feats = np.ones((64, 16), dtype=np.float32)
        labels = np.zeros((64, 2), dtype=np.float32)
        out_feats, _ = aug(feats.copy(), labels.copy())
        assert (out_feats == 0).any()


class TestBinaryToSegments:

    def test_single_segment(self):
        mask = np.array([0, 0, 1, 1, 1, 0, 0])
        segs = _binary_to_segments(mask)
        assert segs == [(2, 4)]

    def test_multiple_segments(self):
        mask = np.array([1, 1, 0, 0, 1, 1])
        segs = _binary_to_segments(mask)
        assert segs == [(0, 1), (4, 5)]

    def test_empty(self):
        mask = np.zeros(10)
        assert _binary_to_segments(mask) == []

    def test_all_positive(self):
        mask = np.ones(5)
        assert _binary_to_segments(mask) == [(0, 4)]


class TestTrainerInit:

    def test_creates_with_defaults(self, tmp_path):
        model = CausalTCN(in_features=16, n_labels=2)
        train_cfg = {"pos_weight": [1.0, 1.0], "lr": 1e-3}
        dataset_cfg = {
            "labels": ["a", "b"],
            "fps": 30.0,
            "paths": {
                "source_splits_dir": str(tmp_path),
                "features_dir": str(tmp_path),
                "labels_dir": str(tmp_path),
            },
            "window": {"T": 64, "stride": 16},
            "sampling": {},
        }
        trainer = Trainer(
            model, train_cfg, dataset_cfg, device="cpu",
            checkpoint_dir=tmp_path / "ckpt",
        )
        assert trainer.patience == 8  # default
        assert (tmp_path / "ckpt").exists()

    def test_save_and_load_checkpoint(self, tmp_path):
        model = CausalTCN(in_features=16, n_labels=2)
        train_cfg = {"pos_weight": [1.0, 1.0], "lr": 1e-3}
        dataset_cfg = {
            "labels": ["a", "b"],
            "fps": 30.0,
            "paths": {
                "source_splits_dir": str(tmp_path),
                "features_dir": str(tmp_path),
                "labels_dir": str(tmp_path),
            },
            "window": {"T": 64, "stride": 16},
            "sampling": {},
        }
        trainer = Trainer(
            model, train_cfg, dataset_cfg, device="cpu",
            checkpoint_dir=tmp_path / "ckpt",
        )
        path = trainer.save_checkpoint("test.pt")
        assert path.exists()

        # Load into a fresh trainer
        model2 = CausalTCN(in_features=16, n_labels=2)
        trainer2 = Trainer(
            model2, train_cfg, dataset_cfg, device="cpu",
            checkpoint_dir=tmp_path / "ckpt",
        )
        trainer2.load_checkpoint("test.pt")
        # Weights should match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)
