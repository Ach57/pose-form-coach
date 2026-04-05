"""Unit tests for src.models.causal_tcn — CausalTCN model."""

from __future__ import annotations

import torch
import pytest

from src.models.causal_tcn import CausalConv1d, TemporalBlock, CausalTCN


class TestCausalConv1d:

    def test_output_length_matches_input(self):
        conv = CausalConv1d(16, 32, kernel_size=3, dilation=2)
        x = torch.randn(2, 16, 64)
        out = conv(x)
        assert out.shape == (2, 32, 64)

    def test_causality(self):
        """Changing future input must not affect past output."""
        conv = CausalConv1d(8, 8, kernel_size=3, dilation=1)
        conv.eval()
        x = torch.randn(1, 8, 32)
        x2 = x.clone()
        x2[:, :, 16:] = 0.0
        with torch.no_grad():
            y1 = conv(x)
            y2 = conv(x2)
        assert torch.allclose(y1[:, :, :16], y2[:, :, :16], atol=1e-6)


class TestTemporalBlock:

    def test_shape_preserved(self):
        block = TemporalBlock(16, 32, kernel_size=3, dilation=2)
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == (2, 32, 64)

    def test_residual_same_channels(self):
        block = TemporalBlock(32, 32, kernel_size=3, dilation=1)
        x = torch.randn(1, 32, 20)
        out = block(x)
        assert out.shape == (1, 32, 20)


class TestCausalTCN:

    @pytest.fixture
    def model(self):
        return CausalTCN(
            in_features=16, channels=[64, 64, 64],
            kernel_size=3, dilations=[1, 2, 4],
            dropout=0.1, n_labels=2,
        )

    def test_output_shape(self, model):
        x = torch.randn(4, 64, 16)
        out = model(x)
        assert out.shape == (4, 64, 2)

    def test_causality_full_model(self, model):
        model.eval()
        x = torch.randn(1, 64, 16)
        x2 = x.clone()
        x2[:, 32:, :] = 0.0
        with torch.no_grad():
            y1 = model(x)
            y2 = model(x2)
        assert torch.allclose(y1[:, :32], y2[:, :32], atol=1e-6)

    def test_receptive_field(self, model):
        assert model.receptive_field == 29

    def test_variable_length_input(self, model):
        for T in [32, 100, 200]:
            x = torch.randn(1, T, 16)
            out = model(x)
            assert out.shape == (1, T, 2)

    def test_from_config(self):
        model = CausalTCN.from_config(
            "configs/shared/model.tcn.base.yaml",
            "configs/ohp/model.yaml",
        )
        assert model.in_features == 16
        assert model.n_labels == 2

    def test_backward_pass(self, model):
        x = torch.randn(2, 64, 16)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None
