"""
Causal TCN — Temporal Convolutional Network with causal (no-future) padding.

Architecture:
  Input (B, T, F) → [TemporalBlock × N] → per-frame logits (B, T, L)

Each TemporalBlock:
  CausalConv1d → BatchNorm → ReLU → Dropout → CausalConv1d → BatchNorm → ReLU → Dropout
  + residual connection (with 1×1 conv if channel mismatch)

Usage:
    model = CausalTCN(in_features=16, channels=[64,64,64], kernel_size=3,
                       dilations=[1,2,4], dropout=0.1, n_labels=2)
    logits = model(features)  # (B, T, 16) → (B, T, 2)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.io import load_config, load_yaml


class CausalConv1d(nn.Module):
    """1D convolution with left-only (causal) padding.

    Ensures output at time t depends only on inputs at times ≤ t.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = self.conv(x)
        # Remove the right-side padding to keep causal
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        return out


class TemporalBlock(nn.Module):
    """Single residual block: two CausalConv1d layers with BN + ReLU + Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        # 1×1 residual projection when channel dimensions differ
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        res = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.drop(out)
        return out + res


class CausalTCN(nn.Module):
    """Multi-label causal TCN for per-frame error detection.

    Parameters
    ----------
    in_features : int
        Number of input features per frame.
    channels : list[int]
        Hidden channel widths for each TemporalBlock.
    kernel_size : int
        Kernel size for causal convolutions.
    dilations : list[int]
        Dilation factor per block (len must match channels).
    dropout : float
        Dropout rate inside temporal blocks.
    n_labels : int
        Number of output labels (sigmoid heads).
    """

    def __init__(
        self,
        in_features: int = 16,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        dilations: list[int] | None = None,
        dropout: float = 0.1,
        n_labels: int = 2,
    ) -> None:
        super().__init__()
        channels = channels or [64, 64, 64]
        dilations = dilations or [1, 2, 4]
        assert len(channels) == len(dilations), (
            "channels and dilations must have the same length"
        )

        self.in_features = in_features
        self.n_labels = n_labels
        self._dilations = dilations
        self._kernel_size = kernel_size

        # Build temporal blocks
        blocks: list[nn.Module] = []
        ch_in = in_features
        for ch_out, dil in zip(channels, dilations):
            blocks.append(TemporalBlock(ch_in, ch_out, kernel_size, dil, dropout))
            ch_in = ch_out
        self.network = nn.Sequential(*blocks)

        # Per-frame classification head
        self.head = nn.Conv1d(channels[-1], n_labels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (B, T, F) — batch of feature sequences

        Returns
        -------
        logits : (B, T, L) — per-frame, per-label logits (pre-sigmoid)
        """
        # Conv1d expects (B, C, T)
        out = x.transpose(1, 2)
        out = self.network(out)
        out = self.head(out)
        # Back to (B, T, L)
        return out.transpose(1, 2)

    @property
    def receptive_field(self) -> int:
        """Total receptive field in frames."""
        rf = 1
        for d in self._dilations:
            rf += 2 * d * (self._kernel_size - 1)  # 2 conv layers per block
        return rf

    @classmethod
    def from_config(cls, *config_paths: str) -> "CausalTCN":
        """Build a CausalTCN from one or more YAML config files.

        Files are deep-merged left-to-right, so later files override earlier ones.
        This supports both the legacy single-file pattern and the new
        base + exercise-override pattern::

            # Legacy (single file)
            CausalTCN.from_config("configs/ohp/model.yaml")

            # New (shared base + exercise override)
            CausalTCN.from_config(
                "configs/shared/model.tcn.base.yaml",
                "configs/ohp/model.yaml",
            )
        """
        cfg = load_config(*config_paths)["model"]
        return cls(
            in_features=cfg["in_features"],
            channels=cfg["channels"],
            kernel_size=cfg["kernel_size"],
            dilations=cfg["dilations"],
            dropout=cfg.get("dropout", 0.1),
            n_labels=cfg["n_labels"],
        )
