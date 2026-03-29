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


class CausalConv1d:
    """1D convolution with left-only (causal) padding.

    Ensures output at time t depends only on inputs at times ≤ t.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int
    dilation : int
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation  # left-pad only
        # nn.Conv1d created in Phase 3


class TemporalBlock:
    """Single residual block: two CausalConv1d layers with BN + ReLU + Dropout.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int
    dilation : int
    dropout : float
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout = dropout
        # Build layers in Phase 3


class CausalTCN:
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
        self.in_features = in_features
        self.channels = channels or [64, 64, 64]
        self.kernel_size = kernel_size
        self.dilations = dilations or [1, 2, 4]
        self.dropout = dropout
        self.n_labels = n_labels

        assert len(self.channels) == len(self.dilations), (
            "channels and dilations must have the same length"
        )
        # Build nn.Module layers in Phase 3

    @property
    def receptive_field(self) -> int:
        """Total receptive field in frames."""
        rf = 1
        for d in self.dilations:
            rf += 2 * d * (self.kernel_size - 1)  # 2 conv layers per block
        return rf
