"""
EdgeGuard: Temporal Encoder Module

2-layer LSTM with Adapter LoRA for temporal sequence modeling.
The Adapter LoRA inserts bottleneck adapters between LSTM layers
with residual connections, minimizing trainable parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class AdapterLoRAConfig:
    """Adapter LoRA configuration for LSTM layers."""
    bottleneck_dim: int = 8
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05


class AdapterLoRA(nn.Module):
    """
    Adapter with LoRA for LSTM hidden states.

    Implements: output = adapter(x) + x
    where adapter is a bottleneck network: Linear(d, r) -> ReLU -> Linear(r, d)

    Args:
        input_dim: Input feature dimension.
        bottleneck_dim: Bottleneck dimension (rank r).
        r: LoRA rank (must equal bottleneck_dim).
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout probability in the bottleneck.
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.scaling = lora_alpha / r

        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.lora_A = nn.Parameter(torch.randn(bottleneck_dim, bottleneck_dim) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(bottleneck_dim, bottleneck_dim))
        self.dropout = nn.Dropout(lora_dropout)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(bottleneck_dim, input_dim)

        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adapter with residual: output = adapter(x) + x."""
        h = self.down_project(x)
        lora_out = (self.dropout(h) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        h = h + lora_out
        h = self.activation(h)
        h = self.dropout(h)
        return self.up_project(h) + x


class TemporalEncoder(nn.Module):
    """
    2-layer bidirectional LSTM with optional Adapter LoRA.

    Processes sequences of multimodal features (visual + text) to capture
    temporal dynamics across video frames. Adapter LoRA is inserted between
    LSTM layers to enable efficient fine-tuning with minimal parameters.

    Args:
        input_dim: Input feature dimension (concatenated visual + text).
        hidden_dim: LSTM hidden state dimension (default 256).
        num_layers: Number of LSTM layers (default 2).
        dropout: Dropout probability between layers.
        adapter_config: Adapter LoRA configuration. If None, adapters are disabled.
        bidirectional: Whether to use bidirectional LSTM.

    Example:
        >>> encoder = TemporalEncoder(input_dim=768, hidden_dim=256)
        >>> seq = torch.randn(4, 16, 768)  # (B, T, features)
        >>> temporal = encoder(seq)  # (4, 256)
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        adapter_config: Optional[AdapterLoRAConfig] = None,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.adapter_enabled = adapter_config is not None
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        if adapter_config is not None:
            self._build_adapters(adapter_config, dropout)
        else:
            self.adapters: nn.ModuleList = nn.ModuleList()

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.layer_norm = nn.LayerNorm(lstm_output_dim)

    def _build_adapters(
        self,
        config: AdapterLoRAConfig,
        dropout: float,
    ) -> None:
        """Build adapter modules between LSTM layers."""
        self.adapters = nn.ModuleList()
        lstm_output_dim = self.hidden_dim * (2 if self.bidirectional else 1)

        for _ in range(self.num_layers - 1):
            self.adapters.append(
                AdapterLoRA(
                    input_dim=lstm_output_dim,
                    bottleneck_dim=config.bottleneck_dim,
                    r=config.r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process sequential features through LSTM layers.

        Args:
            x: Input sequence of shape (B, T, input_dim) where
               T is the temporal window length.

        Returns:
            output: Temporal features of shape (B, hidden_dim * num_directions).
                   Returns the last hidden state.
        """
        lstm_out, (hidden_n, _) = self.lstm(x)

        output = lstm_out[:, -1, :]
        if self.num_layers > 1 and self.adapter_enabled:
            adapted = lstm_out
            for adapter in self.adapters:
                adapted = adapter(adapted)
            output = adapted[:, -1, :]

        output = self.layer_norm(output)
        return output

    def get_trainable_params(self) -> dict[str, int]:
        """Return count of trainable vs total parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {
            "trainable": trainable,
            "total": total,
            "trainable_ratio": trainable / total if total > 0 else 0.0,
        }
