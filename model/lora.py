"""
EdgeGuard: Shared LoRA Module

Unified Low-Rank Adaptation (LoRA) implementation used by both
visual and text encoders. Follows the standard LoRA paper approach:
y = Wx + (alpha/r) * BAx, where B is zero-initialized so training
starts from the pretrained model.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation linear layer.

    Wraps a frozen linear layer with trainable low-rank decomposition:
        y = Wx + (alpha/r) * BAx

    where:
        - W is the frozen pretrained weight
        - A is initialized via kaiming uniform (standard)
        - B is initialized to zero (so pre-trained output is preserved at start)

    Args:
        original_layer: The frozen pretrained linear layer to adapt.
        r: LoRA rank (low-rank dimension).
        lora_alpha: LoRA scaling factor (default 2*r for stability).
        lora_dropout: Dropout probability for the LoRA input (default 0.0).
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        r: int,
        lora_alpha: int,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.original = original_layer
        self.original_weight = original_layer.weight
        self.original_bias = original_layer.bias
        self.original.requires_grad_(False)

        self.r = r
        self.scaling = lora_alpha / r
        self.dropout = nn.Dropout(p=lora_dropout)

        self.lora_A = nn.Parameter(torch.empty(r, original_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(original_layer.out_features, r))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out
