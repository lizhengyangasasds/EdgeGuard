"""
EdgeGuard: Visual Encoder Module

DeiT-small-distilled with LoRA injection for visual feature extraction.
LoRA targets attention q_proj and v_proj layers with configurable rank.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel


@dataclass
class LoRAConfig:
    """LoRA configuration for visual encoder."""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    bias: str = "none"
    task_type: str = "FEATURE_EXTRACTION"


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation linear layer.

    Wraps a native linear layer with LoRA adaptation:
    y = Wx + BAx  where A and B are low-rank matrices.
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
        self.lora_B = nn.Parameter(torch.empty(original_layer.out_features, r))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out


class VisualEncoder(nn.Module):
    """
    DeiT-small-distilled visual encoder with LoRA.

    Extracts visual features from video frames using a distilled vision
    transformer, with LoRA adapters injected into attention layers.

    Args:
        pretrained_model: HuggingFace model identifier or local path.
        feature_dim: Output feature dimension (default 384).
        lora_config: LoRA configuration. If None, LoRA is disabled.
        freeze_backbone: Whether to freeze the base transformer weights.

    Example:
        >>> config = LoRAConfig(r=16, lora_alpha=32)
        >>> encoder = VisualEncoder(
        ...     pretrained_model="timm/deit_small_distilled_patch16_224",
        ...     feature_dim=384,
        ...     lora_config=config,
        ... )
        >>> frames = torch.randn(2, 3, 224, 224)
        >>> features = encoder(frames)  # (2, 384)
    """

    def __init__(
        self,
        pretrained_model: str = "timm/deit_small_distilled_patch16_224",
        feature_dim: int = 384,
        lora_config: Optional[LoRAConfig] = None,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.lora_enabled = lora_config is not None

        self.deit = AutoModel.from_pretrained(
            pretrained_model,
            trust_remote_code=True,
        )

        if freeze_backbone:
            self._freeze_backbone()

        # Projection layer: DeiT may output 384 or other dims depending on model
        deit_hidden = self.deit.config.hidden_size
        self.projection = nn.Linear(deit_hidden, feature_dim)

        if lora_config is not None:
            self._inject_lora(lora_config)

    def _freeze_backbone(self) -> None:
        """Freeze all DeiT parameters except LoRA layers and projection."""
        for name, param in self.deit.named_parameters():
            param.requires_grad_(False)

    def _inject_lora(self, config: LoRAConfig) -> None:
        """
        Replace attention q_proj and v_proj with LoRA-wrapped versions.

        LoRA injects trainable low-rank decomposition matrices A and B
        into the frozen attention projections, dramatically reducing
        the number of trainable parameters.
        """
        for layer in self.deit.encoder.layer:
            if "q_proj" in config.target_modules:
                original_q = layer.attention.attention.query
                lora_q = LoRALinear(
                    original_q, config.r, config.lora_alpha, config.lora_dropout
                )
                layer.attention.attention.query = lora_q

            if "v_proj" in config.target_modules:
                original_v = layer.attention.attention.value
                lora_v = LoRALinear(
                    original_v, config.r, config.lora_alpha, config.lora_dropout
                )
                layer.attention.attention.value = lora_v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from input frames.

        Args:
            x: Input tensor of shape (B, C, H, W) or (B, T, C, H, W) for video clips.
               T = temporal dimension (number of frames per clip).

        Returns:
            features: Visual features of shape (B, feature_dim) for single frame
                      or (B, T, feature_dim) for video clips.
        """
        original_shape = x.shape
        squeeze_temporal = False

        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            squeeze_temporal = True

        outputs = self.deit(x)
        pooled = outputs.last_hidden_state[:, 0]

        features = self.projection(pooled)

        if squeeze_temporal:
            features = features.view(B, T, self.feature_dim)

        return features

    def get_trainable_params(self) -> dict[str, int]:
        """Return count of trainable vs total parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {
            "trainable": trainable,
            "total": total,
            "trainable_ratio": trainable / total if total > 0 else 0.0,
        }
