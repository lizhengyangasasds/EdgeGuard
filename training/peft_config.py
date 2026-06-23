"""
EdgeGuard: PEFT Configuration Module

LoRA and Adapter LoRA configuration for all model components.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from peft import LoraConfig, TaskType


@dataclass
class PEFTConfig:
    """Unified PEFT configuration for EdgeGuard training."""
    # LoRA config for visual and text encoders
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: Literal["none", "all", "lora_only"] = "none"
    lora_modules: tuple[str, ...] = ("q_proj", "v_proj")

    # Adapter LoRA config for LSTM
    adapter_r: int = 8
    adapter_alpha: int = 16
    adapter_dropout: float = 0.05
    adapter_bottleneck: int = 8

    # Which components to apply PEFT to
    apply_to_visual: bool = True
    apply_to_text: bool = True
    apply_to_temporal: bool = True
    apply_to_cross_attn: bool = True

    def get_lora_config(self) -> LoraConfig:
        """Get standard LoRA config for visual/text encoders."""
        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias=self.lora_bias,
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=list(self.lora_modules),
        )

    def get_adapter_config(self) -> dict:
        """Get adapter configuration dict for LSTM."""
        return {
            "r": self.adapter_r,
            "lora_alpha": self.adapter_alpha,
            "lora_dropout": self.adapter_dropout,
            "bottleneck_dim": self.adapter_bottleneck,
        }

    @classmethod
    def from_yaml(cls, data: dict) -> "PEFTConfig":
        """Create config from YAML dictionary."""
        return cls(
            lora_r=data.get("lora_r", 16),
            lora_alpha=data.get("lora_alpha", 32),
            lora_dropout=data.get("lora_dropout", 0.05),
            lora_bias=data.get("lora_bias", "none"),
            lora_modules=tuple(data.get("lora_modules", ["q_proj", "v_proj"])),
            adapter_r=data.get("adapter_r", 8),
            adapter_alpha=data.get("adapter_alpha", 16),
            adapter_dropout=data.get("adapter_dropout", 0.05),
            adapter_bottleneck=data.get("adapter_bottleneck", 8),
        )
