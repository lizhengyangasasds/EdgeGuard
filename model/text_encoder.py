"""
EdgeGuard: Text Encoder Module

DistilBERT with LoRA injection for text feature extraction.
Processes alert text tokens and projects to a shared embedding space.
"""
from __future__ import annotations

from .lora import LoRALinear
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


@dataclass
class TextEncoderConfig:
    """Configuration for the text encoder."""
    pretrained_model: str = "distilbert-base-uncased"
    feature_dim: int = 768
    projection_dim: int = 384
    max_seq_length: int = 128
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


class TextEncoder(nn.Module):
    """
    DistilBERT-based text encoder with LoRA.

    Encodes alert text descriptions into feature vectors compatible
    with the visual encoder's output space.

    Args:
        config: Text encoder configuration.
        lora_enabled: Enable LoRA adaptation (default True).
        freeze_backbone: Freeze base transformer weights (default True).

    Example:
        >>> cfg = TextEncoderConfig(projection_dim=384)
        >>> encoder = TextEncoder(cfg)
        >>> tokenizer = encoder.get_tokenizer()
        >>> tokens = tokenizer("Unauthorized person detected", return_tensors="pt")
        >>> features = encoder(**tokens)  # (1, 384)
    """

    def __init__(
        self,
        config: Optional[TextEncoderConfig] = None,
        lora_enabled: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        if config is None:
            config = TextEncoderConfig()

        self.config = config
        self.feature_dim = config.projection_dim
        self.lora_enabled = lora_enabled

        self.bert = AutoModel.from_pretrained(
            config.pretrained_model,
            trust_remote_code=True,
        )

        if freeze_backbone:
            self._freeze_backbone()

        self.projection = nn.Linear(config.feature_dim, config.projection_dim)

        if lora_enabled:
            self._inject_lora(config)

        self.dropout = nn.Dropout(0.1)

    def _freeze_backbone(self) -> None:
        """Freeze all DistilBERT parameters."""
        for param in self.bert.parameters():
            param.requires_grad_(False)

    def _inject_lora(self, config: TextEncoderConfig) -> None:
        """Inject LoRA into DistilBERT attention layers."""
        for layer in self.bert.transformer.layer:
            q_proj = layer.attention.q_lin
            v_proj = layer.attention.v_lin

            r = config.lora_r
            alpha = config.lora_alpha
            dropout = config.lora_dropout

            layer.attention.q_lin = self._make_lora_linear(
                q_proj, r, alpha, dropout
            )
            layer.attention.v_lin = self._make_lora_linear(
                v_proj, r, alpha, dropout
            )

    @staticmethod
    def _make_lora_linear(
        original: nn.Linear,
        r: int,
        lora_alpha: float,
        lora_dropout: float,
    ) -> nn.Module:
        """Wrap a linear layer with LoRA adaptation."""
        return LoRALinear(original, r, lora_alpha, lora_dropout)

    def get_tokenizer(self) -> AutoTokenizer:
        """Return the tokenizer for this encoder."""
        return AutoTokenizer.from_pretrained(self.config.pretrained_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode text into feature vectors.

        Args:
            input_ids: Token IDs of shape (B, seq_len).
            attention_mask: Attention mask of shape (B, seq_len).

        Returns:
            features: Text features of shape (B, projection_dim).
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        projected = self.projection(pooled)
        return self.dropout(projected)

    def get_trainable_params(self) -> dict[str, int]:
        """Return count of trainable vs total parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {
            "trainable": trainable,
            "total": total,
            "trainable_ratio": trainable / total if total > 0 else 0.0,
        }


