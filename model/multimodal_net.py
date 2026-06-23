"""
EdgeGuard: Multimodal Network

Integrates visual encoder (DeiT+LoRA), text encoder (DistilBERT+LoRA),
temporal encoder (LSTM+Adapter), cross-modal attention, and dual
classification heads into a unified anomaly detection pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from .visual_encoder import VisualEncoder, LoRAConfig
from .text_encoder import TextEncoder, TextEncoderConfig
from .temporal_encoder import TemporalEncoder, AdapterLoRAConfig
from .cross_modal import CrossModalAttention, CrossModalPooler
from .classification_head import BehaviorClassifier, AlertClassifier


@dataclass
class EdgeGuardConfig:
    """Complete configuration for the EdgeGuard multimodal network."""
    # Visual encoder
    visual_pretrained: str = "timm/deit_small_distilled_patch16_224"
    visual_feature_dim: int = 384
    visual_lora_r: int = 16
    visual_lora_alpha: int = 32
    visual_lora_dropout: float = 0.05
    # Text encoder
    text_pretrained: str = "distilbert-base-uncased"
    text_feature_dim: int = 768
    text_projection_dim: int = 384
    text_lora_r: int = 16
    text_lora_alpha: int = 32
    text_lora_dropout: float = 0.05
    # Temporal encoder
    temporal_hidden_dim: int = 256
    temporal_num_layers: int = 2
    temporal_dropout: float = 0.1
    adapter_bottleneck_dim: int = 8
    adapter_r: int = 8
    adapter_alpha: int = 16
    adapter_dropout: float = 0.05
    # Cross-modal attention
    cross_attn_num_heads: int = 8
    cross_attn_dropout: float = 0.1
    # Classification
    classifier_dropout: float = 0.3
    # Behavior: 7 classes, Alert: 5 classes
    num_behavior_classes: int = 7
    num_alert_classes: int = 5
    # Video
    clip_length: int = 16
    frame_size: int = 224


class EdgeGuardMultimodalNet(nn.Module):
    """
    EdgeGuard multimodal anomaly detection network.

    Architecture:
        1. Visual Encoder (DeiT + LoRA): Extract per-frame visual features
        2. Text Encoder (DistilBERT + LoRA): Encode alert text context
        3. Temporal Encoder (LSTM + Adapter): Model temporal sequence
        4. Cross-Modal Attention: Fuse visual and text features
        5. Dual Classifiers: Behavior + Alert classification

    Forward pass:
        frames (B, T, 3, 224, 224) + text_tokens (B, L)
          -> visual_feats (B, T, 384)
          -> text_feats (B, 384)
          -> cross_fused (B, T, 384) via CrossAttn
          -> temporal_out (B, 256) via LSTM
          -> concat (B, 640)
          -> behavior_logits (B, 7) + alert_logits (B, 5)

    Args:
        config: EdgeGuard model configuration.

    Example:
        >>> cfg = EdgeGuardConfig()
        >>> model = EdgeGuardMultimodalNet(cfg)
        >>> frames = torch.randn(2, 16, 3, 224, 224)
        >>> text_tokens = torch.randint(0, 30000, (2, 32))
        >>> behavior_logits, alert_logits = model(frames, text_tokens)
        >>> print(behavior_logits.shape)   # (2, 7)
        >>> print(alert_logits.shape)      # (2, 5)
    """

    def __init__(self, config: Optional[EdgeGuardConfig] = None) -> None:
        super().__init__()

        if config is None:
            config = EdgeGuardConfig()

        self.config = config
        self.clip_length = config.clip_length

        # 1. Visual Encoder with LoRA
        visual_lora_config = LoRAConfig(
            r=config.visual_lora_r,
            lora_alpha=config.visual_lora_alpha,
            lora_dropout=config.visual_lora_dropout,
        )
        self.visual_encoder = VisualEncoder(
            pretrained_model=config.visual_pretrained,
            feature_dim=config.visual_feature_dim,
            lora_config=visual_lora_config,
            freeze_backbone=True,
        )

        # 2. Text Encoder with LoRA
        text_config = TextEncoderConfig(
            pretrained_model=config.text_pretrained,
            feature_dim=config.text_feature_dim,
            projection_dim=config.text_projection_dim,
            lora_r=config.text_lora_r,
            lora_alpha=config.text_lora_alpha,
            lora_dropout=config.text_lora_dropout,
        )
        self.text_encoder = TextEncoder(
            config=text_config,
            lora_enabled=True,
            freeze_backbone=True,
        )

        # 3. Temporal Encoder: LSTM over concatenated visual+text per frame
        temporal_input_dim = config.visual_feature_dim + config.text_projection_dim
        adapter_config = AdapterLoRAConfig(
            bottleneck_dim=config.adapter_bottleneck_dim,
            r=config.adapter_r,
            lora_alpha=config.adapter_alpha,
            lora_dropout=config.adapter_dropout,
        )
        self.temporal_encoder = TemporalEncoder(
            input_dim=temporal_input_dim,
            hidden_dim=config.temporal_hidden_dim,
            num_layers=config.temporal_num_layers,
            dropout=config.temporal_dropout,
            adapter_config=adapter_config,
            bidirectional=False,
        )

        # 4. Cross-Modal Attention: visual attends to text
        self.cross_attention = CrossModalAttention(
            query_dim=config.visual_feature_dim,
            key_value_dim=config.text_projection_dim,
            num_heads=config.cross_attn_num_heads,
            dropout=config.cross_attn_dropout,
        )
        self.cross_attn_pooler = CrossModalPooler(config.visual_feature_dim)

        # 5. Dual Classification Heads
        fusion_dim = config.visual_feature_dim + config.temporal_hidden_dim
        self.behavior_classifier = BehaviorClassifier(
            input_dim=fusion_dim,
            num_classes=config.num_behavior_classes,
            dropout=config.classifier_dropout,
        )
        self.alert_classifier = AlertClassifier(
            input_dim=fusion_dim,
            num_classes=config.num_alert_classes,
            dropout=config.classifier_dropout,
        )

    def encode_visual(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode video frames through visual encoder.

        Args:
            frames: Video frames of shape (B, T, 3, H, W) or (B, 3, H, W).

        Returns:
            Visual features of shape (B, T, visual_dim).
        """
        if frames.dim() == 4:
            frames = frames.unsqueeze(1)
        return self.visual_encoder(frames)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode text tokens through text encoder.

        Args:
            input_ids: Token IDs of shape (B, seq_len).
            attention_mask: Optional attention mask.

        Returns:
            Text features of shape (B, text_dim).
        """
        return self.text_encoder(input_ids, attention_mask)

    def fuse_cross_modal(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse visual and text features via cross-attention.

        Args:
            visual_features: Shape (B, T, visual_dim).
            text_features: Shape (B, text_dim).

        Returns:
            Fused features of shape (B, visual_dim).
        """
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)
        fused = self.cross_attention(visual_features, text_features)
        pooled = self.cross_attn_pooler(fused)
        return pooled

    def forward(
        self,
        frames: torch.Tensor,
        text_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass through the multimodal network.

        Args:
            frames: Video frames of shape (B, T, 3, 224, 224).
            text_tokens: Token IDs of shape (B, seq_len).
            attention_mask: Optional attention mask for text.

        Returns:
            Tuple of (behavior_logits, alert_logits), each of shape (B, num_classes).
        """
        B, T, C, H, W = frames.shape
        visual_feats = self.encode_visual(frames)
        text_feats = self.encode_text(text_tokens, attention_mask)

        cross_fused = self.fuse_cross_modal(visual_feats, text_feats)

        visual_per_frame = visual_feats.mean(dim=1)
        text_expanded = text_feats.unsqueeze(1).expand(-1, T, -1)
        multimodal_seq = torch.cat([visual_per_frame.unsqueeze(1).expand(-1, T, -1), text_expanded], dim=-1)

        temporal_out = self.temporal_encoder(multimodal_seq)

        fusion = torch.cat([cross_fused, temporal_out], dim=-1)

        behavior_logits = self.behavior_classifier(fusion)
        alert_logits = self.alert_classifier(fusion)

        return behavior_logits, alert_logits

    def get_trainable_params(self) -> dict[str, int]:
        """Return trainable parameter statistics."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        lora_params = sum(
            p.numel()
            for name, p in self.named_parameters()
            if p.requires_grad and ("lora_" in name or "adapter" in name or "classifier" in name or "cross_attn" in name or "temporal" in name)
        )
        return {
            "trainable": trainable,
            "total": total,
            "trainable_ratio": trainable / total if total > 0 else 0.0,
            "lora_adapters": lora_params,
        }

    def print_trainable_summary(self) -> None:
        """Print a summary of trainable parameters."""
        stats = self.get_trainable_params()
        print(f"Total parameters:      {stats['total']:,}")
        print(f"Trainable parameters:   {stats['trainable']:,}")
        print(f"Trainable ratio:       {stats['trainable_ratio']:.4%}")
        print(f"  - LoRA/Adapter/Cross: {stats['lora_adapters']:,}")
