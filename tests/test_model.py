"""
EdgeGuard: Model Architecture Tests

Unit tests for all model components to verify correct forward pass shapes,
gradient flow, and parameter counts.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from model.lora import LoRALinear
from model.visual_encoder import VisualEncoder, LoRAConfig
from model.text_encoder import TextEncoder, TextEncoderConfig
from model.temporal_encoder import TemporalEncoder, AdapterLoRA, AdapterLoRAConfig
from model.cross_modal import CrossModalAttention, CrossModalPooler
from model.classification_head import BehaviorClassifier, AlertClassifier, ClassificationHead
from model.multimodal_net import EdgeGuardMultimodalNet, EdgeGuardConfig


class TestLoRALinear:
    """Tests for the LoRA-adapted linear layer."""

    def test_forward_shape(self):
        original = torch.nn.Linear(384, 384)
        lora_layer = LoRALinear(original, r=16, lora_alpha=32, lora_dropout=0.05)
        x = torch.randn(2, 384)
        out = lora_layer(x)
        assert out.shape == (2, 384)

    def test_trainable_params(self):
        original = torch.nn.Linear(384, 384)
        lora_layer = LoRALinear(original, r=16, lora_alpha=32)
        trainable = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
        assert trainable == 16 * 384 + 384 * 16

    def test_output_differs_from_base(self):
        original = torch.nn.Linear(128, 128)
        lora_layer = LoRALinear(original, r=8, lora_alpha=16)
        x = torch.randn(1, 128)
        out_base = original(x)
        out_lora = lora_layer(x)
        # B is zero-initialized, so initial output == base output
        assert torch.allclose(out_base, out_lora, atol=1e-5)

        # But LoRA parameters are trainable and will diverge after training
        loss = out_lora.sum()
        loss.backward()
        assert lora_layer.lora_B.grad is not None

        # After gradient update, outputs should differ
        with torch.no_grad():
            lora_layer.lora_B.add_(lora_layer.lora_B.grad * 1e-3)
        out_after = lora_layer(x)
        assert not torch.allclose(out_base, out_after, atol=1e-4)


class TestVisualEncoder:
    """Tests for the visual encoder."""

    def test_output_shape_single_frame(self):
        encoder = VisualEncoder(feature_dim=384, lora_config=None, freeze_backbone=False)
        x = torch.randn(2, 3, 224, 224)
        out = encoder(x)
        assert out.shape == (2, 384)

    def test_output_shape_video_clip(self):
        encoder = VisualEncoder(feature_dim=384, lora_config=None, freeze_backbone=False)
        x = torch.randn(2, 8, 3, 224, 224)
        out = encoder(x)
        assert out.shape == (2, 8, 384)

    def test_lora_enabled(self):
        config = LoRAConfig(r=16, lora_alpha=32)
        encoder = VisualEncoder(feature_dim=384, lora_config=config, freeze_backbone=False)
        stats = encoder.get_trainable_params()
        assert stats["trainable"] > 0
        assert stats["trainable_ratio"] < 0.1


class TestTextEncoder:
    """Tests for the text encoder."""

    def test_output_shape(self):
        config = TextEncoderConfig(projection_dim=384)
        encoder = TextEncoder(config, lora_enabled=False, freeze_backbone=False)
        input_ids = torch.randint(0, 30000, (2, 32))
        out = encoder(input_ids)
        assert out.shape == (2, 384)

    def test_with_attention_mask(self):
        config = TextEncoderConfig(projection_dim=384)
        encoder = TextEncoder(config, lora_enabled=False, freeze_backbone=False)
        input_ids = torch.randint(0, 30000, (2, 32))
        attention_mask = torch.ones(2, 32)
        out = encoder(input_ids, attention_mask)
        assert out.shape == (2, 384)


class TestTemporalEncoder:
    """Tests for the temporal encoder."""

    def test_output_shape(self):
        encoder = TemporalEncoder(input_dim=768, hidden_dim=256, num_layers=2)
        x = torch.randn(4, 16, 768)
        out = encoder(x)
        assert out.shape == (4, 256)

    def test_with_adapter(self):
        config = AdapterLoRAConfig(bottleneck_dim=8, r=8)
        encoder = TemporalEncoder(input_dim=768, hidden_dim=256, adapter_config=config)
        x = torch.randn(2, 8, 768)
        out = encoder(x)
        assert out.shape == (2, 256)

    def test_bidirectional(self):
        encoder = TemporalEncoder(input_dim=384, hidden_dim=128, bidirectional=True)
        x = torch.randn(2, 10, 384)
        out = encoder(x)
        assert out.shape == (2, 256)


class TestCrossModalAttention:
    """Tests for cross-modal attention."""

    def test_output_shape(self):
        cross_attn = CrossModalAttention(query_dim=384, key_value_dim=384, num_heads=8)
        q = torch.randn(2, 16, 384)
        kv = torch.randn(2, 1, 384)
        out = cross_attn(q, kv)
        assert out.shape == (2, 16, 384)

    def test_cross_attn_pooler(self):
        pooler = CrossModalPooler(feature_dim=384)
        x = torch.randn(2, 16, 384)
        out = pooler(x)
        assert out.shape == (2, 384)


class TestClassificationHead:
    """Tests for classification heads."""

    def test_behavior_classifier_output(self):
        classifier = BehaviorClassifier(input_dim=640, num_classes=7)
        x = torch.randn(4, 640)
        out = classifier(x)
        assert out.shape == (4, 7)

    def test_alert_classifier_output(self):
        classifier = AlertClassifier(input_dim=640, num_classes=5)
        x = torch.randn(4, 640)
        out = classifier(x)
        assert out.shape == (4, 5)

    def test_probabilities_sum_to_one(self):
        classifier = BehaviorClassifier(input_dim=640, num_classes=7)
        x = torch.randn(2, 640)
        probs = classifier.get_probabilities(x)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


class TestEdgeGuardMultimodalNet:
    """Tests for the full multimodal network."""

    def test_full_forward_pass(self):
        config = EdgeGuardConfig()
        model = EdgeGuardMultimodalNet(config)
        model.eval()

        frames = torch.randn(2, 16, 3, 224, 224)
        text_tokens = torch.randint(0, 30000, (2, 32))

        with torch.no_grad():
            behavior_logits, alert_logits = model(frames, text_tokens)

        assert behavior_logits.shape == (2, 7)
        assert alert_logits.shape == (2, 5)

    def test_trainable_ratio(self):
        config = EdgeGuardConfig()
        model = EdgeGuardMultimodalNet(config)
        stats = model.get_trainable_params()
        assert stats["trainable_ratio"] < 0.05, f"Trainable ratio {stats['trainable_ratio']:.2%} exceeds 5%"

    def test_gradient_flow(self):
        config = EdgeGuardConfig()
        model = EdgeGuardMultimodalNet(config)

        frames = torch.randn(1, 8, 3, 224, 224)
        text_tokens = torch.randint(0, 30000, (1, 16))

        behavior_logits, alert_logits = model(frames, text_tokens)
        loss = behavior_logits.mean() + alert_logits.mean()
        loss.backward()

        trainable_params = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is not None]
        assert len(trainable_params) > 0, "No gradients flowing to trainable parameters"

    def test_separate_components(self):
        config = EdgeGuardConfig()
        model = EdgeGuardMultimodalNet(config)

        frames = torch.randn(2, 16, 3, 224, 224)
        text_tokens = torch.randint(0, 30000, (2, 32))

        visual_feats = model.encode_visual(frames)
        text_feats = model.encode_text(text_tokens)
        cross_fused = model.fuse_cross_modal(visual_feats, text_feats)

        assert visual_feats.shape == (2, 16, 384)
        assert text_feats.shape == (2, 384)
        assert cross_fused.shape == (2, 384)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
