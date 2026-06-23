"""
EdgeGuard: Inference Pipeline Tests

Integration tests for the inference pipeline including
video processing, text processing, and end-to-end inference.
"""
from __future__ import annotations

import pytest
import torch
import numpy as np

from data.video_processor import VideoProcessor
from data.text_processor import TextProcessor, AlertTextGenerator
from data.augmentation import VisualAugmentation, MixupAugmentation
from model.multimodal_net import EdgeGuardMultimodalNet, EdgeGuardConfig


class TestVideoProcessor:
    """Tests for video processing."""

    def test_normalize_frames(self):
        processor = VideoProcessor(clip_length=16, frame_size=224)
        frames = np.random.randint(0, 255, (16, 224, 224, 3), dtype=np.uint8)
        normalized = processor.normalize_frames(frames)
        assert normalized.shape == (16, 3, 224, 224)
        assert normalized.dtype == np.float32

    def test_generate_clips(self):
        processor = VideoProcessor(clip_length=16, stride=4)
        frames = np.random.randint(0, 255, (64, 224, 224, 3), dtype=np.uint8)
        clips = list(processor.generate_clips(frames))
        assert len(clips) > 0
        assert clips[0].shape == (16, 224, 224, 3)

    def test_normalize_single_frame(self):
        processor = VideoProcessor()
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        normalized = processor.normalize_frames(frame[np.newaxis, ...])
        assert normalized.shape == (1, 3, 224, 224)


class TestTextProcessor:
    """Tests for text processing."""

    def test_tokenize_single(self):
        processor = TextProcessor()
        tokens = processor.tokenize("Test alert text")
        assert "input_ids" in tokens
        assert "attention_mask" in tokens
        assert tokens["input_ids"].shape[0] <= processor.max_length

    def test_tokenize_batch(self):
        processor = TextProcessor()
        texts = ["Alert text one", "Alert text two", "Alert text three"]
        tokens = processor.tokenize(texts, return_tensors="pt")
        assert tokens["input_ids"].shape[0] == 3

    def test_augment_synonym_replacement(self):
        processor = TextProcessor()
        text = "unauthorized person detected"
        augmented = processor.augment_synonym_replacement(text, augmentation_ratio=0.3)
        assert len(augmented) > 0

    def test_decode(self):
        processor = TextProcessor()
        tokens = processor.tokenize("Test text", return_tensors="pt")
        decoded = processor.decode(tokens["input_ids"])
        assert len(decoded) > 0


class TestAlertTextGenerator:
    """Tests for synthetic alert text generation."""

    def test_generate(self):
        generator = AlertTextGenerator(language="en")
        samples = generator.generate(num_samples=20)
        assert len(samples) == 20
        assert all("text" in s and "label" in s for s in samples)

    def test_generate_chinese(self):
        generator = AlertTextGenerator(language="zh")
        samples = generator.generate(num_samples=10)
        assert len(samples) == 10
        assert all("\u4e00" in s["text"] or s["text"][0].isascii() for s in samples)


class TestDataAugmentation:
    """Tests for data augmentation."""

    def test_visual_augmentation(self):
        aug = VisualAugmentation(crop_size=224, is_training=True)
        frames = torch.randn(16, 3, 224, 224)
        augmented = aug(frames)
        assert augmented.shape == frames.shape

    def test_mixup_augmentation(self):
        mixup = MixupAugmentation(alpha=0.2)
        frames1 = torch.randn(1, 16, 3, 224, 224)
        text1 = {"input_ids": torch.randint(0, 30000, (1, 32)), "attention_mask": torch.ones(1, 32)}
        frames2 = torch.randn(1, 16, 3, 224, 224)
        text2 = {"input_ids": torch.randint(0, 30000, (1, 32)), "attention_mask": torch.ones(1, 32)}

        mixed_frames, mixed_text, _, _ = mixup(
            frames1, text1, torch.tensor([0]), torch.tensor([0]),
            frames2, text2, torch.tensor([1]), torch.tensor([1]),
        )
        assert mixed_frames.shape == frames1.shape


class TestEndToEndInference:
    """End-to-end inference pipeline tests."""

    def test_model_forward_with_real_shapes(self):
        config = EdgeGuardConfig()
        model = EdgeGuardMultimodalNet(config)
        model.eval()

        frames = torch.randn(1, 16, 3, 224, 224)
        text_tokens = torch.randint(0, 30000, (1, 64))

        with torch.no_grad():
            behavior_logits, alert_logits = model(frames, text_tokens)

        assert behavior_logits.shape == (1, 7)
        assert alert_logits.shape == (1, 5)
        assert not torch.isnan(behavior_logits).any()
        assert not torch.isnan(alert_logits).any()

    def test_model_inference_with_attention_mask(self):
        config = EdgeGuardConfig()
        model = EdgeGuardMultimodalNet(config)
        model.eval()

        frames = torch.randn(2, 16, 3, 224, 224)
        text_ids = torch.randint(0, 30000, (2, 64))
        attention_mask = torch.cat([torch.ones(2, 48), torch.zeros(2, 16)], dim=1).long()

        with torch.no_grad():
            behavior_logits, alert_logits = model(frames, text_ids, attention_mask)

        assert behavior_logits.shape == (2, 7)
        assert alert_logits.shape == (2, 5)

    def test_video_processor_integration(self):
        processor = VideoProcessor(clip_length=16, frame_size=224)
        frames = np.random.randint(0, 255, (16, 224, 224, 3), dtype=np.uint8)
        normalized = processor.normalize_frames(frames)
        assert normalized.shape == (16, 3, 224, 224)

        frames_tensor = torch.from_numpy(normalized).unsqueeze(0)

        config = EdgeGuardConfig()
        model = EdgeGuardMultimodalNet(config)
        model.eval()

        text_tokens = torch.randint(0, 30000, (1, 32))

        with torch.no_grad():
            behavior_logits, alert_logits = model(frames_tensor, text_tokens)

        assert behavior_logits.shape == (1, 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
