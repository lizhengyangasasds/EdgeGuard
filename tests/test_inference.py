"""
EdgeGuard: Inference Engine Tests

Tests for TensorRT inference and ONNX export functionality.
"""
from __future__ import annotations

import pytest
import torch
import numpy as np

from model.multimodal_net import EdgeGuardMultimodalNet, EdgeGuardConfig


class TestONNXExport:
    """Tests for ONNX export functionality."""

    def test_onnx_export_runs(self):
        from training.export_onnx import ONNXExporter

        config = EdgeGuardConfig()
        model = EdgeGuardMultimodalNet(config)
        model.eval()

        exporter = ONNXExporter(model, clip_length=16, frame_size=224)

        dummy_frames = torch.randn(1, 16, 3, 224, 224)
        dummy_text = torch.randint(0, 30000, (1, 64))

        try:
            import onnx
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    (dummy_frames, dummy_text),
                    "test_model.onnx",
                    input_names=["frames", "text_tokens"],
                    output_names=["behavior_logits", "alert_logits"],
                    opset_version=14,
                )
            assert True
        except ImportError:
            pytest.skip("ONNX not installed")


class TestTensorRTInference:
    """Tests for TensorRT inference engine."""

    def test_trt_converter_init(self):
        from deployment.onnx_to_trt import TensorRTConverter

        try:
            converter = TensorRTConverter(
                onnx_path="nonexistent.onnx",
                output_path="test.engine",
                precision="fp16",
            )
            assert converter.precision == "fp16"
        except ImportError:
            pytest.skip("TensorRT not installed")

    def test_calibrator_init(self):
        from deployment.calibrator import Int8Calibrator, CalibrationDataset

        data = np.random.randn(32, 16, 3, 224, 224).astype(np.float32)
        calibrator = Int8Calibrator(data, batch_size=8)
        assert calibrator.num_samples == 32
        assert calibrator.batch_size == 8


class TestInferenceEngine:
    """Tests for the inference engine wrapper."""

    def test_pytorch_inference_runs(self):
        config = EdgeGuardConfig()
        model = EdgeGuardMultimodalNet(config)
        model.eval()

        frames = torch.randn(1, 16, 3, 224, 224)
        text_tokens = torch.randint(0, 30000, (1, 64))

        with torch.no_grad():
            behavior_logits, alert_logits = model(frames, text_tokens)

        assert behavior_logits.shape == (1, 7)
        probs = torch.softmax(behavior_logits, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
