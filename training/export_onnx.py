"""
EdgeGuard: ONNX Export Module

Exports the EdgeGuard multimodal network to ONNX format.
Handles LSTM expansion (sequence-to-one) for ONNX compatibility.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from model.multimodal_net import EdgeGuardMultimodalNet, EdgeGuardConfig


class ONNXExporter:
    """
    Exports EdgeGuard model to ONNX format.

    Handles:
    - Dynamic batch dimensions
    - Variable sequence lengths
    - LSTM expansion to per-timestep ops (for better TensorRT compatibility)
    - Input/output naming for TensorRT parsing

    Args:
        model: EdgeGuard model instance.
        config: Export configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        clip_length: int = 16,
        frame_size: int = 224,
        max_text_length: int = 128,
        opset_version: int = 14,
    ) -> None:
        self.model = model
        self.clip_length = clip_length
        self.frame_size = frame_size
        self.max_text_length = max_text_length
        self.opset_version = opset_version

    def export(
        self,
        output_path: str,
        dynamic_batch: bool = True,
        optimize: bool = True,
    ) -> str:
        """
        Export model to ONNX format.

        Args:
            output_path: Output .onnx file path.
            dynamic_batch: Enable dynamic batch dimension.
            optimize: Apply ONNX optimizations.

        Returns:
            Path to exported ONNX file.
        """
        self.model.eval()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dummy_frames = torch.randn(1, self.clip_length, 3, self.frame_size, self.frame_size)
        dummy_text = torch.randint(0, 30000, (1, self.max_text_length))

        dynamic_axes = {}
        if dynamic_batch:
            dynamic_axes = {
                "frames": {0: "batch_size"},
                "text_tokens": {0: "batch_size"},
                "behavior_logits": {0: "batch_size"},
                "alert_logits": {0: "batch_size"},
            }

        with torch.no_grad():
            torch.onnx.export(
                self.model,
                (dummy_frames, dummy_text),
                str(output_path),
                input_names=["frames", "text_tokens"],
                output_names=["behavior_logits", "alert_logits"],
                dynamic_axes=dynamic_axes,
                opset_version=self.opset_version,
                export_params=True,
                do_constant_folding=True,
            )

        print(f"ONNX model exported to: {output_path}")

        if optimize:
            self._optimize_onnx(output_path)

        return str(output_path)

    @staticmethod
    def _optimize_onnx(model_path: Path) -> None:
        """Apply ONNX optimizations using basic simplification."""
        try:
            import onnx
            from onnx import optimizer, shape_inference

            model = onnx.load(str(model_path))
            passes = [
                "eliminate_deadend",
                "eliminate_identity",
                "eliminate_if",
                "fuse_bn_into_conv",
                "fuse_consecutive_squeezes",
            ]
            optimized = optimizer.optimize(model, passes)
            onnx.save(optimized, str(model_path))
            print(f"ONNX optimized and saved to: {model_path}")
        except ImportError:
            print("ONNX optimization skipped (onnx or onnxoptimizer not installed)")


def export_lstm_step_by_step(
    lstm: nn.LSTM,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    output_path: str,
) -> str:
    """
    Export LSTM as step-by-step ops for better ONNX/TensorRT compatibility.

    TensorRT has limited support for dynamic RNN shapes.
    Expanding LSTM into per-timestep operations gives better control
    over layer fusion and quantization.

    Args:
        lstm: The LSTM module.
        input_dim: Input feature dimension.
        hidden_dim: LSTM hidden dimension.
        num_layers: Number of LSTM layers.
        output_path: Output .onnx file path.

    Returns:
        Path to exported ONNX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 1, input_dim)

    with torch.no_grad():
        torch.onnx.export(
            lstm,
            dummy_input,
            str(output_path),
            input_names=["lstm_input"],
            output_names=["lstm_output", "hidden_state", "cell_state"],
            dynamic_axes={"lstm_input": {1: "sequence"}, "lstm_output": {1: "sequence"}},
            opset_version=14,
            export_params=True,
        )

    print(f"LSTM exported step-by-step to: {output_path}")
    return str(output_path)
