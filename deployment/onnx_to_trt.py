"""
EdgeGuard: ONNX to TensorRT Converter

Converts ONNX models to TensorRT engine for edge deployment.
Handles INT8 quantization, layer fusion, and dynamic batching.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class TensorRTConverter:
    """
    Converts ONNX models to TensorRT engines.

    Supports FP32, FP16, and INT8 precision modes.
    INT8 mode uses a calibration dataset for quantization.

    Args:
        onnx_path: Path to input ONNX model.
        output_path: Path for output TensorRT engine.
        precision: Precision mode ("fp32", "fp16", "int8").
        max_batch_size: Maximum batch size for inference.
        min_batch_size: Minimum batch size.
        opt_batch_size: Optimal batch size for profiling.
        workspace_size: GPU workspace size in GB.
        dla_core: DLA core ID (for Jetson platforms, -1 to disable).
    """

    def __init__(
        self,
        onnx_path: str,
        output_path: str,
        precision: str = "fp16",
        max_batch_size: int = 8,
        min_batch_size: int = 1,
        opt_batch_size: int = 4,
        workspace_size: int = 4,
        dla_core: int = -1,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        self.output_path = Path(output_path)
        self.precision = precision.lower()
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.opt_batch_size = opt_batch_size
        self.workspace_size = workspace_size
        self.dla_core = dla_core

        self._check_tensorrt()

    @staticmethod
    def _check_tensorrt() -> None:
        """Check if TensorRT is available."""
        try:
            import tensorrt as trt
        except ImportError:
            print("WARNING: TensorRT not found. Install from https://developer.nvidia.com/tensorrt")

    def build_engine(
        self,
        calibration_data: Optional[np.ndarray] = None,
        calibration_cache: Optional[str] = None,
        enable_profiling: bool = False,
    ) -> str:
        """
        Build TensorRT engine from ONNX model.

        Args:
            calibration_data: Data for INT8 calibration (optional).
            calibration_cache: Path to save/load calibration cache.
            enable_profiling: Enable layer profiling.

        Returns:
            Path to generated TensorRT engine file.
        """
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {self.onnx_path}")

        with open(self.onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = [parser.get_error(i) for i in range(parser.num_errors)]
                raise RuntimeError(f"ONNX parse errors: {errors}")

        config = builder.create_builder_config()
        config.max_workspace_size = self.workspace_size * (1 << 30)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace_size * (1 << 30))

        if self.precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        elif self.precision == "int8":
            if not builder.platform_has_fast_int8:
                print("WARNING: Platform doesn't support fast INT8, falling back to FP16")
                config.set_flag(trt.BuilderFlag.FP16)
            else:
                config.set_flag(trt.BuilderFlag.INT8)
                if calibration_data is not None:
                    calibrator = Int8Calibrator(
                        calibration_data,
                        cache_file=calibration_cache,
                        batch_size=self.opt_batch_size,
                    )
                    config.int8_calibrator = calibrator

        if enable_profiling:
            config.set_flag(trt.BuilderFlag.PROFILING)

        profile = builder.create_optimization_profile()
        profile.set_shape(
            "frames",
            (self.min_batch_size, 16, 3, 224, 224),
            (self.opt_batch_size, 16, 3, 224, 224),
            (self.max_batch_size, 16, 3, 224, 224),
        )
        profile.set_shape(
            "text_tokens",
            (self.min_batch_size, 128),
            (self.opt_batch_size, 128),
            (self.max_batch_size, 128),
        )
        config.add_optimization_profile(profile)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        engine_bytes = builder.build_serialized_network(network, config)

        if engine_bytes is None:
            raise RuntimeError("TensorRT engine build failed")

        with open(self.output_path, "wb") as f:
            f.write(engine_bytes)

        print(f"TensorRT engine saved to: {self.output_path}")
        self._print_engine_info(engine_bytes)
        return str(self.output_path)

    @staticmethod
    def _print_engine_info(engine_bytes: bytes) -> None:
        """Print basic engine information."""
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine:
            print(f"Engine has {engine.num_layers} layers and {engine.num_weights} weights")
            print(f"IO bindings: {[engine.get_binding_name(i) for i in range(engine.num_io_tensors)]}")


class Int8Calibrator:
    """
    INT8 calibration for TensorRT.

    Uses entropy calibration (default) to determine optimal quantization
    scaling factors for each layer.

    Args:
        calibration_data: Numpy array of calibration samples.
        cache_file: Path to save/load calibration cache.
        batch_size: Batch size for calibration.
        max_calibration_samples: Maximum samples to use for calibration.
    """

    def __init__(
        self,
        calibration_data: np.ndarray,
        cache_file: str | None = None,
        batch_size: int = 8,
        max_calibration_samples: int = 1000,
    ) -> None:
        self.calibration_data = calibration_data
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.max_calibration_samples = max_calibration_samples
        self.batch_idx = 0
        self.num_samples = min(len(calibration_data), max_calibration_samples)

        self._calibration_cache: dict[str, np.ndarray] = {}

        if cache_file and Path(cache_file).exists():
            self._load_cache(cache_file)

    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def get_batch(self, names: list[str]) -> list[np.ndarray] | None:
        """Get next calibration batch."""
        if self.batch_idx >= self.num_samples:
            return None

        start = self.batch_idx
        end = min(start + self.batch_size, self.num_samples)
        self.batch_idx = end

        return [self.calibration_data[start:end].astype(np.float32)]

    def get_batch_size(self) -> int:
        """Return calibration batch size."""
        return self.batch_size

    def read_calibration_cache(self) -> bytes | None:
        """Read calibration cache."""
        if self.cache_file and Path(self.cache_file).exists():
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        """Write calibration cache."""
        if self.cache_file:
            Path(self.cache_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "wb") as f:
                f.write(cache)
            print(f"Calibration cache saved to: {self.cache_file}")

    def _load_cache(self, cache_file: str) -> None:
        """Load calibration cache from file."""
        try:
            with open(cache_file, "rb") as f:
                content = f.read()
            print(f"Loaded calibration cache from: {cache_file}")
        except Exception:
            pass


def onnx_to_trt(
    onnx_path: str,
    output_path: str,
    precision: str = "fp16",
    calibration_data: Optional[np.ndarray] = None,
    **kwargs,
) -> str:
    """
    Convenience function to convert ONNX to TensorRT.

    Args:
        onnx_path: Input ONNX model path.
        output_path: Output TensorRT engine path.
        precision: Precision mode.
        calibration_data: Data for INT8 calibration.
        **kwargs: Additional arguments for TensorRTConverter.

    Returns:
        Path to generated engine.
    """
    converter = TensorRTConverter(
        onnx_path=onnx_path,
        output_path=output_path,
        precision=precision,
        **kwargs,
    )
    return converter.build_engine(calibration_data=calibration_data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT engine")
    parser.add_argument("--onnx", type=str, required=True, help="Input ONNX model path")
    parser.add_argument("--output", type=str, required=True, help="Output TensorRT engine path")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--max_batch", type=int, default=8)
    parser.add_argument("--workspace", type=int, default=4, help="Workspace size in GB")
    parser.add_argument("--calibration_cache", type=str, default=None)

    args = parser.parse_args()

    onnx_to_trt(
        onnx_path=args.onnx,
        output_path=args.output,
        precision=args.precision,
        max_batch_size=args.max_batch,
        workspace_size=args.workspace,
        calibration_cache=args.calibration_cache,
    )
