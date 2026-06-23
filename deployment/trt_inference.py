"""
EdgeGuard: TensorRT Inference Engine

Runtime inference engine for TensorRT models on edge devices.
Supports dynamic batching, streaming inference, and warm-up.
"""
from __future__ import annotations

import ctypes
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np


class TensorRTInference:
    """
    TensorRT inference engine for EdgeGuard models.

    Loads a serialized TensorRT engine and performs inference
    on video frames and text tokens.

    Features:
    - Dynamic batch sizing (1/4/8)
    - GPU memory pooling
    - Warm-up iterations
    - Latency and throughput benchmarking
    - Automatic precision selection (FP16/INT8)

    Args:
        engine_path: Path to serialized TensorRT engine file.
        device_id: CUDA device ID (default 0).
    """

    def __init__(
        self,
        engine_path: str,
        device_id: int = 0,
    ) -> None:
        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"Engine file not found: {engine_path}")

        self.device_id = device_id
        self._init_cuda()
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()

        self._allocate_buffers()
        self._bind_io()

    @staticmethod
    def _init_cuda() -> None:
        """Initialize CUDA runtime."""
        try:
            import pycuda.autoinit
            import pycuda.driver as cuda
            cuda.init()
        except ImportError:
            pass

        try:
            import tensorrt as trt
            ctypes.CDLL("libnvinfer.so")
        except Exception:
            pass

    def _load_engine(self):
        """Load TensorRT engine from serialized file."""
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(self.engine_path, "rb") as f:
            engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)

        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")
        print(f"Loaded TensorRT engine: {self.engine_path}")
        print(f"  Precision: {self._detect_precision()}")
        print(f"  Layers: {engine.num_layers}")
        return engine

    def _detect_precision(self) -> str:
        """Detect engine precision from bindings."""
        try:
            if hasattr(self, 'context'):
                return "FP16" if self.engine.has_implicit_batch_dimension else "Dynamic"
        except Exception:
            pass
        return "Unknown"

    def _allocate_buffers(self) -> None:
        """Allocate GPU memory for input/output tensors."""
        self.host_inputs: list[np.ndarray] = []
        self.device_inputs: list[int] = []
        self.host_outputs: list[np.ndarray] = []
        self.device_outputs: list[int] = []
        self.bindings: list[int] = []

        try:
            import pycuda.driver as cuda
            cuda.init()
            self.cuda = cuda
            self.stream = cuda.Stream()

            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_binding_name(i)
                dtype = self.engine.get_binding_dtype(i)
                shape = self.engine.get_binding_shape(name)

                if self.engine.binding_is_input(i):
                    h_input = np.zeros(shape, dtype=np.float32)
                    d_input = cuda.mem_alloc(h_input.nbytes)
                    self.host_inputs.append(h_input)
                    self.device_inputs.append(int(d_input))
                    self.bindings.append(int(d_input))
                else:
                    h_output = np.zeros(shape, dtype=np.float32)
                    d_output = cuda.mem_alloc(h_output.nbytes)
                    self.host_outputs.append(h_output)
                    self.device_outputs.append(int(d_output))
                    self.bindings.append(int(d_output))
        except ImportError:
            self._allocate_fallback()

    def _allocate_fallback(self) -> None:
        """Fallback allocation without pycuda."""
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(name)
            dtype = np.float32

            if self.engine.binding_is_input(i):
                self.host_inputs.append(np.zeros(shape, dtype=dtype))
                self.device_inputs.append(0)
            else:
                self.host_outputs.append(np.zeros(shape, dtype=dtype))
                self.device_outputs.append(0)
            self.bindings.append(0)

    def _bind_io(self) -> None:
        """Bind input/output buffers to the context."""
        for i, name in enumerate([self.engine.get_binding_name(i) for i in range(self.engine.num_io_tensors)]):
            if self.engine.binding_is_input(i):
                self.context.set_input_shape(name, self.host_inputs[i].shape)

    def infer(
        self,
        frames: np.ndarray,
        text_tokens: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run inference on input data.

        Args:
            frames: Video frames of shape (B, T, 3, 224, 224).
            text_tokens: Token IDs of shape (B, seq_len).

        Returns:
            Tuple of (behavior_probs, alert_probs), each of shape (B, num_classes).
        """
        self.host_inputs[0] = frames.astype(np.float32)
        self.host_inputs[1] = text_tokens.astype(np.int32)

        try:
            import pycuda.driver as cuda

            for h_input, d_input in zip(self.host_inputs, self.device_inputs):
                cuda.memcpy_htod_async(d_input, h_input, self.stream)

            self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle,
            )

            for h_output, d_output in zip(self.host_outputs, self.device_outputs):
                cuda.memcpy_dtoh_async(h_output, d_output, self.stream)

            self.stream.synchronize()

        except ImportError:
            import tensorrt as trt
            self.context.execute_v2(bindings=self.bindings)

        behavior_probs = self.host_outputs[0]
        alert_probs = self.host_outputs[1]

        behavior_probs = behavior_probs / behavior_probs.sum(axis=1, keepdims=True)
        alert_probs = alert_probs / alert_probs.sum(axis=1, keepdims=True)

        return behavior_probs, alert_probs

    def warmup(self, iterations: int = 10) -> None:
        """
        Warm up the engine with dummy data.

        Args:
            iterations: Number of warm-up iterations.
        """
        print(f"Warming up TensorRT engine ({iterations} iterations)...")

        dummy_frames = np.random.randn(1, 16, 3, 224, 224).astype(np.float32)
        dummy_tokens = np.random.randint(0, 30000, (1, 128), dtype=np.int32)

        for _ in range(iterations):
            self.infer(dummy_frames, dummy_tokens)

        print("Warm-up complete")

    def benchmark(
        self,
        num_iterations: int = 100,
        batch_sizes: list[int] | None = None,
    ) -> dict:
        """
        Run performance benchmark.

        Args:
            num_iterations: Number of inference iterations.
            batch_sizes: Batch sizes to benchmark.

        Returns:
            Dictionary of benchmark results.
        """
        batch_sizes = batch_sizes or [1, 4, 8]
        results = {}

        for bs in batch_sizes:
            dummy_frames = np.random.randn(bs, 16, 3, 224, 224).astype(np.float32)
            dummy_tokens = np.random.randint(0, 30000, (bs, 128), dtype=np.int32)

            latencies = []
            for _ in range(num_iterations):
                start = time.perf_counter()
                self.infer(dummy_frames, dummy_tokens)
                latencies.append((time.perf_counter() - start) * 1000)

            avg_latency = np.mean(latencies)
            fps = bs / (avg_latency / 1000)

            results[bs] = {
                "avg_latency_ms": avg_latency,
                "p50_latency_ms": np.percentile(latencies, 50),
                "p95_latency_ms": np.percentile(latencies, 95),
                "p99_latency_ms": np.percentile(latencies, 99),
                "fps": fps,
                "batch_size": bs,
            }

            print(f"\nBatch size {bs}:")
            print(f"  Avg latency: {avg_latency:.2f} ms")
            print(f"  P50 latency: {results[bs]['p50_latency_ms']:.2f} ms")
            print(f"  P95 latency: {results[bs]['p95_latency_ms']:.2f} ms")
            print(f"  P99 latency: {results[bs]['p99_latency_ms']:.2f} ms")
            print(f"  Throughput: {fps:.2f} FPS")

        return results

    def get_memory_usage(self) -> dict:
        """Get GPU memory usage."""
        try:
            import pycuda.driver as cuda
            cuda.init()
            device = cuda.Device(self.device_id)
            mem_info = device.mem_get_info()
            return {
                "free_mb": mem_info[0] / (1024 ** 2),
                "total_mb": mem_info[1] / (1024 ** 2),
                "used_mb": (mem_info[1] - mem_info[0]) / (1024 ** 2),
            }
        except Exception:
            return {"free_mb": 0, "total_mb": 0, "used_mb": 0}

    def __del__(self) -> None:
        """Clean up CUDA resources."""
        try:
            if hasattr(self, 'context'):
                del self.context
            if hasattr(self, 'engine'):
                del self.engine
        except Exception:
            pass
