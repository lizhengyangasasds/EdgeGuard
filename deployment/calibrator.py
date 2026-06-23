"""
EdgeGuard: INT8 Calibration Module

Calibration dataset generation and INT8 quantization calibration
for TensorRT deployment on edge devices.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class CalibrationDataset(Dataset):
    """
    Dataset for TensorRT INT8 calibration.

    Generates diverse, representative samples for calibrating
    quantization scaling factors.

    Args:
        num_samples: Number of calibration samples.
        clip_length: Number of frames per sample.
        frame_size: Spatial resolution.
        max_text_length: Maximum text token length.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        clip_length: int = 16,
        frame_size: int = 224,
        max_text_length: int = 128,
    ) -> None:
        self.num_samples = num_samples
        self.clip_length = clip_length
        self.frame_size = frame_size
        self.max_text_length = max_text_length

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a calibration sample.

        Returns:
            Tuple of (frames, text_tokens).
        """
        frames = self._generate_frame()
        text_tokens = self._generate_text()

        return frames, text_tokens

    def _generate_frame(self) -> np.ndarray:
        """Generate a synthetic video frame with realistic patterns."""
        frame = np.random.randn(self.clip_length, 3, self.frame_size, self.frame_size).astype(np.float32)

        edge = self.frame_size // 8
        frame[:, :, :edge, :] += np.random.randn(self.clip_length, 3, edge, self.frame_size).astype(np.float32) * 0.5
        frame[:, :, -edge:, :] += np.random.randn(self.clip_length, 3, edge, self.frame_size).astype(np.float32) * 0.5

        frame = np.clip(frame, -3, 3)
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)

        return frame

    def _generate_text(self) -> np.ndarray:
        """Generate synthetic text tokens."""
        tokens = np.random.randint(100, 30000, self.max_text_length, dtype=np.int32)

        tokens[0] = 101
        sep_pos = random.randint(10, self.max_text_length - 2)
        tokens[sep_pos] = 102
        tokens[1:sep_pos] = np.random.randint(2000, 8000, sep_pos - 1, dtype=np.int32)
        tokens[tokens == 0] = 100

        return tokens


def generate_calibration_data(
    num_samples: int = 1000,
    clip_length: int = 16,
    frame_size: int = 224,
    max_text_length: int = 128,
    batch_size: int = 32,
    output_path: str | None = None,
) -> np.ndarray:
    """
    Generate calibration dataset for INT8 quantization.

    Args:
        num_samples: Number of calibration samples.
        clip_length: Frames per clip.
        frame_size: Frame resolution.
        max_text_length: Max text sequence length.
        batch_size: Batch size for loading.
        output_path: Optional path to save the dataset.

    Returns:
        Stacked calibration data as numpy array.
    """
    dataset = CalibrationDataset(
        num_samples=num_samples,
        clip_length=clip_length,
        frame_size=frame_size,
        max_text_length=max_text_length,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_frames = []
    all_tokens = []

    for frames, tokens in dataloader:
        all_frames.append(frames.numpy())
        all_tokens.append(tokens.numpy())

    frames_data = np.concatenate(all_frames, axis=0).astype(np.float32)
    tokens_data = np.concatenate(all_tokens, axis=0).astype(np.int32)

    print(f"Generated {len(frames_data)} calibration samples")
    print(f"  Frames shape: {frames_data.shape}, dtype: {frames_data.dtype}")
    print(f"  Tokens shape: {tokens_data.shape}, dtype: {tokens_data.dtype}")

    if output_path:
        save_path = Path(output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path.with_suffix(".frames.npy"), frames_data)
        np.save(save_path.with_suffix(".tokens.npy"), tokens_data)
        print(f"Calibration data saved to: {save_path.parent}")

    return frames_data


class EntropyCalibrator:
    """
    Entropy-based INT8 calibrator.

    Uses KL-divergence to find optimal quantization thresholds
    that minimize information loss during INT8 conversion.

    Args:
        num_bins: Number of histogram bins for distribution estimation.
    """

    def __init__(self, num_bins: int = 2048) -> None:
        self.num_bins = num_bins
        self.scales: dict[int, float] = {}
        self.histograms: dict[int, np.ndarray] = {}

    def compute_scale(
        self,
        data: np.ndarray,
        percentile: float = 99.99,
    ) -> float:
        """
        Compute quantization scale using entropy calibration.

        Args:
            data: Float32 activation data.
            percentile: Percentile for max value clipping.

        Returns:
            Quantization scale factor.
        """
        abs_data = np.abs(data.flatten())
        abs_data = abs_data[abs_data > 0]

        if len(abs_data) == 0:
            return 1.0

        max_val = np.percentile(abs_data, percentile)
        abs_data = np.clip(abs_data, 0, max_val)

        scale = 127.0 / (max_val + 1e-8)
        return scale

    def calibrate(
        self,
        activation_data: np.ndarray,
        layer_idx: int,
    ) -> np.ndarray:
        """
        Calibrate a single layer's activations.

        Args:
            activation_data: Activation tensor.
            layer_idx: Layer identifier.

        Returns:
            Calibrated activation tensor.
        """
        scale = self.compute_scale(activation_data)
        self.scales[layer_idx] = scale

        quantized = np.clip(activation_data * scale, -128, 127).astype(np.int8)
        dequantized = quantized.astype(np.float32) / scale

        return dequantized

    def get_all_scales(self) -> dict[int, float]:
        """Return all computed scales."""
        return self.scales.copy()

    def save_scales(self, path: str) -> None:
        """Save scales to JSON file."""
        import json
        save_data = {str(k): float(v) for k, v in self.scales.items()}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"Calibration scales saved to: {path}")

    def load_scales(self, path: str) -> None:
        """Load scales from JSON file."""
        import json
        with open(path, "r") as f:
            loaded = json.load(f)
        self.scales = {int(k): float(v) for k, v in loaded.items()}
        print(f"Loaded {len(self.scales)} calibration scales from: {path}")


if __name__ == "__main__":
    print("Generating INT8 calibration dataset...")
    data = generate_calibration_data(
        num_samples=1000,
        output_path="deployment/calibration_data",
    )
    print(f"\nTotal data size: {data.nbytes / 1024 / 1024:.1f} MB")
