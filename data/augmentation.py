"""
EdgeGuard: Data Augmentation Module

Visual augmentation (crop, flip, color jitter) and Mixup augmentation
for training the multimodal network.
"""
from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn as nn


class VisualAugmentation(nn.Module):
    """
    Composable visual augmentations for video frames.

    Applies random crop, horizontal flip, and color jitter during training.
    In evaluation mode, only resizes to crop_size.

    Args:
        crop_size: Target crop size (default 224).
        is_training: Enable random augmentations when True.
        hflip_prob: Probability of horizontal flip (default 0.5).
        color_jitter: Enable color jitter when True.

    Example:
        >>> aug = VisualAugmentation(crop_size=224, is_training=True)
        >>> frames = torch.randn(16, 3, 256, 256)
        >>> augmented = aug(frames)  # (16, 3, 224, 224)
    """

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(
        self,
        crop_size: int = 224,
        is_training: bool = True,
        hflip_prob: float = 0.5,
        color_jitter: bool = True,
    ) -> None:
        super().__init__()
        self.crop_size = crop_size
        self.is_training = is_training
        self.hflip_prob = hflip_prob
        self.color_jitter = color_jitter

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to a batch of frames.

        Args:
            frames: Frames of shape (T, C, H, W) or (B, T, C, H, W).

        Returns:
            Augmented frames of same shape.
        """
        squeeze_batch = False
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
            squeeze_batch = True

        B, T, C, H, W = frames.shape

        if self.is_training and random.random() < self.hflip_prob:
            frames = frames.flip(-1)

        if self.is_training and self.color_jitter:
            frames = self._color_jitter(frames)

        frames = self._center_crop(frames, self.crop_size)

        if squeeze_batch:
            frames = frames.squeeze(0)
        return frames

    def _color_jitter(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply brightness, contrast, and saturation jitter."""
        for _ in range(3):
            j_type = random.choice(["brightness", "contrast", "saturation"])
            factor = random.uniform(0.8, 1.2)
            if j_type == "brightness":
                frames = frames + (factor - 1.0)
            elif j_type == "contrast":
                mean = frames.mean(dim=(-2, -1), keepdim=True)
                frames = (frames - mean) * factor + mean
            else:
                gray = frames.mean(dim=1, keepdim=True)
                frames = gray + (frames - gray) * factor
        return torch.clamp(frames, 0.0, 1.0)

    def _center_crop(self, frames: torch.Tensor, size: int) -> torch.Tensor:
        """Center-crop frames to target size, with resize if smaller."""
        B, T, C, H, W = frames.shape
        if H <= size and W <= size:
            padded = torch.zeros(B, T, C, size, size, device=frames.device, dtype=frames.dtype)
            y_off = (size - H) // 2
            x_off = (size - W) // 2
            padded[:, :, :, y_off : y_off + H, x_off : x_off + W] = frames
            return padded

        y_start = (H - size) // 2
        x_start = (W - size) // 2
        return frames[:, :, :, y_start : y_start + size, x_start : x_start + size]


class MixupAugmentation:
    """
    Mixup augmentation for multimodal video-text pairs.

    Blends two samples (frames + text + labels) with a lambda coefficient
    drawn from a Beta distribution. Keeps text tokens discrete (hard mix).

    Args:
        alpha: Beta distribution parameter (default 0.2).

    Example:
        >>> mixup = MixupAugmentation(alpha=0.2)
        >>> mixed_frames, mixed_text, lam, y1, y2 = mixup(
        ...     frames1, text1, labels1_b, labels1_a,
        ...     frames2, text2, labels2_b, labels2_a,
        ... )
    """

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha

    def __call__(
        self,
        frames1: torch.Tensor,
        text1: dict,
        behavior_labels1: torch.Tensor,
        alert_labels1: torch.Tensor,
        frames2: torch.Tensor,
        text2: dict,
        behavior_labels2: torch.Tensor,
        alert_labels2: torch.Tensor,
    ) -> tuple[torch.Tensor, dict, float, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Generate a mixed sample from two inputs.

        Args:
            frames1: First sample frames (B, T, C, H, W).
            text1: First sample text tokens dict.
            behavior_labels1: First sample behavior labels.
            alert_labels1: First sample alert labels.
            frames2: Second sample frames.
            text2: Second sample text tokens dict.
            behavior_labels2: Second sample behavior labels.
            alert_labels2: Second sample alert labels.

        Returns:
            Tuple of (mixed_frames, mixed_text, lambda, labels_tuple).
        """
        lam = self._sample_lambda(frames1.size(0))
        mixed_frames = lam * frames1 + (1.0 - lam) * frames2

        mixed_text = {}
        if "input_ids" in text1 and "input_ids" in text2:
            if torch.rand(1).item() > 0.5:
                mixed_text["input_ids"] = text1["input_ids"]
            else:
                mixed_text["input_ids"] = text2["input_ids"]
        else:
            mixed_text = text1

        if "attention_mask" in text1:
            mixed_text["attention_mask"] = text1["attention_mask"]

        return (
            mixed_frames,
            mixed_text,
            lam,
            (behavior_labels1, alert_labels1, behavior_labels2, alert_labels2),
        )

    def _sample_lambda(self, batch_size: int) -> float:
        """Sample lambda from Beta distribution."""
        if self.alpha > 0:
            return float(torch.distributions.Beta(self.alpha, self.alpha).sample())
        return 1.0
