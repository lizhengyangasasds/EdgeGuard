"""
EdgeGuard: Video Processor Module

Video frame extraction, normalization, clip generation, and preprocessing
for the multimodal inference pipeline.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np


class VideoProcessor:
    """
    Process raw video frames for EdgeGuard inference.

    Handles frame normalization (ImageNet stats), clip generation with
    configurable stride, and batch-level preprocessing.

    Args:
        clip_length: Number of frames per clip (default 16).
        frame_size: Target frame size (HxW, default 224).
        stride: Frame stride when generating clips (default 1).
        normalize: Whether to apply ImageNet normalization.

    Example:
        >>> processor = VideoProcessor(clip_length=16, frame_size=224)
        >>> frames = np.random.randint(0, 255, (64, 224, 224, 3), dtype=np.uint8)
        >>> clips = list(processor.generate_clips(frames))
        >>> normalized = processor.normalize_frames(frames)
    """

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        clip_length: int = 16,
        frame_size: int = 224,
        stride: int = 1,
        normalize: bool = True,
    ) -> None:
        self.clip_length = clip_length
        self.frame_size = frame_size
        self.stride = stride
        self.normalize = normalize

    def normalize_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        Normalize frames using ImageNet mean/std and convert HWC -> CHW.

        Args:
            frames: Raw frames of shape (T, H, W, C) or (H, W, C).

        Returns:
            Normalized frames of shape (T, C, H, W) as float32 in [-2, +2] range.
        """
        if frames.ndim == 3:
            frames = frames[np.newaxis, ...]

        frames = frames.astype(np.float32) / 255.0

        if frames.shape[-1] != 3:
            frames = frames[..., ::-1]

        T, H, W, C = frames.shape
        if H != self.frame_size or W != self.frame_size:
            frames = self._resize_frames(frames, (self.frame_size, self.frame_size))

        frames = frames.transpose(0, 3, 1, 2)

        if self.normalize:
            mean = self.IMAGENET_MEAN.reshape(1, 3, 1, 1)
            std = self.IMAGENET_STD.reshape(1, 3, 1, 1)
            frames = (frames - mean) / std

        return frames

    def _resize_frames(self, frames: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        """Resize frames to target size using nearest-neighbor (fast, no interpolation libs needed)."""
        try:
            import cv2
            out = []
            for f in frames:
                resized = cv2.resize(f, target_size, interpolation=cv2.INTER_LINEAR)
                out.append(resized)
            return np.stack(out, axis=0)
        except ImportError:
            T, H, W, C = frames.shape
            th, tw = target_size
            out = np.zeros((T, th, tw, C), dtype=frames.dtype)
            for i in range(T):
                y_ratio = H / th
                x_ratio = W / tw
                for y in range(th):
                    for x in range(tw):
                        src_y = int(y * y_ratio)
                        src_x = int(x * x_ratio)
                        src_y = min(src_y, H - 1)
                        src_x = min(src_x, W - 1)
                        out[i, y, x] = frames[i, src_y, src_x]
            return out

    def generate_clips(self, frames: np.ndarray) -> Iterator[np.ndarray]:
        """
        Generate consecutive clips from a frame sequence.

        Args:
            frames: Frame array of shape (T, H, W, C).

        Yields:
            Normalized clips of shape (clip_length, H, W, C) as float32.
        """
        T = frames.shape[0]
        for start in range(0, T - self.clip_length + 1, self.stride):
            clip = frames[start : start + self.clip_length]
            normalized = self.normalize_frames(clip)
            # Yield in HWC format for model consumption
            yield normalized.transpose(0, 2, 3, 1)

    def process_video_file(self, video_path: str) -> list[np.ndarray]:
        """
        Read a video file and return normalized clips.

        Args:
            video_path: Path to the video file.

        Returns:
            List of normalized clips, each of shape (clip_length, C, H, W).

        Raises:
            ImportError: If neither cv2 nor decord is installed.
        """
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            frames = np.stack(frames, axis=0)
            return list(self.generate_clips(frames))
        except ImportError:
            try:
                import decord
                decord.bridge.set_bridge("native")
                vr = decord.VideoReader(video_path)
                frames = vr.get_batch(range(len(vr))).asnumpy()
                return list(self.generate_clips(frames))
            except ImportError:
                raise ImportError(
                    "Reading video files requires either cv2 or decord. "
                    "Install with: pip install opencv-python-headless decord"
                )
