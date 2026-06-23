"""
EdgeGuard: Dataset Module

Dataset classes for multimodal security data:
- Video clips (frames + behavior labels)
- Alert text (text tokens + alert labels)
- Combined multimodal dataset
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class VideoFrameDataset(Dataset):
    """
    Dataset of video clips with behavior labels.

    For demo/testing, generates synthetic data when real video is unavailable.

    Args:
        root: Root directory of video data.
        clip_length: Number of frames per clip.
        frame_size: Spatial resolution (H=W).
        split: "train" or "val".
        transform: Optional transforms for frames.
        generate_synthetic: Generate random data if no real data exists.
        num_samples: Number of synthetic samples if generating.
    """

    BEHAVIOR_CLASSES = ["fighting", "falling", "climbing", "loitering", "retrograde", "gathering", "normal"]

    def __init__(
        self,
        root: str = "data/video",
        clip_length: int = 16,
        frame_size: int = 224,
        split: str = "train",
        transform: Optional[Any] = None,
        generate_synthetic: bool = True,
        num_samples: int = 100,
    ) -> None:
        self.root = Path(root)
        self.clip_length = clip_length
        self.frame_size = frame_size
        self.split = split
        self.transform = transform
        self.num_classes = len(self.BEHAVIOR_CLASSES)

        self.data = self._load_data(generate_synthetic, num_samples)

    def _load_data(self, generate_synthetic: bool, num_samples: int) -> list[dict]:
        """Load real data or generate synthetic samples."""
        if self.root.exists() and any(self.root.iterdir()):
            return self._load_real_data()
        elif generate_synthetic:
            return self._generate_synthetic(num_samples)
        else:
            raise FileNotFoundError(f"No data at {self.root} and generate_synthetic=False")

    def _load_real_data(self) -> list[dict]:
        """Load data from disk."""
        data = []
        for cls_idx, cls_name in enumerate(self.BEHAVIOR_CLASSES):
            cls_dir = self.root / self.split / cls_name
            if cls_dir.exists():
                for clip_file in cls_dir.glob("*.pt"):
                    data.append({"path": str(clip_file), "label": cls_idx})
        return data

    def _generate_synthetic(self, num_samples: int) -> list[dict]:
        """Generate synthetic video clip data."""
        data = []
        for i in range(num_samples):
            label = random.randint(0, self.num_classes - 1)
            data.append({"synthetic": True, "label": label, "sample_id": i})
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get a video clip and its label."""
        item = self.data[idx]

        if item.get("synthetic", False):
            frames = torch.randn(
                self.clip_length, 3, self.frame_size, self.frame_size
            )
            frames = (frames - frames.min()) / (frames.max() - frames.min() + 1e-8)
        else:
            frames = torch.load(item["path"])

        if self.transform:
            frames = self.transform(frames)

        return frames, item["label"]


class AlertTextDataset(Dataset):
    """
    Dataset of alert text descriptions with classification labels.

    Args:
        text_file: Path to JSON file with text data or synthetic config.
        tokenizer: HuggingFace tokenizer for text encoding.
        max_length: Maximum token sequence length.
        generate_synthetic: Generate random text if no file exists.
        num_samples: Number of synthetic samples.
    """

    ALERT_CLASSES = ["intrusion", "fault", "violation", "anomaly", "normal"]

    SYNTHETIC_TEMPLATES = [
        "Unauthorized entry detected at zone {zone}",
        "Camera {cam} connection lost",
        "Person climbing fence at perimeter",
        "Unusual crowd gathering in {area}",
        "Vehicle moving in reverse direction",
        "Object left unattended for {time} minutes",
        "Perimeter breach detected",
        "Motion sensor triggered in restricted area",
        "Door forced open alarm",
        "Safety helmet not detected",
        "Worker fall detected in construction zone",
        "Fire alarm triggered in building {bldg}",
        "Restricted zone access attempt",
        "Abnormal temperature detected",
        "Trespasser detected after hours",
        "Vehicle parked in no-parking zone",
        "Abnormal sound level detected",
        "Security guard patrol deviation",
        "Baggage left unattended at terminal",
        "Perimeter fence damage detected",
    ]

    ZONES = ["A1", "B2", "C3", "D4", "E5"]
    CAMS = [f"cam{i:03d}" for i in range(1, 21)]
    AREAS = ["entrance", "parking_lot", "loading_dock", "rooftop", "stairwell"]
    TIMES = ["5", "10", "15", "30"]
    BLDGS = [f"B{i}" for i in range(1, 10)]

    def __init__(
        self,
        text_file: str = "data/alerts.json",
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        generate_synthetic: bool = True,
        num_samples: int = 100,
    ) -> None:
        self.text_file = Path(text_file)
        self.max_length = max_length
        self.num_classes = len(self.ALERT_CLASSES)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.data = self._load_data(generate_synthetic, num_samples)

    def _load_data(self, generate_synthetic: bool, num_samples: int) -> list[dict]:
        """Load real text data or generate synthetic."""
        if self.text_file.exists():
            return self._load_real_data()
        elif generate_synthetic:
            return self._generate_synthetic(num_samples)
        else:
            raise FileNotFoundError(f"No data at {self.text_file}")

    def _load_real_data(self) -> list[dict]:
        """Load from JSON file."""
        with open(self.text_file, "r", encoding="utf-8") as f:
            items = json.load(f)
        return items

    def _generate_synthetic(self, num_samples: int) -> list[dict]:
        """Generate synthetic alert text data."""
        data = []
        templates = self.SYNTHETIC_TEMPLATES
        classes = self.ALERT_CLASSES

        for i in range(num_samples):
            template = templates[i % len(templates)]
            text = template.format(
                zone=random.choice(self.ZONES),
                cam=random.choice(self.CAMS),
                area=random.choice(self.AREAS),
                time=random.choice(self.TIMES),
                bldg=random.choice(self.BLDGS),
            )
            label = i % self.num_classes
            data.append({"text": text, "label": label})

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], int]:
        """Tokenize text and return tokens with label."""
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        return encoding, item["label"]


class MultimodalDataset(Dataset):
    """
    Combined multimodal dataset for video + text.

    Aligns video clips with corresponding alert text by index.
    Both use synthetic data by default for demonstration.

    Args:
        video_dataset: Video clip dataset.
        text_dataset: Alert text dataset.
        align_by_label: Align video behavior with text alert labels.
    """

    def __init__(
        self,
        video_dataset: Optional[VideoFrameDataset] = None,
        text_dataset: Optional[AlertTextDataset] = None,
        align_by_label: bool = False,
        num_samples: int = 100,
        clip_length: int = 16,
        frame_size: int = 224,
        max_text_length: int = 128,
        tokenizer_name: str = "distilbert-base-uncased",
    ) -> None:
        self.align_by_label = align_by_label

        if video_dataset is not None:
            self.video_dataset = video_dataset
        else:
            self.video_dataset = VideoFrameDataset(
                generate_synthetic=True,
                num_samples=num_samples,
                clip_length=clip_length,
                frame_size=frame_size,
            )

        if text_dataset is not None:
            self.text_dataset = text_dataset
        else:
            self.text_dataset = AlertTextDataset(
                generate_synthetic=True,
                num_samples=num_samples,
                max_length=max_text_length,
                tokenizer_name=tokenizer_name,
            )

        self.num_samples = min(len(self.video_dataset), len(self.text_dataset))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], int, int]:
        """
        Get a paired video-text sample.

        Returns:
            Tuple of (frames, text_tokens, behavior_label, alert_label).
        """
        frames, behavior_label = self.video_dataset[idx]
        text_tokens, alert_label = self.text_dataset[idx]

        if self.align_by_label:
            alert_label = behavior_label % self.text_dataset.num_classes

        return frames, text_tokens, behavior_label, alert_label
