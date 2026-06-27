"""
EdgeGuard: Classification Head Module

Dual-task classification heads for behavior recognition and alert classification.
Each head uses a shared feature representation with independent output layers.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Classification head with configurable dropout and activation.

    Args:
        input_dim: Input feature dimension.
        num_classes: Number of output classes.
        dropout: Dropout probability before classification.
        activation: Activation function ("relu", "gelu", "silu").
    """

    BEHAVIOR_CLASSES = ["fighting", "falling", "climbing", "loitering", "retrograde", "gathering", "normal"]
    ALERT_CLASSES = ["intrusion", "fault", "violation", "anomaly", "normal"]

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.3,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        activation_map = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        act_fn = activation_map.get(activation, nn.ReLU)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim // 2),
            act_fn(),
            nn.Dropout(dropout / 2),
            nn.Linear(input_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize classification head weights."""
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify input features.

        Args:
            x: Input features of shape (B, input_dim).

        Returns:
            logits: Unnormalized class scores of shape (B, num_classes).
        """
        return self.head(x)


class BehaviorClassifier(nn.Module):
    """
    Behavior classification head for anomaly detection.

    Classifies detected behaviors into 7 categories:
    fighting, falling, climbing, loitering, retrograde, gathering, normal

    Args:
        input_dim: Input feature dimension.
        num_classes: Number of behavior classes (default 7).
        dropout: Dropout probability.
    """

    BEHAVIOR_CLASSES = ["fighting", "falling", "climbing", "loitering", "retrograde", "gathering", "normal"]

    def __init__(
        self,
        input_dim: int = 640,
        num_classes: int = 7,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.class_names = self.BEHAVIOR_CLASSES
        self.classifier = ClassificationHead(input_dim, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify behavior from fused features."""
        return self.classifier(x)

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def get_top_k_predictions(
        self,
        x: torch.Tensor,
        k: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return top-k predictions with probabilities."""
        probs = self.get_probabilities(x)
        top_probs, top_indices = torch.topk(probs, k, dim=-1)
        return top_probs, top_indices


class AlertClassifier(nn.Module):
    """
    Alert classification head for text-based alarm categorization.

    Classifies alerts into 5 categories:
    intrusion, fault, violation, anomaly, normal

    Args:
        input_dim: Input feature dimension.
        num_classes: Number of alert classes (default 5).
        dropout: Dropout probability.
    """

    ALERT_CLASSES = ["intrusion", "fault", "violation", "anomaly", "normal"]

    def __init__(
        self,
        input_dim: int = 640,
        num_classes: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.class_names = self.ALERT_CLASSES
        self.classifier = ClassificationHead(input_dim, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify alert type from fused features."""
        return self.classifier(x)

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def get_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """Return confidence score (max probability)."""
        probs = self.get_probabilities(x)
        return probs.max(dim=-1).values
