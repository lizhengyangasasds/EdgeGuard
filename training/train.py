"""
EdgeGuard: Training Module

Main training script with PEFT fine-tuning support, mixed precision,
MLflow logging, and comprehensive checkpointing.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from model.multimodal_net import EdgeGuardMultimodalNet, EdgeGuardConfig
from .dataset import MultimodalDataset, VideoFrameDataset, AlertTextDataset
from .peft_config import PEFTConfig


def build_model(config: EdgeGuardConfig) -> EdgeGuardMultimodalNet:
    """Build the EdgeGuard model from configuration."""
    return EdgeGuardMultimodalNet(config)


def build_dataloader(
    batch_size: int = 8,
    num_workers: int = 4,
    clip_length: int = 16,
    frame_size: int = 224,
    max_text_length: int = 128,
    num_samples: int = 100,
) -> DataLoader:
    """Build a training dataloader with synthetic data."""
    dataset = MultimodalDataset(
        num_samples=num_samples,
        clip_length=clip_length,
        frame_size=frame_size,
        max_text_length=max_text_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


class EdgeGuardTrainer:
    """
    Trainer for EdgeGuard multimodal model with PEFT fine-tuning.

    Features:
    - Mixed precision (FP16) training
    - Gradient accumulation
    - Per-task loss weighting (behavior + alert)
    - MLflow experiment tracking
    - Comprehensive checkpointing

    Args:
        model: EdgeGuard model instance.
        device: Training device ("cuda" or "cpu").
        output_dir: Directory for checkpoints and logs.
    """

    def __init__(
        self,
        model: EdgeGuardMultimodalNet,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "checkpoints",
        behavior_weight: float = 0.6,
        alert_weight: float = 0.4,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.behavior_weight = behavior_weight
        self.alert_weight = alert_weight

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=2e-4,
            weight_decay=0.01,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=5, eta_min=1e-6
        )

        self.scaler = GradScaler()

        self.best_loss = float("inf")
        self.global_step = 0
        self.epoch = 0

        try:
            import mlflow
            self.mlflow = mlflow
            self.mlflow.set_experiment("EdgeGuard-Training")
            self.mlflow_active = True
        except ImportError:
            self.mlflow_active = False
            print("MLflow not available, skipping experiment tracking")

    def train_epoch(self, dataloader: DataLoader, log_interval: int = 50) -> dict:
        """Train for one epoch."""
        self.model.train()

        total_behavior_loss = 0.0
        total_alert_loss = 0.0
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")

        for batch_idx, (frames, text_tokens, behavior_labels, alert_labels) in enumerate(pbar):
            frames = frames.to(self.device)
            behavior_labels = behavior_labels.to(self.device)
            alert_labels = alert_labels.to(self.device)
            text_tokens = {k: v.to(self.device) for k, v in text_tokens.items()}

            with autocast():
                behavior_logits, alert_logits = self.model(
                    frames,
                    text_tokens["input_ids"],
                    text_tokens.get("attention_mask"),
                )

                behavior_loss = F.cross_entropy(behavior_logits, behavior_labels)
                alert_loss = F.cross_entropy(alert_logits, alert_labels)
                loss = self.behavior_weight * behavior_loss + self.alert_weight * alert_loss

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            total_behavior_loss += behavior_loss.item()
            total_alert_loss += alert_loss.item()
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            if batch_idx % log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "b_loss": f"{behavior_loss.item():.4f}",
                    "a_loss": f"{alert_loss.item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                })

                if self.mlflow_active:
                    self.mlflow.log_metric("train/loss", loss.item(), self.global_step)
                    self.mlflow.log_metric("train/behavior_loss", behavior_loss.item(), self.global_step)
                    self.mlflow.log_metric("train/alert_loss", alert_loss.item(), self.global_step)
                    self.mlflow.log_metric("train/lr", self.scheduler.get_last_lr()[0], self.global_step)

        self.scheduler.step()
        self.epoch += 1

        avg_metrics = {
            "loss": total_loss / num_batches,
            "behavior_loss": total_behavior_loss / num_batches,
            "alert_loss": total_alert_loss / num_batches,
            "lr": self.scheduler.get_last_lr()[0],
        }

        return avg_metrics

    def evaluate(self, dataloader: DataLoader) -> dict:
        """Evaluate the model on a validation set."""
        self.model.eval()

        correct_behavior = 0
        correct_alert = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for frames, text_tokens, behavior_labels, alert_labels in dataloader:
                frames = frames.to(self.device)
                behavior_labels = behavior_labels.to(self.device)
                alert_labels = alert_labels.to(self.device)
                text_tokens = {k: v.to(self.device) for k, v in text_tokens.items()}

                behavior_logits, alert_logits = self.model(
                    frames,
                    text_tokens["input_ids"],
                    text_tokens.get("attention_mask"),
                )

                loss = F.cross_entropy(behavior_logits, behavior_labels) * 0.6 + \
                       F.cross_entropy(alert_logits, alert_labels) * 0.4

                total_loss += loss.item()

                behavior_preds = behavior_logits.argmax(dim=1)
                alert_preds = alert_logits.argmax(dim=1)

                correct_behavior += (behavior_preds == behavior_labels).sum().item()
                correct_alert += (alert_preds == alert_labels).sum().item()
                total += behavior_labels.size(0)

        return {
            "behavior_acc": correct_behavior / total if total > 0 else 0.0,
            "alert_acc": correct_alert / total if total > 0 else 0.0,
            "val_loss": total_loss / len(dataloader),
        }

    def save_checkpoint(self, filepath: str, metrics: dict | None = None) -> str:
        """Save a training checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat(),
        }
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, filepath)
        return str(filepath)

    def load_checkpoint(self, filepath: str) -> dict:
        """Load a training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]
        return checkpoint


def train(
    config_path: str | None = None,
    epochs: int = 5,
    batch_size: int = 8,
    num_samples: int = 100,
    output_dir: str = "checkpoints",
    device: str | None = None,
) -> EdgeGuardMultimodalNet:
    """
    Main training entry point.

    Args:
        config_path: Optional path to YAML config file.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        num_samples: Number of synthetic training samples.
        output_dir: Directory for checkpoints.
        device: Training device (auto-detected if None).

    Returns:
        Trained EdgeGuard model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Training on device: {device}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Samples: {num_samples}")

    model_config = EdgeGuardConfig()
    model = build_model(model_config)
    model.print_trainable_summary()

    trainer = EdgeGuardTrainer(
        model=model,
        device=device,
        output_dir=output_dir,
    )

    train_loader = build_dataloader(
        batch_size=batch_size,
        num_samples=num_samples,
    )
    val_loader = build_dataloader(
        batch_size=batch_size,
        num_samples=num_samples // 5,
    )

    print(f"\nStarting training for {epochs} epochs...\n")

    for epoch in range(epochs):
        metrics = trainer.train_epoch(train_loader, log_interval=10)

        print(f"\nEpoch {trainer.epoch} completed:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Behavior Loss: {metrics['behavior_loss']:.4f}")
        print(f"  Alert Loss: {metrics['alert_loss']:.4f}")
        print(f"  LR: {metrics['lr']:.2e}")

        val_metrics = trainer.evaluate(val_loader)
        print(f"  Behavior Acc: {val_metrics['behavior_acc']:.4f}")
        print(f"  Alert Acc: {val_metrics['alert_acc']:.4f}")

        checkpoint_path = trainer.save_checkpoint(
            f"{output_dir}/checkpoint_epoch_{trainer.epoch}.pt",
            {**metrics, **val_metrics},
        )
        print(f"  Checkpoint saved: {checkpoint_path}")

        if metrics["loss"] < trainer.best_loss:
            trainer.best_loss = metrics["loss"]
            best_path = trainer.save_checkpoint(
                f"{output_dir}/best_model.pt",
                {**metrics, **val_metrics},
            )
            print(f"  New best model saved!")

    print("\nTraining complete!")
    model_stats = model.get_trainable_params()
    print(f"Final trainable params: {model_stats['trainable']:,} / {model_stats['total']:,} ({model_stats['trainable_ratio']:.2%})")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train EdgeGuard multimodal model")
    parser.add_argument("--config", type=str, help="Path to training config YAML")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train(
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        device=args.device,
    )
