"""
EdgeGuard: Cloud Training Demo

Demonstrates the cloud-side training pipeline with synthetic data.
Tests the full PEFT fine-tuning loop with LoRA and Adapter LoRA.
"""
from __future__ import annotations

import argparse
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, ".")
from model.multimodal_net import EdgeGuardMultimodalNet, EdgeGuardConfig
from training.dataset import MultimodalDataset
from training.peft_config import PEFTConfig


def run_cloud_demo(
    epochs: int = 3,
    batch_size: int = 4,
    num_samples: int = 50,
    lr: float = 2e-4,
    use_cuda: bool = True,
) -> dict:
    """
    Run the cloud training demo.

    Args:
        epochs: Number of training epochs.
        batch_size: Training batch size.
        num_samples: Number of synthetic training samples.
        lr: Learning rate.
        use_cuda: Use GPU if available.

    Returns:
        Dictionary of training statistics.
    """
    device = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
    print(f"\n{'='*60}")
    print(f"EdgeGuard Cloud Training Demo")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Samples: {num_samples}")
    print(f"Learning rate: {lr}")
    print(f"{'='*60}\n")

    config = EdgeGuardConfig()
    model = EdgeGuardMultimodalNet(config)
    model.to(device)

    print("Initial model parameter statistics:")
    model.print_trainable_summary()
    print()

    dataset = MultimodalDataset(
        num_samples=num_samples,
        clip_length=16,
        frame_size=224,
        max_text_length=64,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    epoch_stats = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_behavior_loss = 0.0
        epoch_alert_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, (frames, text_tokens, behavior_labels, alert_labels) in enumerate(pbar):
            frames = frames.to(device)
            behavior_labels = behavior_labels.to(device)
            alert_labels = alert_labels.to(device)
            text_ids = text_tokens["input_ids"].to(device)
            attention_mask = text_tokens.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            optimizer.zero_grad()

            behavior_logits, alert_logits = model(frames, text_ids, attention_mask)

            behavior_loss = F.cross_entropy(behavior_logits, behavior_labels)
            alert_loss = F.cross_entropy(alert_logits, alert_labels)
            loss = 0.6 * behavior_loss + 0.4 * alert_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_behavior_loss += behavior_loss.item()
            epoch_alert_loss += alert_loss.item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "b_loss": f"{behavior_loss.item():.4f}",
                "a_loss": f"{alert_loss.item():.4f}",
            })

        scheduler.step()

        avg_loss = epoch_loss / num_batches
        avg_b_loss = epoch_behavior_loss / num_batches
        avg_a_loss = epoch_alert_loss / num_batches

        stats = model.get_trainable_params()
        print(f"\nEpoch {epoch+1} completed:")
        print(f"  Avg Loss:       {avg_loss:.4f}")
        print(f"  Behavior Loss:  {avg_b_loss:.4f}")
        print(f"  Alert Loss:     {avg_a_loss:.4f}")
        print(f"  LR:             {scheduler.get_last_lr()[0]:.2e}")
        print(f"  Trainable:      {stats['trainable']:,} / {stats['total']:,} ({stats['trainable_ratio']:.2%})")

        epoch_stats.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "behavior_loss": avg_b_loss,
            "alert_loss": avg_a_loss,
            "lr": scheduler.get_last_lr()[0],
        })

    print(f"\n{'='*60}")
    print("Training Complete")
    print(f"{'='*60}")
    print(f"Final trainable parameter ratio: {model.get_trainable_params()['trainable_ratio']:.2%}")
    print(f"Target: < 1%")
    print(f"Status: {'PASS' if model.get_trainable_params()['trainable_ratio'] < 0.01 else 'FAIL'}")

    return {
        "epochs": epoch_stats,
        "final_trainable_ratio": model.get_trainable_params()["trainable_ratio"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EdgeGuard Cloud Training Demo")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    run_cloud_demo(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        lr=args.lr,
        use_cuda=not args.cpu,
    )
