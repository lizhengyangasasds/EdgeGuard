"""
EdgeGuard: Edge Inference Demo

Demonstrates the full edge-side inference pipeline with synthetic data.
Tests the multimodal network forward pass and simulates real-time inference.
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch

from model.multimodal_net import EdgeGuardMultimodalNet, EdgeGuardConfig
from data.video_processor import VideoProcessor
from data.text_processor import TextProcessor, AlertTextGenerator
from evaluation.monitor import MetricsCollector


BEHAVIOR_CLASSES = ["Fighting", "Falling", "Climbing", "Loitering", "Retrograde", "Gathering", "Normal"]
ALERT_CLASSES = ["Intrusion", "Fault", "Violation", "Anomaly", "Normal"]


def generate_synthetic_clip(
    clip_length: int = 16,
    frame_size: int = 224,
) -> torch.Tensor:
    """Generate a synthetic video clip."""
    clip = np.random.randn(clip_length, 3, frame_size, frame_size).astype(np.float32)
    clip = (clip - clip.min()) / (clip.max() - clip.min() + 1e-8)
    return torch.from_numpy(clip).unsqueeze(0)


def generate_synthetic_text(
    text_processor: TextProcessor,
    batch_size: int = 1,
) -> dict[str, torch.Tensor]:
    """Generate synthetic alert text."""
    templates = [
        "Unauthorized entry detected in zone A1",
        "Person climbing fence at perimeter",
        "Abnormal crowd gathering detected",
        "Vehicle moving in reverse direction",
        "Object left unattended for 10 minutes",
        "Camera connection lost",
        "Motion detected in restricted area",
        "Security patrol deviation alert",
    ]
    text = random.choice(templates)
    tokens = text_processor.tokenize([text], return_tensors="pt")
    tokens = {k: v for k, v in tokens.items()}
    return tokens


def run_edge_demo(
    num_iterations: int = 20,
    use_cuda: bool = True,
    print_details: bool = True,
) -> dict:
    """
    Run the edge inference demo.

    Args:
        num_iterations: Number of inference iterations.
        use_cuda: Use GPU if available.
        print_details: Print detailed per-iteration results.

    Returns:
        Dictionary of demo results and statistics.
    """
    device = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
    print(f"\n{'='*60}")
    print(f"EdgeGuard Edge Demo")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Iterations: {num_iterations}")
    print(f"{'='*60}\n")

    config = EdgeGuardConfig()
    model = EdgeGuardMultimodalNet(config)
    model.to(device)
    model.eval()

    print("Model Architecture Summary:")
    model.print_trainable_summary()
    print()

    video_processor = VideoProcessor(clip_length=16, frame_size=224)
    text_processor = TextProcessor(tokenizer_name="distilbert-base-uncased")
    metrics_collector = MetricsCollector(window_size=60)

    latencies = []
    behavior_preds = []
    alert_preds = []

    print(f"{'Iter':>5} | {'Behavior':<12} | {'Conf':>6} | {'Alert':<10} | {'Conf':>6} | {'Lat(ms)':>8} | {'FPS':>6}")
    print("-" * 75)

    with torch.no_grad():
        for i in range(num_iterations):
            frames = generate_synthetic_clip(16, 224)
            text_tokens = generate_synthetic_text(text_processor)

            frames = frames.to(device)
            text_ids = text_tokens["input_ids"].to(device)
            attention_mask = text_tokens.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            start_time = time.perf_counter()
            behavior_logits, alert_logits = model(frames, text_ids, attention_mask)
            latency_ms = (time.perf_counter() - start_time) * 1000

            behavior_probs = torch.softmax(behavior_logits, dim=-1)
            alert_probs = torch.softmax(alert_logits, dim=-1)

            b_class = behavior_probs.argmax(dim=-1).item()
            b_conf = behavior_probs.max(dim=-1).values.item()
            a_class = alert_probs.argmax(dim=-1).item()
            a_conf = alert_probs.max(dim=-1).values.item()

            fps = 1.0 / (latency_ms / 1000) if latency_ms > 0 else 0

            latencies.append(latency_ms)
            behavior_preds.append(b_class)
            alert_preds.append(a_class)

            metrics_collector.record(
                latency_ms=latency_ms,
                fps=fps,
                memory_mb=256,
                batch_size=1,
                behavior_class=b_class,
                behavior_confidence=b_conf,
                alert_class=a_class,
                alert_confidence=a_conf,
            )

            if print_details:
                print(f"{i+1:5d} | {BEHAVIOR_CLASSES[b_class]:<12} | {b_conf:>6.2f} | "
                      f"{ALERT_CLASSES[a_class]:<10} | {a_conf:>6.2f} | {latency_ms:>8.1f} | {fps:>6.1f}")

    summary = metrics_collector.get_summary()

    print()
    print(f"{'='*60}")
    print("Demo Results Summary")
    print(f"{'='*60}")
    print(f"Total iterations:      {num_iterations}")
    print(f"Avg latency:           {summary['latency']['mean_ms']:.1f} ms")
    print(f"P95 latency:           {summary['latency']['p95_ms']:.1f} ms")
    print(f"P99 latency:           {summary['latency']['p99_ms']:.1f} ms")
    print(f"Mean FPS:              {summary['throughput']['mean_fps']:.1f}")
    print(f"Min FPS:               {summary['throughput']['min_fps']:.1f}")
    print()
    print("Behavior Prediction Distribution:")
    for idx, name in enumerate(BEHAVIOR_CLASSES):
        count = behavior_preds.count(idx)
        pct = count / len(behavior_preds) * 100
        print(f"  {name:<12}: {count:3d} ({pct:5.1f}%)")
    print()
    print("Alert Prediction Distribution:")
    for idx, name in enumerate(ALERT_CLASSES):
        count = alert_preds.count(idx)
        pct = count / len(alert_preds) * 100
        print(f"  {name:<10}: {count:3d} ({pct:5.1f}%)")
    print()
    print("Performance Targets:")
    latency_pass = summary['latency']['mean_ms'] < 85
    fps_pass = summary['throughput']['mean_fps'] >= 15
    print(f"  Latency <85ms:      {'PASS' if latency_pass else 'FAIL'} ({summary['latency']['mean_ms']:.1f}ms)")
    print(f"  FPS >= 15:          {'PASS' if fps_pass else 'FAIL'} ({summary['throughput']['mean_fps']:.1f})")
    print(f"{'='*60}\n")

    return {
        "iterations": num_iterations,
        "latency": summary["latency"],
        "throughput": summary["throughput"],
        "behavior_preds": behavior_preds,
        "alert_preds": alert_preds,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EdgeGuard Edge Demo")
    parser.add_argument("--iterations", type=int, default=20, help="Number of inference iterations")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-iteration output")

    args = parser.parse_args()

    run_edge_demo(
        num_iterations=args.iterations,
        use_cuda=not args.cpu,
        print_details=not args.quiet,
    )
