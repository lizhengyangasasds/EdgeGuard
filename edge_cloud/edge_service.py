"""
EdgeGuard: Edge Service Module

Main edge-side service that orchestrates video inference,
MQTT communication, and model hot-updates.
"""
from __future__ import annotations

import os
import sys
import signal
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from model.multimodal_net import EdgeGuardMultimodalNet, EdgeGuardConfig
from edge_cloud.mqtt_client import EdgeGuardMQTTClient, MQTTConfig
from edge_cloud.model_registry import ModelRegistry


class EdgeService:
    """
    Main edge service for EdgeGuard anomaly detection.

    Orchestrates:
    - TensorRT/PyTorch inference on video streams
    - MQTT communication with cloud
    - Model hot-update on receiving new versions
    - Real-time performance monitoring

    Args:
        device_id: Unique edge device identifier.
        model_path: Path to TensorRT engine or PyTorch checkpoint.
        mqtt_config: MQTT configuration.
        alert_threshold: Confidence threshold for anomaly alerts.
        use_trt: Use TensorRT engine if True, PyTorch if False.
    """

    BEHAVIOR_NAMES = ["fighting", "falling", "climbing", "loitering", "retrograde", "gathering", "normal"]
    ALERT_NAMES = ["intrusion", "fault", "violation", "anomaly", "normal"]

    def __init__(
        self,
        device_id: str = "edge-001",
        model_path: Optional[str] = None,
        mqtt_config: Optional[MQTTConfig] = None,
        alert_threshold: float = 0.75,
        use_trt: bool = False,
    ) -> None:
        self.device_id = device_id
        self.alert_threshold = alert_threshold
        self.use_trt = use_trt
        self.running = False

        self.model = self._load_model(model_path)
        self.registry = ModelRegistry()

        if model_path and Path(model_path).exists():
            self.registry.register_model(
                model_path=model_path,
                version="1.0.0",
                precision="fp16" if use_trt else "fp32",
                framework="tensorrt" if use_trt else "pytorch",
            )
            self.registry.set_active_version("1.0.0")

        self.mqtt = EdgeGuardMQTTClient(
            config=mqtt_config,
            device_id=device_id,
            on_model_received=self._on_model_received,
            on_alert_triggered=self._on_alert_triggered,
        )

        self._init_inference_engine(model_path)
        self._setup_signal_handlers()

        self.inference_count = 0
        self.total_latency_ms = 0.0
        self.fps = 0.0

    def _load_model(self, model_path: Optional[str]) -> EdgeGuardMultimodalNet:
        """Load PyTorch model."""
        config = EdgeGuardConfig()
        model = EdgeGuardMultimodalNet(config)
        model.eval()

        if model_path and Path(model_path).exists() and model_path.endswith(".pt"):
            checkpoint = torch.load(model_path, map_location="cpu")
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            print(f"[Service] Loaded checkpoint from {model_path}")

        return model

    def _init_inference_engine(self, model_path: Optional[str]) -> None:
        """Initialize inference engine (PyTorch or TensorRT)."""
        if self.use_trt and model_path and Path(model_path).exists():
            try:
                from deployment.trt_inference import TensorRTInference
                self.trt_engine = TensorRTInference(model_path)
                self.trt_engine.warmup(10)
                self.inference_fn = self._inference_trt
                print(f"[Service] TensorRT engine loaded: {model_path}")
            except Exception as e:
                print(f"[Service] Could not load TensorRT engine ({e}), falling back to PyTorch")
                self.inference_fn = self._inference_pytorch
                self.trt_engine = None
        else:
            self.inference_fn = self._inference_pytorch
            self.trt_engine = None
            print("[Service] Using PyTorch inference")

    def _inference_pytorch(
        self,
        frames: np.ndarray,
        text_tokens: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run inference using PyTorch."""
        frames_tensor = torch.from_numpy(frames).float()
        text_tensor = torch.from_numpy(text_tokens).long()

        with torch.no_grad():
            behavior_logits, alert_logits = self.model(frames_tensor, text_tensor)

        behavior_probs = torch.softmax(behavior_logits, dim=-1).numpy()
        alert_probs = torch.softmax(alert_logits, dim=-1).numpy()

        return behavior_probs, alert_probs

    def _inference_trt(
        self,
        frames: np.ndarray,
        text_tokens: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run inference using TensorRT."""
        if self.trt_engine is None:
            return self._inference_pytorch(frames, text_tokens)
        return self.trt_engine.infer(frames, text_tokens)

    def _on_model_received(self, header: dict, model_data: bytes) -> None:
        """Handle new model pushed from cloud."""
        version = header.get("version", "unknown")
        model_dir = self.registry.storage_dir
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"edgeguard_v{version}.engine"

        try:
            with open(model_path, "wb") as f:
                f.write(model_data)

            if self.registry.validate_checksum(version, str(model_path)):
                self.registry.register_model(
                    str(model_path),
                    version=version,
                    precision=header.get("precision", "fp16"),
                    framework="tensorrt",
                    metrics=header.get("metrics", {}),
                )

                self.registry.set_active_version(version)
                self._init_inference_engine(str(model_path))

                print(f"[Service] Model v{version} installed and activated")
            else:
                print(f"[Service] Checksum mismatch for v{version}")
                self.mqtt.publish_model_ack(version, success=False, error_message="Checksum mismatch")

        except Exception as e:
            print(f"[Service] Failed to install model v{version}: {e}")
            self.mqtt.publish_model_ack(version, success=False, error_message=str(e))

    def _on_alert_triggered(self, alert: dict) -> None:
        """Handle triggered anomaly alert."""
        print(f"[ALERT] {alert['severity'].upper()}: Behavior class {alert['behavior']['class']} "
              f"(conf: {alert['behavior']['confidence']:.2f})")

    def _setup_signal_handlers(self) -> None:
        """Set up graceful shutdown handlers."""
        def signal_handler(sig, frame):
            print("\n[Service] Shutdown signal received")
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def start(self) -> None:
        """Start the edge service."""
        self.running = True
        self.mqtt.start()
        self.mqtt.publish_status("running", model_version=self.registry.get_active_version())
        print(f"[Service] EdgeGuard service started (device: {self.device_id})")

        self._run_loop()

    def stop(self) -> None:
        """Stop the edge service."""
        self.running = False
        self.mqtt.publish_status("stopped")
        self.mqtt.stop()
        print("[Service] EdgeGuard service stopped")

    def infer(
        self,
        frames: np.ndarray,
        text_tokens: np.ndarray,
        metadata: dict | None = None,
    ) -> dict:
        """
        Run inference on a video clip.

        Args:
            frames: Video frames of shape (1, T, 3, H, W).
            text_tokens: Token IDs of shape (1, seq_len).
            metadata: Optional metadata (timestamp, frame_id, etc.).

        Returns:
            Dictionary with predictions and performance metrics.
        """
        start_time = time.perf_counter()

        behavior_probs, alert_probs = self.inference_fn(frames, text_tokens)

        latency_ms = (time.perf_counter() - start_time) * 1000

        behavior_class = int(behavior_probs[0].argmax())
        behavior_conf = float(behavior_probs[0].max())
        alert_class = int(alert_probs[0].argmax())
        alert_conf = float(alert_probs[0].max())

        result = {
            "behavior": {
                "class": behavior_class,
                "name": self.BEHAVIOR_NAMES[behavior_class],
                "confidence": behavior_conf,
                "probabilities": behavior_probs[0].tolist(),
            },
            "alert": {
                "class": alert_class,
                "name": self.ALERT_NAMES[alert_class],
                "confidence": alert_conf,
                "probabilities": alert_probs[0].tolist(),
            },
            "performance": {
                "latency_ms": latency_ms,
                "fps": 1.0 / (latency_ms / 1000) if latency_ms > 0 else 0,
            },
            "metadata": metadata or {},
        }

        self.inference_count += 1
        self.total_latency_ms += latency_ms
        self.fps = self.inference_count / (time.time() - self.start_time + 1e-6)

        if behavior_conf >= self.alert_threshold and behavior_class != 6:
            self.mqtt.publish_anomaly_alert(
                behavior_class=behavior_class,
                behavior_confidence=behavior_conf,
                alert_class=alert_class,
                alert_confidence=alert_conf,
                metadata=metadata,
            )

        self.mqtt.publish_inference_report(
            behavior_class=behavior_class,
            behavior_confidence=behavior_conf,
            alert_class=alert_class,
            alert_confidence=alert_conf,
            latency_ms=latency_ms,
            fps=result["performance"]["fps"],
            metadata=metadata,
        )

        return result

    def _run_loop(self) -> None:
        """Main inference loop with synthetic data for demo."""
        self.start_time = time.time()
        print("[Service] Entering inference loop (Ctrl+C to stop)")

        while self.running:
            dummy_frames = np.random.randn(1, 16, 3, 224, 224).astype(np.float32)
            dummy_tokens = np.random.randint(0, 30000, (1, 128), dtype=np.int32)

            result = self.infer(dummy_frames, dummy_tokens, {"source": "demo"})

            if self.inference_count % 10 == 0:
                avg_latency = self.total_latency_ms / max(self.inference_count, 1)
                print(f"[Service] Count: {self.inference_count}, "
                      f"Avg Latency: {avg_latency:.1f}ms, "
                      f"FPS: {self.fps:.1f}, "
                      f"Last: {result['behavior']['name']} ({result['behavior']['confidence']:.2f})")

            time.sleep(0.1)


def main() -> None:
    """Entry point for edge service."""
    import argparse

    parser = argparse.ArgumentParser(description="EdgeGuard Edge Service")
    parser.add_argument("--device_id", type=str, default="edge-001")
    parser.add_argument("--model", type=str, default=None, help="Path to model file (.pt or .engine)")
    parser.add_argument("--mqtt_host", type=str, default="localhost")
    parser.add_argument("--mqtt_port", type=int, default=1883)
    parser.add_argument("--alert_threshold", type=float, default=0.75)
    parser.add_argument("--use_trt", action="store_true", help="Use TensorRT engine")
    parser.add_argument("--no_mqtt", action="store_true", help="Run without MQTT")

    args = parser.parse_args()

    mqtt_config = None
    if not args.no_mqtt:
        mqtt_config = MQTTConfig(
            broker_host=args.mqtt_host,
            broker_port=args.mqtt_port,
        )

    service = EdgeService(
        device_id=args.device_id,
        model_path=args.model,
        mqtt_config=mqtt_config,
        alert_threshold=args.alert_threshold,
        use_trt=args.use_trt,
    )

    service.start()


if __name__ == "__main__":
    main()
