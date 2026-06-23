"""
EdgeGuard: MQTT Communication Module

MQTT client for edge-cloud bidirectional communication.
Handles model push from cloud to edge and inference reporting from edge to cloud.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import paho.mqtt.client as mqtt


@dataclass
class MQTTConfig:
    """MQTT connection and topic configuration."""
    broker_host: str = "localhost"
    broker_port: int = 1883
    keepalive: int = 60
    qos: int = 1
    username: Optional[str] = None
    password: Optional[str] = None

    topics: dict[str, str] = field(default_factory=lambda: {
        "model_push": "edgeguard/model/push",
        "model_ack": "edgeguard/model/ack",
        "inference_report": "edgeguard/inference/report",
        "anomaly_alert": "edgeguard/alert/anomaly",
        "status": "edgeguard/edge/status",
    })


class EdgeGuardMQTTClient:
    """
    MQTT client for EdgeGuard edge-cloud communication.

    Responsibilities:
    - Subscribe to model push messages from cloud
    - Publish inference results to cloud
    - Publish anomaly alerts when threshold exceeded
    - Handle model hot-update acknowledgements

    Args:
        config: MQTT configuration.
        device_id: Unique edge device identifier.
        on_model_received: Callback when new model is pushed.
        on_alert_triggered: Callback when anomaly is detected.
    """

    def __init__(
        self,
        config: Optional[MQTTConfig] = None,
        device_id: str = "edge-001",
        on_model_received: Optional[Callable[[dict, bytes], None]] = None,
        on_alert_triggered: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.config = config or MQTTConfig()
        self.device_id = device_id
        self.on_model_received = on_model_received
        self.on_alert_triggered = on_alert_triggered

        self.client = mqtt.Client(client_id=f"edgeguard-{device_id}")
        self._setup_callbacks()
        self._apply_auth()
        self.connected = False
        self._message_handlers: dict[str, Callable] = {}
        self._publish_lock = threading.Lock()

    def _setup_callbacks(self) -> None:
        """Set up MQTT client callbacks."""
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.on_publish = self._on_publish

    def _apply_auth(self) -> None:
        """Apply username/password authentication."""
        if self.config.username:
            self.client.username_pw_set(
                self.config.username,
                self.config.password or "",
            )

    def _on_connect(self, client, userdata, flags, rc: int) -> None:
        """Handle connection established."""
        if rc == 0:
            self.connected = True
            print(f"[MQTT] Connected to {self.config.broker_host}:{self.config.broker_port}")
            self._subscribe_topics()
        else:
            print(f"[MQTT] Connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc: int) -> None:
        """Handle disconnection."""
        self.connected = False
        print(f"[MQTT] Disconnected (code: {rc})")
        if rc != 0:
            print("[MQTT] Attempting to reconnect...")
            time.sleep(5)
            self.connect()

    def _on_message(self, client, userdata, msg: mqtt.MQTTMessage) -> None:
        """Handle incoming message."""
        topic = msg.topic
        payload = msg.payload

        try:
            if topic == self.config.topics["model_push"]:
                self._handle_model_push(payload)
            else:
                handler = self._message_handlers.get(topic)
                if handler:
                    handler(payload)
        except Exception as e:
            print(f"[MQTT] Error handling message on {topic}: {e}")

    def _on_publish(self, client, userdata, mid: int) -> None:
        """Handle publish confirmation."""
        pass

    def _subscribe_topics(self) -> None:
        """Subscribe to relevant topics."""
        for topic_name, topic_path in self.config.topics.items():
            if topic_name != "model_push":
                self.client.subscribe(topic_path, qos=self.config.qos)

        self.client.subscribe(self.config.topics["model_push"], qos=self.config.qos)

    def _handle_model_push(self, payload: bytes) -> None:
        """Handle incoming model push from cloud."""
        try:
            header_size = int.from_bytes(payload[:4], byteorder="big")
            header_json = payload[4:4 + header_size].decode("utf-8")
            header = json.loads(header_json)
            model_data = payload[4 + header_size:]

            if header.get("target_device") and header["target_device"] != self.device_id:
                return

            print(f"[MQTT] Received model push: version={header.get('version')}")

            if self.on_model_received:
                self.on_model_received(header, model_data)

            self.publish_model_ack(header.get("version"), success=True)

        except Exception as e:
            print(f"[MQTT] Error processing model push: {e}")

    def connect(self, broker: str | None = None, port: int | None = None) -> None:
        """
        Connect to the MQTT broker.

        Args:
            broker: Override broker host.
            port: Override broker port.
        """
        broker = broker or self.config.broker_host
        port = port or self.config.broker_port
        self.client.connect(broker, port, self.config.keepalive)

    def start(self) -> None:
        """Start the MQTT client loop in a background thread."""
        thread = threading.Thread(target=self.client.loop_start, daemon=True)
        thread.start()

    def stop(self) -> None:
        """Stop the MQTT client and disconnect."""
        self.client.loop_stop()
        self.client.disconnect()

    def publish_inference_report(
        self,
        behavior_class: int,
        behavior_confidence: float,
        alert_class: int,
        alert_confidence: float,
        latency_ms: float,
        fps: float,
        metadata: dict | None = None,
    ) -> None:
        """
        Publish inference results to the cloud.

        Args:
            behavior_class: Predicted behavior class index.
            behavior_confidence: Prediction confidence score.
            alert_class: Predicted alert class index.
            alert_confidence: Alert confidence score.
            latency_ms: Inference latency in milliseconds.
            fps: Current frames per second.
            metadata: Additional metadata (timestamp, frame_id, etc.).
        """
        report = {
            "device_id": self.device_id,
            "timestamp": time.time(),
            "behavior": {
                "class": behavior_class,
                "confidence": float(behavior_confidence),
            },
            "alert": {
                "class": alert_class,
                "confidence": float(alert_confidence),
            },
            "performance": {
                "latency_ms": float(latency_ms),
                "fps": float(fps),
            },
            "metadata": metadata or {},
        }

        self._publish(
            self.config.topics["inference_report"],
            json.dumps(report),
        )

    def publish_anomaly_alert(
        self,
        behavior_class: int,
        behavior_confidence: float,
        alert_class: int,
        alert_confidence: float,
        video_segment_path: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """
        Publish anomaly alert when behavior exceeds confidence threshold.

        Args:
            behavior_class: Anomaly class index.
            behavior_confidence: Anomaly confidence score.
            alert_class: Alert type index.
            alert_confidence: Alert confidence score.
            video_segment_path: Path to anomaly video clip (optional).
            metadata: Additional alert metadata.
        """
        alert = {
            "device_id": self.device_id,
            "timestamp": time.time(),
            "severity": "high" if behavior_confidence > 0.9 else "medium",
            "behavior": {
                "class": behavior_class,
                "confidence": float(behavior_confidence),
            },
            "alert": {
                "class": alert_class,
                "confidence": float(alert_confidence),
            },
            "video_segment": video_segment_path,
            "metadata": metadata or {},
        }

        self._publish(
            self.config.topics["anomaly_alert"],
            json.dumps(alert),
        )

        if self.on_alert_triggered:
            self.on_alert_triggered(alert)

    def publish_model_ack(
        self,
        model_version: str,
        success: bool,
        error_message: str | None = None,
    ) -> None:
        """Publish model update acknowledgement to cloud."""
        ack = {
            "device_id": self.device_id,
            "timestamp": time.time(),
            "model_version": model_version,
            "success": success,
            "error_message": error_message,
        }

        self._publish(
            self.config.topics["model_ack"],
            json.dumps(ack),
        )

    def publish_status(
        self,
        status: str,
        gpu_memory_mb: float | None = None,
        fps: float | None = None,
        model_version: str | None = None,
    ) -> None:
        """Publish edge device status."""
        status_msg = {
            "device_id": self.device_id,
            "timestamp": time.time(),
            "status": status,
            "gpu_memory_mb": gpu_memory_mb,
            "fps": fps,
            "model_version": model_version,
        }

        self._publish(
            self.config.topics["status"],
            json.dumps(status_msg),
        )

    def register_handler(self, topic: str, handler: Callable[[bytes], None]) -> None:
        """Register a custom message handler for a topic."""
        self._message_handlers[topic] = handler
        self.client.subscribe(topic, qos=self.config.qos)

    def _publish(self, topic: str, payload: str | bytes, qos: int | None = None) -> None:
        """Publish a message with thread safety."""
        qos = qos if qos is not None else self.config.qos
        with self._publish_lock:
            if self.connected:
                self.client.publish(topic, payload, qos=qos)
