"""
EdgeGuard: Monitoring Module

Grafana/Prometheus integration for real-time edge device monitoring.
Exports metrics for inference latency, FPS, memory usage, and alert counts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class InferenceMetrics:
    """Metrics for a single inference cycle."""
    timestamp: float
    latency_ms: float
    fps: float
    memory_mb: float
    batch_size: int
    behavior_class: int
    behavior_confidence: float
    alert_class: int
    alert_confidence: float


class MetricsCollector:
    """
    Collects and aggregates inference metrics for monitoring.

    Supports Prometheus exposition format and in-memory aggregation.

    Args:
        window_size: Rolling window size for metrics aggregation (seconds).
    """

    BEHAVIOR_NAMES = ["fighting", "falling", "climbing", "loitering", "retrograde", "gathering", "normal"]
    ALERT_NAMES = ["intrusion", "fault", "violation", "anomaly", "normal"]

    def __init__(self, window_size: int = 60) -> None:
        self.window_size = window_size
        self.metrics: list[InferenceMetrics] = []
        self.start_time = time.time()
        self.alert_counts = {name: 0 for name in self.BEHAVIOR_NAMES}
        self.alert_counts["total"] = 0

    def record(
        self,
        latency_ms: float,
        fps: float,
        memory_mb: float,
        batch_size: int,
        behavior_class: int,
        behavior_confidence: float,
        alert_class: int,
        alert_confidence: float,
    ) -> None:
        """Record a single inference result."""
        metric = InferenceMetrics(
            timestamp=time.time(),
            latency_ms=latency_ms,
            fps=fps,
            memory_mb=memory_mb,
            batch_size=batch_size,
            behavior_class=behavior_class,
            behavior_confidence=behavior_confidence,
            alert_class=alert_class,
            alert_confidence=alert_confidence,
        )
        self.metrics.append(metric)
        self._prune_old()

        if behavior_confidence > 0.75 and behavior_class != 6:
            self.alert_counts[self.BEHAVIOR_NAMES[behavior_class]] += 1
            self.alert_counts["total"] += 1

    def _prune_old(self) -> None:
        """Remove metrics outside the rolling window."""
        cutoff = time.time() - self.window_size
        self.metrics = [m for m in self.metrics if m.timestamp >= cutoff]

    def get_summary(self) -> dict:
        """Get aggregated metrics summary."""
        if not self.metrics:
            return self._empty_summary()

        recent = self.metrics
        latencies = [m.latency_ms for m in recent]
        fps_values = [m.fps for m in recent]
        memory_values = [m.memory_mb for m in recent]

        return {
            "uptime_seconds": time.time() - self.start_time,
            "sample_count": len(recent),
            "latency": {
                "mean_ms": float(np.mean(latencies)),
                "p50_ms": float(np.percentile(latencies, 50)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "p99_ms": float(np.percentile(latencies, 99)),
                "min_ms": float(np.min(latencies)),
                "max_ms": float(np.max(latencies)),
            },
            "throughput": {
                "mean_fps": float(np.mean(fps_values)),
                "min_fps": float(np.min(fps_values)),
                "max_fps": float(np.max(fps_values)),
            },
            "memory": {
                "mean_mb": float(np.mean(memory_values)),
                "peak_mb": float(np.max(memory_values)),
            },
            "alerts": self.alert_counts.copy(),
            "alerts_per_minute": self.alert_counts["total"] / max(self.window_size / 60, 1),
        }

    def _empty_summary(self) -> dict:
        """Return empty summary when no metrics available."""
        return {
            "uptime_seconds": time.time() - self.start_time,
            "sample_count": 0,
            "latency": {"mean_ms": 0, "p95_ms": 0},
            "throughput": {"mean_fps": 0},
            "memory": {"peak_mb": 0},
            "alerts": self.alert_counts.copy(),
            "alerts_per_minute": 0,
        }

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus text exposition format.

        Returns:
            Prometheus-formatted metrics string.
        """
        summary = self.get_summary()
        lines = [
            '# HELP edgeguard_uptime_seconds Time since service start',
            '# TYPE edgeguard_uptime_seconds gauge',
            f'edgeguard_uptime_seconds {summary["uptime_seconds"]:.2f}',
            '',
            '# HELP edgeguard_inference_samples_total Total number of inference samples',
            '# TYPE edgeguard_inference_samples_total counter',
            f'edgeguard_inference_samples_total {summary["sample_count"]}',
            '',
            '# HELP edgeguard_latency_ms Inference latency in milliseconds',
            '# TYPE edgeguard_latency_ms summary',
            f'edgeguard_latency_ms{{quantile="0.5"}} {summary["latency"]["p50_ms"]:.2f}',
            f'edgeguard_latency_ms{{quantile="0.95"}} {summary["latency"]["p95_ms"]:.2f}',
            f'edgeguard_latency_ms{{quantile="0.99"}} {summary["latency"]["p99_ms"]:.2f}',
            f'edgeguard_latency_ms_sum {summary["latency"]["mean_ms"] * summary["sample_count"]:.2f}',
            f'edgeguard_latency_ms_count {summary["sample_count"]}',
            '',
            '# HELP edgeguard_fps Frames per second',
            '# TYPE edgeguard_fps gauge',
            f'edgeguard_fps {summary["throughput"]["mean_fps"]:.2f}',
            '',
            '# HELP edgeguard_memory_mb GPU memory usage in MB',
            '# TYPE edgeguard_memory_mb gauge',
            f'edgeguard_memory_mb {summary["memory"]["peak_mb"]:.1f}',
            '',
            '# HELP edgeguard_alerts_total Total anomaly alerts by type',
            '# TYPE edgeguard_alerts_total counter',
        ]

        for name, count in self.alert_counts.items():
            if name != "total":
                lines.append(f'edgeguard_alerts_total{{type="{name}"}} {count}')

        lines.append(f'edgeguard_alerts_total{{type="all"}} {self.alert_counts["total"]}')

        return "\n".join(lines)


class GrafanaExporter:
    """
    Export metrics for Grafana dashboard integration.

    Formats metrics as JSON for Grafana SimpleJSON plugin
    or as Prometheus metrics endpoint.

    Args:
        collector: MetricsCollector instance.
    """

    def __init__(self, collector: MetricsCollector) -> None:
        self.collector = collector

    def get_json_payload(self) -> dict:
        """Get metrics as JSON payload for Grafana SimpleJSON."""
        summary = self.collector.get_summary()

        return {
            "uptime": summary["uptime_seconds"],
            "inference_count": summary["sample_count"],
            "latency_mean_ms": summary["latency"]["mean_ms"],
            "latency_p95_ms": summary["latency"]["p95_ms"],
            "fps": summary["throughput"]["mean_fps"],
            "memory_mb": summary["memory"]["peak_mb"],
            "alert_count": summary["alerts"]["total"],
            "alerts_per_minute": summary["alerts_per_minute"],
        }

    def get_targets(self) -> list[dict]:
        """Get Grafana targets for dashboard panels."""
        return [
            {"target": "inference_latency_ms", "datapoints": [[self.collector.get_summary()["latency"]["mean_ms"], int(time.time() * 1000)]]},
            {"target": "throughput_fps", "datapoints": [[self.collector.get_summary()["throughput"]["mean_fps"], int(time.time() * 1000)]]},
            {"target": "memory_mb", "datapoints": [[self.collector.get_summary()["memory"]["peak_mb"], int(time.time() * 1000)]]},
            {"target": "alerts_total", "datapoints": [[self.collector.get_summary()["alerts"]["total"], int(time.time() * 1000)]]},
        ]
