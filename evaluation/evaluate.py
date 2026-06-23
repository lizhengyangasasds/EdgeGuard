"""
EdgeGuard: Evaluation Module

Comprehensive evaluation metrics and visualizations for model performance.
Supports behavior classification, alert classification, and system-level metrics.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
)


def compute_behavior_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    """
    Compute comprehensive metrics for behavior classification.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_probs: Predicted probabilities of shape (N, num_classes).
        class_names: Optional class name labels.

    Returns:
        Dictionary of computed metrics.
    """
    if class_names is None:
        class_names = [f"class_{i}" for i in range(y_probs.shape[1])]

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    metrics = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "f1_per_class": {name: float(f1) for name, f1 in zip(class_names, f1_per_class)},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_normalized.tolist(),
        "num_samples": len(y_true),
        "num_classes": len(class_names),
    }

    for i, name in enumerate(class_names):
        mask_true = y_true == i
        if mask_true.sum() > 0:
            tp = ((y_pred == i) & (y_true == i)).sum()
            fn = ((y_pred != i) & (y_true == i)).sum()
            fp = ((y_pred == i) & (y_true != i)).sum()
            metrics[f"{name}_recall"] = float(tp / (tp + fn + 1e-10))
            metrics[f"{name}_precision"] = float(tp / (tp + fp + 1e-10))
        else:
            metrics[f"{name}_recall"] = 0.0
            metrics[f"{name}_precision"] = 0.0

    recall_rates = []
    miss_rate = 0.0
    false_alarm_rate = 0.0
    num_normal = (y_true == 6).sum()
    if num_normal > 0:
        false_alarms = ((y_pred != 6) & (y_true == 6)).sum()
        false_alarm_rate = false_alarms / num_normal

    metrics["miss_rate"] = float(miss_rate)
    metrics["false_alarm_rate"] = float(false_alarm_rate)

    return metrics


def compute_text_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    """Compute metrics for alert text classification."""
    if class_names is None:
        class_names = [f"alert_{i}" for i in range(5)]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_per_class": {
            name: float(f1)
            for name, f1 in zip(class_names, f1_score(y_true, y_pred, average=None, zero_division=0))
        },
    }


def compute_system_metrics(
    latencies_ms: np.ndarray,
    fps_values: np.ndarray,
    memory_mb: np.ndarray,
    latency_target_ms: float = 85.0,
    fps_target: float = 15.0,
    memory_target_mb: float = 2048.0,
) -> dict:
    """Compute system-level performance metrics."""
    return {
        "latency": {
            "mean_ms": float(np.mean(latencies_ms)),
            "p50_ms": float(np.percentile(latencies_ms, 50)),
            "p95_ms": float(np.percentile(latencies_ms, 95)),
            "p99_ms": float(np.percentile(latencies_ms, 99)),
            "meets_target": bool(np.mean(latencies_ms) < latency_target_ms),
        },
        "throughput": {
            "mean_fps": float(np.mean(fps_values)),
            "min_fps": float(np.min(fps_values)),
            "meets_target": bool(np.mean(fps_values) >= fps_target),
        },
        "memory": {
            "mean_mb": float(np.mean(memory_mb)),
            "peak_mb": float(np.max(memory_mb)),
            "meets_target": bool(np.max(memory_mb) < memory_target_mb),
        },
    }


def compute_pr_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    num_classes: int,
) -> dict:
    """
    Compute PR and ROC curve data points.

    Args:
        y_true: Ground truth binary labels (for one-vs-rest).
        y_probs: Predicted probabilities for positive class.
        num_classes: Total number of classes.

    Returns:
        Dictionary with precision-recall and ROC curve data.
    """
    results = {}

    for class_idx in range(num_classes):
        binary_true = (y_true == class_idx).astype(int)
        probs = y_probs[:, class_idx]

        precision, recall, pr_thresholds = precision_recall_curve(binary_true, probs)
        fpr, tpr, roc_thresholds = roc_curve(binary_true, probs)
        ap = average_precision_score(binary_true, probs)

        results[f"class_{class_idx}"] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "average_precision": float(ap),
            "auc_roc": float(np.trapz(tpr, fpr)),
        }

    return results
