"""
EdgeGuard: Visualization Module

Generates evaluation visualizations: confusion matrices, PR curves, ROC curves,
and training metric plots. Supports both matplotlib and export to image files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: str | None = None,
    normalize: bool = True,
    cmap: str = "Blues",
    figsize: tuple[int, int] = (10, 8),
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """
    Plot a confusion matrix heatmap.

    Args:
        cm: Confusion matrix array.
        class_names: List of class name labels.
        output_path: Optional path to save the figure.
        normalize: Whether to show normalized percentages.
        cmap: Matplotlib colormap name.
        figsize: Figure size in inches.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if normalize:
        cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_display = np.nan_to_num(cm_normalized)
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Proportion"},
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to: {output_path}")

    return fig


def plot_pr_curves(
    pr_data: dict,
    class_names: list[str],
    output_path: str | None = None,
    figsize: tuple[int, int] = (12, 8),
    title: str = "Precision-Recall Curves",
) -> plt.Figure:
    """
    Plot precision-recall curves for all classes.

    Args:
        pr_data: Dictionary with PR curve data per class.
        class_names: List of class name labels.
        output_path: Optional path to save the figure.
        figsize: Figure size in inches.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    for i, class_name in enumerate(class_names):
        data = pr_data.get(f"class_{i}", {})
        if data.get("precision"):
            recall = data["precision"]
            precision_vals = data["recall"]
            ap = data.get("average_precision", 0)
            label = f"{class_name} (AP={ap:.3f})"
            ax.plot(recall_vals, precision, label=label, color=colors[i], linewidth=2)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved PR curves to: {output_path}")

    return fig


def plot_roc_curves(
    roc_data: dict,
    class_names: list[str],
    output_path: str | None = None,
    figsize: tuple[int, int] = (10, 8),
    title: str = "ROC Curves",
) -> plt.Figure:
    """Plot ROC curves for all classes."""
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    for i, class_name in enumerate(class_names):
        data = roc_data.get(f"class_{i}", {})
        if data.get("fpr"):
            ax.plot(
                data["fpr"],
                data["tpr"],
                label=f"{class_name} (AUC={data.get('auc_roc', 0):.3f})",
                color=colors[i],
                linewidth=2,
            )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved ROC curves to: {output_path}")

    return fig


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    train_accs: list[float] | None = None,
    val_accs: list[float] | None = None,
    output_path: str | None = None,
    title: str = "Training Curves",
) -> plt.Figure:
    """
    Plot training and validation loss/accuracy curves.

    Args:
        train_losses: Training loss per epoch.
        val_losses: Validation loss per epoch.
        train_accs: Training accuracy per epoch.
        val_accs: Validation accuracy per epoch.
        output_path: Optional save path.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    num_plots = 1 + (train_accs is not None)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))

    if num_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, "b-", linewidth=2, label="Train Loss")
    if val_losses:
        axes[0].plot(epochs, val_losses, "r-", linewidth=2, label="Val Loss")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Loss", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if train_accs is not None and num_plots > 1:
        axes[1].plot(epochs, train_accs, "b-", linewidth=2, label="Train Acc")
        if val_accs:
            axes[1].plot(epochs, val_accs, "r-", linewidth=2, label="Val Acc")
        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("Accuracy", fontsize=12)
        axes[1].set_title("Accuracy", fontsize=14, fontweight="bold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves to: {output_path}")

    return fig


def plot_benchmark_results(
    benchmark_results: dict,
    output_path: str | None = None,
) -> plt.Figure:
    """
    Plot TensorRT benchmark results (latency and throughput).

    Args:
        benchmark_results: Dictionary of benchmark data per batch size.
        output_path: Optional save path.

    Returns:
        Matplotlib Figure object.
    """
    batch_sizes = sorted(int(k) for k in benchmark_results.keys())
    avg_latencies = [benchmark_results[str(bs)]["avg_latency_ms"] for bs in batch_sizes]
    p95_latencies = [benchmark_results[str(bs)]["p95_latency_ms"] for bs in batch_sizes]
    fps_values = [benchmark_results[str(bs)]["throughput_fps"] for bs in batch_sizes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(batch_sizes))
    width = 0.35

    ax1.bar(x - width / 2, avg_latencies, width, label="Avg Latency", color="steelblue")
    ax1.bar(x + width / 2, p95_latencies, width, label="P95 Latency", color="coral")
    ax1.axhline(y=85, color="red", linestyle="--", linewidth=1.5, label="Target (85ms)")
    ax1.set_xlabel("Batch Size", fontsize=12)
    ax1.set_ylabel("Latency (ms)", fontsize=12)
    ax1.set_title("Inference Latency", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(batch_sizes)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x, fps_values, color="seagreen", width=0.6)
    ax2.axhline(y=15, color="red", linestyle="--", linewidth=1.5, label="Target (15 FPS)")
    ax2.set_xlabel("Batch Size", fontsize=12)
    ax2.set_ylabel("Throughput (FPS)", fontsize=12)
    ax2.set_title("Throughput", fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(batch_sizes)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved benchmark results to: {output_path}")

    return fig


def generate_performance_report(
    metrics: dict,
    output_path: str,
) -> None:
    """Generate a text-based performance report."""
    report_lines = [
        "=" * 60,
        "EdgeGuard Performance Report",
        "=" * 60,
        "",
    ]

    if "behavior" in metrics:
        report_lines.append("Behavior Classification:")
        report_lines.append(f"  Accuracy: {metrics['behavior']['accuracy']:.2%}")
        report_lines.append(f"  F1 (macro): {metrics['behavior']['f1_macro']:.4f}")
        report_lines.append(f"  F1 (weighted): {metrics['behavior']['f1_weighted']:.4f}")
        report_lines.append("")

    if "system" in metrics:
        sys_m = metrics["system"]
        report_lines.append("System Performance:")
        report_lines.append(f"  Avg Latency: {sys_m['latency']['mean_ms']:.1f} ms (target: <85ms)")
        report_lines.append(f"  P95 Latency: {sys_m['latency']['p95_ms']:.1f} ms")
        report_lines.append(f"  Mean FPS: {sys_m['throughput']['mean_fps']:.1f} (target: >=15)")
        report_lines.append(f"  Peak Memory: {sys_m['memory']['peak_mb']:.0f} MB (target: <2048MB)")
        report_lines.append("")

    report_text = "\n".join(report_lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_text)

    print(f"Performance report saved to: {output_path}")
    print("\n" + report_text)
