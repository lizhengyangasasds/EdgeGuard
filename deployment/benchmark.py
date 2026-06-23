"""
EdgeGuard: TensorRT Benchmark Script

Comprehensive performance benchmarking for TensorRT inference engines.
Measures latency, throughput, memory usage, and accuracy degradation.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def benchmark_trt_engine(
    engine_path: str,
    num_warmup: int = 10,
    num_iterations: int = 100,
    batch_sizes: list[int] | None = None,
    output_file: str | None = None,
) -> dict:
    """
    Benchmark a TensorRT engine for performance.

    Args:
        engine_path: Path to TensorRT engine file.
        num_warmup: Number of warm-up iterations.
        num_iterations: Number of benchmark iterations.
        batch_sizes: List of batch sizes to test.
        output_file: Path to save results JSON.

    Returns:
        Dictionary of benchmark results.
    """
    from trt_inference import TensorRTInference

    batch_sizes = batch_sizes or [1, 4, 8]

    print("=" * 60)
    print(f"EdgeGuard TensorRT Benchmark")
    print(f"Engine: {engine_path}")
    print(f"Warmup: {num_warmup}, Iterations: {num_iterations}")
    print("=" * 60)

    engine = TensorRTInference(engine_path)
    engine.warmup(num_warmup)

    results = {
        "engine_path": engine_path,
        "timestamp": datetime.now().isoformat(),
        "warmup_iterations": num_warmup,
        "benchmark_iterations": num_iterations,
        "batch_sizes": {},
        "memory": engine.get_memory_usage(),
    }

    print("\nMemory Usage:")
    mem = results["memory"]
    print(f"  Used: {mem['used_mb']:.1f} MB / {mem['total_mb']:.1f} MB")
    print()

    for bs in batch_sizes:
        print(f"\nBenchmarking batch_size={bs}...")

        dummy_frames = np.random.randn(bs, 16, 3, 224, 224).astype(np.float32)
        dummy_tokens = np.random.randint(0, 30000, (bs, 128), dtype=np.int32)

        latencies = []
        for i in range(num_iterations):
            start = time.perf_counter()
            engine.infer(dummy_frames, dummy_tokens)
            latencies.append((time.perf_counter() - start) * 1000)

        avg_lat = np.mean(latencies)
        fps = bs / (avg_lat / 1000)

        results["batch_sizes"][str(bs)] = {
            "avg_latency_ms": round(float(avg_lat), 2),
            "min_latency_ms": round(float(np.min(latencies)), 2),
            "max_latency_ms": round(float(np.max(latencies)), 2),
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2),
            "throughput_fps": round(float(fps), 2),
            "batch_size": bs,
        }

        print(f"  Avg latency: {avg_lat:.2f} ms")
        print(f"  P50 latency: {results['batch_sizes'][str(bs)]['p50_latency_ms']:.2f} ms")
        print(f"  P95 latency: {results['batch_sizes'][str(bs)]['p95_latency_ms']:.2f} ms")
        print(f"  Throughput: {fps:.2f} FPS")

    results["summary"] = {
        "meets_latency_target": all(
            results["batch_sizes"][str(bs)]["avg_latency_ms"] < 85
            for bs in batch_sizes
        ),
        "meets_throughput_target": any(
            results["batch_sizes"][str(bs)]["throughput_fps"] >= 15
            for bs in batch_sizes
        ),
        "meets_memory_target": results["memory"]["used_mb"] < 2048,
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Latency target (<85ms): {'PASS' if results['summary']['meets_latency_target'] else 'FAIL'}")
    print(f"Throughput target (>=15 FPS): {'PASS' if results['summary']['meets_throughput_target'] else 'FAIL'}")
    print(f"Memory target (<2GB): {'PASS' if results['summary']['meets_memory_target'] else 'FAIL'}")

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results


def compare_precisions(
    base_engine_path: str,
    fp16_engine_path: str,
    int8_engine_path: str | None = None,
    num_iterations: int = 50,
) -> dict:
    """
    Compare inference results across precision modes.

    Args:
        base_engine_path: FP32 reference engine.
        fp16_engine_path: FP16 optimized engine.
        int8_engine_path: INT8 quantized engine.
        num_iterations: Number of comparison iterations.

    Returns:
        Dictionary of comparison results.
    """
    from trt_inference import TensorRTInference

    engines = {"fp32": base_engine_path}
    if Path(fp16_engine_path).exists():
        engines["fp16"] = fp16_engine_path
    if int8_engine_path and Path(int8_engine_path).exists():
        engines["int8"] = int8_engine_path

    loaded_engines = {}
    for name, path in engines.items():
        try:
            loaded_engines[name] = TensorRTInference(path)
            loaded_engines[name].warmup(5)
        except Exception as e:
            print(f"Could not load {name} engine: {e}")

    reference_results = None
    comparison = {}

    for name, engine in loaded_engines.items():
        latencies = []
        outputs = []

        for _ in range(num_iterations):
            dummy_frames = np.random.randn(1, 16, 3, 224, 224).astype(np.float32)
            dummy_tokens = np.random.randint(0, 30000, (1, 128), dtype=np.int32)

            start = time.perf_counter()
            behavior, alert = engine.infer(dummy_frames, dummy_tokens)
            latencies.append((time.perf_counter() - start) * 1000)
            outputs.append((behavior, alert))

            if reference_results is None:
                reference_results = (behavior.copy(), alert.copy())

        comparison[name] = {
            "avg_latency_ms": round(float(np.mean(latencies)), 2),
            "throughput_fps": round(float(1 / (np.mean(latencies) / 1000)), 2),
        }

        if name != "fp32":
            beh_diff = np.abs(outputs[0][0] - reference_results[0]).mean()
            alert_diff = np.abs(outputs[0][1] - reference_results[1]).mean()
            comparison[name]["behavior_diff_from_fp32"] = round(float(beh_diff), 6)
            comparison[name]["alert_diff_from_fp32"] = round(float(alert_diff), 6)

    print("\nPrecision Comparison:")
    print(f"{'Precision':<12} {'Latency':<12} {'FPS':<10} {'Beh Diff':<12} {'Alert Diff'}")
    for name, stats in comparison.items():
        beh_diff = stats.get("behavior_diff_from_fp32", "-")
        alert_diff = stats.get("alert_diff_from_fp32", "-")
        print(f"{name:<12} {stats['avg_latency_ms']:<12.2f} {stats['throughput_fps']:<10.2f} {beh_diff:<12} {alert_diff}")

    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark TensorRT inference engine")
    parser.add_argument("--engine", type=str, help="Path to TensorRT engine")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--compare", action="store_true", help="Compare precision modes")
    parser.add_argument("--fp16_engine", type=str, default=None)
    parser.add_argument("--int8_engine", type=str, default=None)

    args = parser.parse_args()

    if args.compare:
        if args.engine and args.fp16_engine:
            compare_precisions(
                args.engine,
                args.fp16_engine,
                args.int8_engine,
                args.iterations,
            )
        else:
            print("Error: --engine and --fp16_engine required for comparison")
    elif args.engine:
        benchmark_trt_engine(
            args.engine,
            num_warmup=args.warmup,
            num_iterations=args.iterations,
            batch_sizes=args.batch_sizes,
            output_file=args.output,
        )
    else:
        print("No engine specified. Use --engine <path> to benchmark.")
