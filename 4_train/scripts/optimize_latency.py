#!/usr/bin/env python3
"""
Comprehensive latency optimization benchmark for DSC-CBAM-GRU.
Tests multiple strategies to achieve 25%+ latency reduction vs FP32 baseline.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import argparse
from typing import Any

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import create_dataloaders, load_npz_data
from src.models import DSC_CBAM_GRU, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comprehensive latency optimization benchmark")
    parser.add_argument("--data_dir", default="../dataset_cicids17")
    parser.add_argument("--checkpoint", default="checkpoints_gru/cicids17_gru_best.pt")
    parser.add_argument("--output_dir", default="experiments/compression")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--benchmark_batches", type=int, default=200)
    parser.add_argument("--warmup_batches", type=int, default=30)
    parser.add_argument("--num_runs", type=int, default=3, help="Number of benchmark runs for averaging")
    parser.add_argument("--input_dim", type=int, default=18)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--bidirectional", action="store_true", default=True)
    parser.add_argument("--dropout", type=float, default=0.3)
    return parser.parse_args()


def build_model(args: argparse.Namespace, hidden_dim: int | None = None) -> DSC_CBAM_GRU:
    h = hidden_dim or args.hidden_dim
    return DSC_CBAM_GRU(
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        hidden_dim=h,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
    )


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def create_test_loader(data_dir: str, batch_size: int):
    x_train, y_train, x_val, y_val, x_test, y_test = load_npz_data(data_dir)
    _, _, test_loader = create_dataloaders(
        x_train, y_train, x_val, y_val, x_test, y_test,
        batch_size=batch_size, num_workers=0, pin_memory=False,
    )
    return test_loader


def evaluate_model(model: nn.Module, loader, device: torch.device) -> dict[str, Any]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features.to(device))
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision": float(precision_score(all_labels, all_preds, average="weighted")),
        "recall": float(recall_score(all_labels, all_preds, average="weighted")),
        "f1": float(f1_score(all_labels, all_preds, average="weighted")),
    }


def benchmark_steady_state(
    model: nn.Module,
    loader,
    device: torch.device,
    num_batches: int = 200,
    warmup_batches: int = 30,
) -> dict[str, float]:
    """Benchmark with proper warmup and steady-state measurements."""
    model.eval()
    measured_batches = 0
    measured_samples = 0
    total_seconds = 0.0

    with torch.no_grad():
        for batch_idx, (features, _) in enumerate(loader):
            features = features.to(device)
            start = time.perf_counter()
            _ = model(features)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start

            if batch_idx >= warmup_batches:
                total_seconds += elapsed
                measured_batches += 1
                measured_samples += features.size(0)
            if measured_batches >= num_batches:
                break

    if measured_batches == 0:
        return {"latency_ms_per_sample": 0.0, "throughput": 0.0}
    return {
        "latency_ms_per_sample": float(total_seconds * 1000.0 / measured_samples),
        "throughput": float(measured_samples / total_seconds),
        "total_seconds": float(total_seconds),
        "measured_batches": measured_batches,
        "measured_samples": measured_samples,
    }


def benchmark_multiple_runs(model, loader, device, num_runs=3, **kwargs) -> dict[str, float]:
    """Run benchmark multiple times and return median results."""
    results = []
    for i in range(num_runs):
        torch.cuda.synchronize(device) if device.type == "cuda" else gc.collect()
        r = benchmark_steady_state(model, loader, device, **kwargs)
        results.append(r)
    latencies = sorted([r["latency_ms_per_sample"] for r in results])
    median_idx = len(latencies) // 2
    return {
        "latency_ms_per_sample": latencies[median_idx],
        "latency_p50": latencies[median_idx],
        "latency_p5": latencies[0],
        "latency_p95": latencies[-1],
        "throughput": results[median_idx]["throughput"],
        "all_runs": results,
    }


def apply_int8_dynamic(model: nn.Module) -> nn.Module:
    """Apply dynamic INT8 quantization to Linear and GRU layers."""
    return torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.GRU},
        dtype=torch.qint8,
    )


def apply_fp16_manual(model: nn.Module) -> nn.Module:
    """Convert model to FP16 for inference."""
    model = model.half()
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_compression_sweep(
    args: argparse.Namespace,
    test_loader,
    device: torch.device,
    base_model_path: str,
) -> dict[str, Any]:
    """Run comprehensive compression sweep across hidden dims and methods."""
    results = {}
    hidden_dims = [64, 56, 48, 40, 32]

    for hdim in hidden_dims:
        print(f"\n{'='*60}")
        print(f"Testing hidden_dim={hdim}")
        print('='*60)

        # Build and load model
        model = build_model(args, hidden_dim=hdim)
        model = load_checkpoint(model, base_model_path, device)
        n_params = count_params(model)
        print(f"Parameters: {n_params:,}")

        # Baseline FP32
        print(f"  Benchmarking FP32 baseline...")
        base_metrics = benchmark_multiple_runs(
            model, test_loader, device,
            num_runs=args.num_runs,
            num_batches=args.benchmark_batches,
            warmup_batches=args.warmup_batches,
        )
        base_acc = evaluate_model(model, test_loader, device)
        results[f"fp32_hdim{hdim}"] = {
            "hidden_dim": hdim,
            "method": "fp32",
            "parameters": n_params,
            "accuracy": base_acc["accuracy"],
            "f1": base_acc["f1"],
            **base_metrics,
        }
        print(f"  FP32: {base_metrics['latency_ms_per_sample']:.4f} ms/sample, "
              f"acc={base_acc['accuracy']:.4f}")

        # INT8 Dynamic Quantization
        print(f"  Testing INT8 dynamic quantization...")
        int8_model = build_model(args, hidden_dim=hdim)
        int8_model = load_checkpoint(int8_model, base_model_path, device)
        int8_model = apply_int8_dynamic(int8_model)
        int8_metrics = benchmark_multiple_runs(
            int8_model, test_loader, device,
            num_runs=args.num_runs,
            num_batches=args.benchmark_batches,
            warmup_batches=args.warmup_batches,
        )
        int8_acc = evaluate_model(int8_model, test_loader, device)
        lat_red = 1.0 - int8_metrics['latency_ms_per_sample'] / base_metrics['latency_ms_per_sample']
        results[f"int8_hdim{hdim}"] = {
            "hidden_dim": hdim,
            "method": "int8_dynamic",
            "parameters": n_params,
            "accuracy": int8_acc["accuracy"],
            "f1": int8_acc["f1"],
            **int8_metrics,
            "latency_reduction": lat_red,
        }
        print(f"  INT8: {int8_metrics['latency_ms_per_sample']:.4f} ms/sample "
              f"(Δ={lat_red*100:+.1f}%), acc={int8_acc['accuracy']:.4f}")

        # Pruning 30% + INT8
        print(f"  Testing Pruning30%+INT8...")
        import torch.nn.utils.prune as prune
        prune_model = build_model(args, hidden_dim=hdim)
        prune_model = load_checkpoint(prune_model, base_model_path, device)
        prunable_params = [
            (prune_model.conv, "weight"),
            (prune_model.dsc.dw, "weight"),
            (prune_model.dsc.pw, "weight"),
            (prune_model.cbam.channel_attention.fc[0], "weight"),
            (prune_model.cbam.channel_attention.fc[2], "weight"),
            (prune_model.cbam.spatial_attention.conv, "weight"),
            (prune_model.fc[0], "weight"),
            (prune_model.fc[3], "weight"),
            (prune_model.gru, "weight_ih_l0"),
            (prune_model.gru, "weight_hh_l0"),
        ]
        if prune_model.bidirectional:
            prunable_params.extend([
                (prune_model.gru, "weight_ih_l0_reverse"),
                (prune_model.gru, "weight_hh_l0_reverse"),
            ])
        prune.global_unstructured(prunable_params, pruning_method=prune.L1Unstructured, amount=0.30)
        for m, n in prunable_params:
            try:
                prune.remove(m, n)
            except ValueError:
                pass
        prune_model = apply_int8_dynamic(prune_model)
        prune_int8_metrics = benchmark_multiple_runs(
            prune_model, test_loader, device,
            num_runs=args.num_runs,
            num_batches=args.benchmark_batches,
            warmup_batches=args.warmup_batches,
        )
        prune_int8_acc = evaluate_model(prune_model, test_loader, device)
        lat_red_p = 1.0 - prune_int8_metrics['latency_ms_per_sample'] / base_metrics['latency_ms_per_sample']
        results[f"prune30_int8_hdim{hdim}"] = {
            "hidden_dim": hdim,
            "method": "prune30_int8",
            "parameters": n_params,
            "accuracy": prune_int8_acc["accuracy"],
            "f1": prune_int8_acc["f1"],
            **prune_int8_metrics,
            "latency_reduction": lat_red_p,
        }
        print(f"  Prune30+INT8: {prune_int8_metrics['latency_ms_per_sample']:.4f} ms/sample "
              f"(Δ={lat_red_p*100:+.1f}%), acc={prune_int8_acc['accuracy']:.4f}")

        # JIT Scripted FP32 (no quantization overhead)
        print(f"  Testing JIT Scripted FP32...")
        jit_model = build_model(args, hidden_dim=hdim)
        jit_model = load_checkpoint(jit_model, base_model_path, device)
        scripted = torch.jit.script(jit_model)
        jit_metrics = benchmark_multiple_runs(
            scripted, test_loader, device,
            num_runs=args.num_runs,
            num_batches=args.benchmark_batches,
            warmup_batches=args.warmup_batches,
        )
        jit_acc = evaluate_model(jit_model, test_loader, device)
        lat_red_jit = 1.0 - jit_metrics['latency_ms_per_sample'] / base_metrics['latency_ms_per_sample']
        results[f"jit_fp32_hdim{hdim}"] = {
            "hidden_dim": hdim,
            "method": "jit_fp32",
            "parameters": n_params,
            "accuracy": jit_acc["accuracy"],
            "f1": jit_acc["f1"],
            **jit_metrics,
            "latency_reduction": lat_red_jit,
        }
        print(f"  JIT FP32: {jit_metrics['latency_ms_per_sample']:.4f} ms/sample "
              f"(Δ={lat_red_jit*100:+.1f}%), acc={jit_acc['accuracy']:.4f}")

        del model, int8_model, prune_model, jit_model, scripted
        gc.collect()
        torch.cuda.synchronize() if device.type == "cuda" else None

    return results


def find_best_config(results: dict[str, Any], target_reduction: float = 0.25) -> dict[str, Any]:
    """Find the best configuration meeting accuracy and latency targets."""
    candidates = []
    for key, r in results.items():
        lat_red = r.get("latency_reduction", 0.0)
        acc = r.get("accuracy", 0.0)
        candidates.append({
            "config": key,
            "latency_reduction": lat_red,
            "accuracy": acc,
            "meets_25_target": lat_red >= target_reduction,
            "meets_acc_target": acc >= 0.95,
        })
    return sorted(candidates, key=lambda x: x["latency_reduction"], reverse=True)


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Quantized engine: {torch.backends.quantized.engine}")
    print(f"CPU threads: {torch.get_num_threads()}")
    torch.set_num_threads(8)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "compression"), exist_ok=True)

    test_loader = create_test_loader(args.data_dir, args.batch_size)
    print(f"Test batches: {len(test_loader)}")

    print("\n" + "="*60)
    print("COMPREHENSIVE LATENCY OPTIMIZATION BENCHMARK")
    print("="*60)

    results = run_compression_sweep(args, test_loader, device, args.checkpoint)

    # Save raw results
    output_json = os.path.join(args.output_dir, "compression", "optimization_results.json")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    # Find best configs
    ranked = find_best_config(results)
    print("\n" + "="*60)
    print("RANKED RESULTS (by latency reduction)")
    print("="*60)
    print(f"{'Config':<30} {'Lat.Red%':>10} {'Accuracy':>10} {'25%Target':>10}")
    print("-"*60)
    for c in ranked:
        flag = "✓" if c["meets_25_target"] else "✗"
        print(f"{c['config']:<30} {c['latency_reduction']*100:>+9.1f}% {c['accuracy']:>9.4f} {flag:>10}")

    best = ranked[0]
    print(f"\nBest config: {best['config']}")
    print(f"Best latency reduction: {best['latency_reduction']*100:+.1f}%")
    print(f"Meets 25% target: {best['meets_25_target']}")
    print(f"\nFull results saved to: {output_json}")

    # Summary table
    summary = {
        "target": "25% latency reduction vs FP32 baseline",
        "best_config": best["config"],
        "best_latency_reduction": best["latency_reduction"],
        "best_accuracy": best["accuracy"],
        "meets_target": best["meets_25_target"],
        "all_configs": ranked,
        "raw_results": results,
    }
    summary_json = os.path.join(args.output_dir, "compression", "optimization_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved to: {summary_json}")


if __name__ == "__main__":
    main()
