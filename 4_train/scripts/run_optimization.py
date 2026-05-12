#!/usr/bin/env python3
"""
Comprehensive optimization: FP32 baseline, INT8, pruning+INT8, and hybrid approaches.
Runs all experiments in sequence and produces a final report.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import argparse
from typing import Any

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import create_dataloaders, load_npz_data
from src.models import DSC_CBAM_GRU, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comprehensive optimization benchmark")
    parser.add_argument("--data_dir", default="../dataset_cicids17")
    parser.add_argument("--checkpoint", default="checkpoints_gru/cicids17_gru_best.pt")
    parser.add_argument("--output_dir", default="experiments/compression")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--benchmark_batches", type=int, default=300)
    parser.add_argument("--warmup_batches", type=int, default=50)
    parser.add_argument("--input_dim", type=int, default=18)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--bidirectional", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_threads", type=int, default=8)
    return parser.parse_args()


def create_test_loader(data_dir: str, batch_size: int):
    x_train, y_train, x_val, y_val, x_test, y_test = load_npz_data(data_dir)
    _, _, test_loader = create_dataloaders(
        x_train, y_train, x_val, y_val, x_test, y_test,
        batch_size=batch_size, num_workers=0, pin_memory=False,
    )
    return test_loader


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


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


def benchmark_inference(
    model: nn.Module,
    loader,
    device: torch.device,
    num_batches: int = 300,
    warmup_batches: int = 50,
    model_label: str = "model",
) -> dict[str, Any]:
    """Robust inference benchmark."""
    model.eval()

    # Warmup
    warmup_loader = iter(loader)
    for _ in range(warmup_batches):
        try:
            features, _ = next(warmup_loader)
        except StopIteration:
            warmup_loader = iter(loader)
            features, _ = next(warmup_loader)
        with torch.no_grad():
            _ = model(features.to(device))

    # Benchmark
    total_seconds = 0.0
    measured_batches = 0
    measured_samples = 0

    bench_loader = iter(loader)
    with torch.no_grad():
        while measured_batches < num_batches:
            try:
                features, _ = next(bench_loader)
            except StopIteration:
                bench_loader = iter(loader)
                features, _ = next(bench_loader)

            features = features.to(device)
            start = time.perf_counter()
            _ = model(features)
            elapsed = time.perf_counter() - start

            total_seconds += elapsed
            measured_batches += 1
            measured_samples += features.size(0)

    return {
        "latency_ms_per_sample": float(total_seconds * 1000.0 / measured_samples),
        "latency_ms_per_batch": float(total_seconds * 1000.0 / measured_batches),
        "throughput_samples_per_sec": float(measured_samples / total_seconds),
        "total_inference_seconds": float(total_seconds),
        "measured_batches": measured_batches,
        "measured_samples": measured_samples,
    }


def count_nonzero(model: nn.Module) -> int:
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total += int(torch.count_nonzero(p).item())
    return total


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.num_threads)
    device = torch.device("cpu")

    print(f"PyTorch: {torch.__version__}")
    print(f"Threads: {torch.get_num_threads()}")
    print(f"Quantized engine: {torch.backends.quantized.engine}")

    output_subdir = os.path.join(args.output_dir, "compression")
    os.makedirs(output_subdir, exist_ok=True)
    test_loader = create_test_loader(args.data_dir, args.batch_size)
    print(f"Test loader: {len(test_loader)} batches of {args.batch_size}")

    results = {}
    n_params = count_parameters(DSC_CBAM_GRU(
        input_dim=args.input_dim, num_classes=args.num_classes,
        hidden_dim=args.hidden_dim, bidirectional=args.bidirectional, dropout=args.dropout
    ))

    # ─── FP32 BASELINE ───
    print("\n" + "="*60)
    print("1. FP32 BASELINE")
    print("="*60)
    model_fp32 = DSC_CBAM_GRU(
        input_dim=args.input_dim, num_classes=args.num_classes,
        hidden_dim=args.hidden_dim, bidirectional=args.bidirectional, dropout=args.dropout
    )
    model_fp32 = load_checkpoint(model_fp32, args.checkpoint, device)
    acc_fp32 = evaluate_model(model_fp32, test_loader, device)
    lat_fp32 = benchmark_inference(model_fp32, test_loader, device,
                                     num_batches=args.benchmark_batches,
                                     warmup_batches=args.warmup_batches,
                                     model_label="FP32")
    results["fp32_baseline"] = {
        "method": "fp32",
        "accuracy": acc_fp32["accuracy"],
        "f1": acc_fp32["f1"],
        "precision": acc_fp32["precision"],
        "recall": acc_fp32["recall"],
        "n_params": n_params,
        **lat_fp32,
    }
    print(f"FP32: {lat_fp32['latency_ms_per_sample']:.6f} ms/sample, "
          f"{lat_fp32['throughput_samples_per_sec']:.0f} samples/sec, "
          f"acc={acc_fp32['accuracy']:.4f}")

    baseline_latency = lat_fp32["latency_ms_per_sample"]

    # ─── INT8 DYNAMIC QUANTIZATION ───
    print("\n" + "="*60)
    print("2. INT8 DYNAMIC QUANTIZATION")
    print("="*60)
    model_int8 = DSC_CBAM_GRU(
        input_dim=args.input_dim, num_classes=args.num_classes,
        hidden_dim=args.hidden_dim, bidirectional=args.bidirectional, dropout=args.dropout
    )
    model_int8 = load_checkpoint(model_int8, args.checkpoint, device)
    model_int8 = torch.quantization.quantize_dynamic(
        model_int8, {nn.Linear, nn.GRU}, dtype=torch.qint8
    )
    acc_int8 = evaluate_model(model_int8, test_loader, device)
    lat_int8 = benchmark_inference(model_int8, test_loader, device,
                                     num_batches=args.benchmark_batches,
                                     warmup_batches=args.warmup_batches,
                                     model_label="INT8")
    results["int8_dynamic"] = {
        "method": "int8_dynamic",
        "accuracy": acc_int8["accuracy"],
        "f1": acc_int8["f1"],
        **lat_int8,
        "latency_reduction": 1.0 - lat_int8["latency_ms_per_sample"] / baseline_latency,
    }
    print(f"INT8: {lat_int8['latency_ms_per_sample']:.6f} ms/sample, "
          f"{lat_int8['throughput_samples_per_sec']:.0f} samples/sec, "
          f"acc={acc_int8['accuracy']:.4f}, "
          f"Δ={results['int8_dynamic']['latency_reduction']*100:+.1f}%")

    # Save INT8 model
    int8_path = os.path.join(output_subdir, "cicids17_gru_int8.pt")
    torch.save({"state_dict": model_int8.state_dict(), "config": {
        "input_dim": args.input_dim, "num_classes": args.num_classes,
        "hidden_dim": args.hidden_dim, "bidirectional": args.bidirectional,
        "dropout": args.dropout, "quantized": "int8_dynamic"
    }}, int8_path)
    results["int8_dynamic"]["checkpoint"] = int8_path

    # ─── PRUNING 40% + INT8 ───
    print("\n" + "="*60)
    print("3. PRUNING 40% + INT8")
    print("="*60)
    model_p40 = DSC_CBAM_GRU(
        input_dim=args.input_dim, num_classes=args.num_classes,
        hidden_dim=args.hidden_dim, bidirectional=args.bidirectional, dropout=args.dropout
    )
    model_p40 = load_checkpoint(model_p40, args.checkpoint, device)
    prunable_params = [
        (model_p40.conv, "weight"),
        (model_p40.dsc.dw, "weight"),
        (model_p40.dsc.pw, "weight"),
        (model_p40.cbam.channel_attention.fc[0], "weight"),
        (model_p40.cbam.channel_attention.fc[2], "weight"),
        (model_p40.cbam.spatial_attention.conv, "weight"),
        (model_p40.fc[0], "weight"),
        (model_p40.fc[3], "weight"),
        (model_p40.gru, "weight_ih_l0"),
        (model_p40.gru, "weight_hh_l0"),
    ]
    if model_p40.bidirectional:
        prunable_params.extend([
            (model_p40.gru, "weight_ih_l0_reverse"),
            (model_p40.gru, "weight_hh_l0_reverse"),
        ])
    prune.global_unstructured(prunable_params, pruning_method=prune.L1Unstructured, amount=0.40)
    for m, n in prunable_params:
        try:
            prune.remove(m, n)
        except ValueError:
            pass
    nonzero_p40 = count_nonzero(model_p40)
    model_p40 = torch.quantization.quantize_dynamic(
        model_p40, {nn.Linear, nn.GRU}, dtype=torch.qint8
    )
    acc_p40 = evaluate_model(model_p40, test_loader, device)
    lat_p40 = benchmark_inference(model_p40, test_loader, device,
                                   num_batches=args.benchmark_batches,
                                   warmup_batches=args.warmup_batches,
                                   model_label="Prune40+INT8")
    results["prune40_int8"] = {
        "method": "prune40_int8",
        "accuracy": acc_p40["accuracy"],
        "f1": acc_p40["f1"],
        "n_params": n_params,
        "nonzero_params": nonzero_p40,
        "sparsity": 1.0 - nonzero_p40 / n_params,
        **lat_p40,
        "latency_reduction": 1.0 - lat_p40["latency_ms_per_sample"] / baseline_latency,
    }
    print(f"Prune40+INT8: {lat_p40['latency_ms_per_sample']:.6f} ms/sample, "
          f"acc={acc_p40['accuracy']:.4f}, "
          f"sparsity={results['prune40_int8']['sparsity']*100:.1f}%, "
          f"Δ={results['prune40_int8']['latency_reduction']*100:+.1f}%")

    # ─── PRUNING 50% + INT8 ───
    print("\n" + "="*60)
    print("4. PRUNING 50% + INT8")
    print("="*60)
    model_p50 = DSC_CBAM_GRU(
        input_dim=args.input_dim, num_classes=args.num_classes,
        hidden_dim=args.hidden_dim, bidirectional=args.bidirectional, dropout=args.dropout
    )
    model_p50 = load_checkpoint(model_p50, args.checkpoint, device)
    prune.global_unstructured(prunable_params, pruning_method=prune.L1Unstructured, amount=0.50)
    for m, n in prunable_params:
        try:
            prune.remove(m, n)
        except ValueError:
            pass
    nonzero_p50 = count_nonzero(model_p50)
    model_p50 = torch.quantization.quantize_dynamic(
        model_p50, {nn.Linear, nn.GRU}, dtype=torch.qint8
    )
    acc_p50 = evaluate_model(model_p50, test_loader, device)
    lat_p50 = benchmark_inference(model_p50, test_loader, device,
                                   num_batches=args.benchmark_batches,
                                   warmup_batches=args.warmup_batches,
                                   model_label="Prune50+INT8")
    results["prune50_int8"] = {
        "method": "prune50_int8",
        "accuracy": acc_p50["accuracy"],
        "f1": acc_p50["f1"],
        "n_params": n_params,
        "nonzero_params": nonzero_p50,
        "sparsity": 1.0 - nonzero_p50 / n_params,
        **lat_p50,
        "latency_reduction": 1.0 - lat_p50["latency_ms_per_sample"] / baseline_latency,
    }
    print(f"Prune50+INT8: {lat_p50['latency_ms_per_sample']:.6f} ms/sample, "
          f"acc={acc_p50['accuracy']:.4f}, "
          f"sparsity={results['prune50_int8']['sparsity']*100:.1f}%, "
          f"Δ={results['prune50_int8']['latency_reduction']*100:+.1f}%")

    # ─── INT8 WITH QNNPACK ENGINE ───
    print("\n" + "="*60)
    print("5. INT8 with QNNPACK engine")
    print("="*60)
    torch.backends.quantized.engine = 'qnnpack'
    model_qnnp = DSC_CBAM_GRU(
        input_dim=args.input_dim, num_classes=args.num_classes,
        hidden_dim=args.hidden_dim, bidirectional=args.bidirectional, dropout=args.dropout
    )
    model_qnnp = load_checkpoint(model_qnnp, args.checkpoint, device)
    model_qnnp = torch.quantization.quantize_dynamic(
        model_qnnp, {nn.Linear, nn.GRU}, dtype=torch.qint8
    )
    acc_qnnp = evaluate_model(model_qnnp, test_loader, device)
    lat_qnnp = benchmark_inference(model_qnnp, test_loader, device,
                                    num_batches=args.benchmark_batches,
                                    warmup_batches=args.warmup_batches,
                                    model_label="INT8_QNNPACK")
    results["int8_qnnpack"] = {
        "method": "int8_qnnpack",
        "accuracy": acc_qnnp["accuracy"],
        "f1": acc_qnnp["f1"],
        **lat_qnnp,
        "latency_reduction": 1.0 - lat_qnnp["latency_ms_per_sample"] / baseline_latency,
    }
    print(f"INT8_QNNPACK: {lat_qnnp['latency_ms_per_sample']:.6f} ms/sample, "
          f"acc={acc_qnnp['accuracy']:.4f}, "
          f"Δ={results['int8_qnnpack']['latency_reduction']*100:+.1f}%")
    torch.backends.quantized.engine = 'x86'

    # ─── JIT SCRIPTED FP32 ───
    print("\n" + "="*60)
    print("6. JIT SCRIPTED FP32")
    print("="*60)
    model_jit = DSC_CBAM_GRU(
        input_dim=args.input_dim, num_classes=args.num_classes,
        hidden_dim=args.hidden_dim, bidirectional=args.bidirectional, dropout=args.dropout
    )
    model_jit = load_checkpoint(model_jit, args.checkpoint, device)
    scripted = torch.jit.script(model_jit)
    acc_jit = evaluate_model(model_jit, test_loader, device)
    lat_jit = benchmark_inference(scripted, test_loader, device,
                                    num_batches=args.benchmark_batches,
                                    warmup_batches=args.warmup_batches,
                                    model_label="JIT_FP32")
    results["jit_fp32"] = {
        "method": "jit_fp32",
        "accuracy": acc_jit["accuracy"],
        "f1": acc_jit["f1"],
        **lat_jit,
        "latency_reduction": 1.0 - lat_jit["latency_ms_per_sample"] / baseline_latency,
    }
    print(f"JIT_FP32: {lat_jit['latency_ms_per_sample']:.6f} ms/sample, "
          f"acc={acc_jit['accuracy']:.4f}, "
          f"Δ={results['jit_fp32']['latency_reduction']*100:+.1f}%")

    # ─── BF16 INFERENCE ───
    print("\n" + "="*60)
    print("7. BF16 INFERENCE")
    print("="*60)
    model_bf16 = DSC_CBAM_GRU(
        input_dim=args.input_dim, num_classes=args.num_classes,
        hidden_dim=args.hidden_dim, bidirectional=args.bidirectional, dropout=args.dropout
    )
    model_bf16 = load_checkpoint(model_bf16, args.checkpoint, device)
    model_bf16 = model_bf16.to(torch.bfloat16)
    # Note: BF16 on CPU may not be faster, but let's test
    acc_bf16 = evaluate_model(model_bf16, test_loader, torch.device("cpu"))
    lat_bf16 = benchmark_inference(model_bf16, test_loader, torch.device("cpu"),
                                    num_batches=args.benchmark_batches,
                                    warmup_batches=args.warmup_batches,
                                    model_label="BF16")
    results["bf16"] = {
        "method": "bf16",
        "accuracy": acc_bf16["accuracy"],
        "f1": acc_bf16["f1"],
        **lat_bf16,
        "latency_reduction": 1.0 - lat_bf16["latency_ms_per_sample"] / baseline_latency,
    }
    print(f"BF16: {lat_bf16['latency_ms_per_sample']:.6f} ms/sample, "
          f"acc={acc_bf16['accuracy']:.4f}, "
          f"Δ={results['bf16']['latency_reduction']*100:+.1f}%")

    # ─── FINAL SUMMARY ───
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"Baseline FP32: {baseline_latency:.6f} ms/sample")
    print(f"Target: 25% reduction = {baseline_latency*0.75:.6f} ms/sample")
    print()
    print(f"{'Method':<20} {'Latency':>12} {'Reduction':>10} {'Accuracy':>10} {'Status':>8}")
    print("-"*60)

    target_latency = baseline_latency * 0.75
    best = None
    for key, r in sorted(results.items(), key=lambda x: x[1].get("latency_reduction", -999), reverse=True):
        lat = r["latency_ms_per_sample"]
        red = r.get("latency_reduction", 0)
        acc = r["accuracy"]
        meets = "✓" if lat <= target_latency and acc >= 0.95 else "✗"
        print(f"{r['method']:<20} {lat:>12.6f} {red*100:>+9.1f}% {acc:>10.4f} {meets:>8}")
        if meets == "✓" and (best is None or red > best[1]):
            best = (key, red, acc, lat)

    print()
    if best:
        print(f"BEST MEETING TARGET: {best[0]} with {best[1]*100:+.1f}% reduction, acc={best[2]:.4f}")
    else:
        print("No configuration met both 25% latency reduction AND 95% accuracy targets.")
        # Find best latency reduction
        best_red = max(results.items(), key=lambda x: x[1].get("latency_reduction", -999))
        print(f"Best latency reduction: {best_red[0]} with {best_red[1].get('latency_reduction', 0)*100:+.1f}%")

    # Save results
    output_json = os.path.join(output_subdir, "full_optimization_results.json")
    with open(output_json, "w") as f:
        json.dump({
            "baseline_latency_ms": baseline_latency,
            "target_latency_ms": target_latency,
            "target_reduction": 0.25,
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nFull results saved: {output_json}")


if __name__ == "__main__":
    main()
