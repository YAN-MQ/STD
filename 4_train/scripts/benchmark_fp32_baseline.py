#!/usr/bin/env python3
"""
Dedicated FP32 baseline benchmark with proper methodology.
This establishes a reliable baseline for comparison.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import create_dataloaders, load_npz_data
from src.models import DSC_CBAM_GRU, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reliable FP32 baseline benchmark")
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
) -> dict[str, Any]:
    """Robust inference benchmark with extended warmup."""
    model.eval()

    # Extended warmup to reach steady state
    warmup_done = False
    total_seconds = 0.0
    measured_batches = 0
    measured_samples = 0

    with torch.no_grad():
        for batch_idx, (features, _) in enumerate(loader):
            features = features.to(device)

            # First batch triggers model setup
            if not warmup_done:
                _ = model(features)
                if batch_idx >= warmup_batches:
                    warmup_done = True
                continue

            start = time.perf_counter()
            _ = model(features)
            elapsed = time.perf_counter() - start

            total_seconds += elapsed
            measured_batches += 1
            measured_samples += features.size(0)

            if measured_batches >= num_batches:
                break

    return {
        "latency_ms_per_sample": float(total_seconds * 1000.0 / measured_samples),
        "latency_ms_per_batch": float(total_seconds * 1000.0 / measured_batches),
        "throughput_samples_per_sec": float(measured_samples / total_seconds),
        "total_inference_seconds": float(total_seconds),
        "measured_batches": measured_batches,
        "measured_samples": measured_samples,
    }


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

    # Build and load FP32 baseline
    model = DSC_CBAM_GRU(
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
    )
    model = load_checkpoint(model, args.checkpoint, device)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    # Accuracy evaluation
    print("\nEvaluating accuracy...")
    acc_metrics = evaluate_model(model, test_loader, device)
    print(f"Accuracy: {acc_metrics['accuracy']:.4f}, F1: {acc_metrics['f1']:.4f}")

    # Warmup then benchmark
    print(f"\nWarming up ({args.warmup_batches} batches)...")
    warmup_loader = iter(test_loader)
    for _ in range(args.warmup_batches):
        features, _ = next(warmup_loader)
        with torch.no_grad():
            _ = model(features.to(device))

    print(f"Benchmarking ({args.benchmark_batches} batches)...")
    lat_metrics = benchmark_inference(model, test_loader, device, num_batches=args.benchmark_batches, warmup_batches=0)

    print(f"\n{'='*50}")
    print(f"FP32 BASELINE RESULTS (hidden_dim={args.hidden_dim})")
    print(f"{'='*50}")
    print(f"Latency per sample: {lat_metrics['latency_ms_per_sample']:.6f} ms")
    print(f"Latency per batch:  {lat_metrics['latency_ms_per_batch']:.4f} ms")
    print(f"Throughput:         {lat_metrics['throughput_samples_per_sec']:.1f} samples/sec")
    print(f"Accuracy:          {acc_metrics['accuracy']:.4f}")

    results = {
        "config": "fp32_baseline",
        "hidden_dim": args.hidden_dim,
        "bidirectional": args.bidirectional,
        "batch_size": args.batch_size,
        "num_params": n_params,
        "accuracy": acc_metrics["accuracy"],
        "f1": acc_metrics["f1"],
        "precision": acc_metrics["precision"],
        "recall": acc_metrics["recall"],
        **lat_metrics,
    }

    out_path = os.path.join(output_subdir, "fp32_baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
