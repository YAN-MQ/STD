#!/usr/bin/env python3
"""
Train smaller hidden_dim models AND benchmark all optimizations together.
Goal: Find model config + optimization that achieves 25%+ latency reduction.
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
from src.training import get_optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train smaller model + comprehensive benchmark")
    parser.add_argument("--data_dir", default="../dataset_cicids17")
    parser.add_argument("--checkpoint", default="checkpoints_gru/cicids17_gru_best.pt")
    parser.add_argument("--output_dir", default="experiments/compression")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--benchmark_batches", type=int, default=300)
    parser.add_argument("--warmup_batches", type=int, default=50)
    parser.add_argument("--input_dim", type=int, default=18)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--hidden_dim_base", type=int, default=64)
    parser.add_argument("--bidirectional", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--train_epochs", type=int, default=30)
    return parser.parse_args()


def create_test_loader(data_dir: str, batch_size: int):
    x_train, y_train, x_val, y_val, x_test, y_test = load_npz_data(data_dir)
    _, _, test_loader = create_dataloaders(
        x_train, y_train, x_val, y_val, x_test, y_test,
        batch_size=batch_size, num_workers=0, pin_memory=False,
    )
    return test_loader, x_test, y_test


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
    model: nn.Module, loader, device: torch.device,
    num_batches: int = 300, warmup_batches: int = 50,
) -> dict[str, Any]:
    model.eval()
    warmup_loader = iter(loader)
    for _ in range(warmup_batches):
        try:
            features, _ = next(warmup_loader)
        except StopIteration:
            warmup_loader = iter(loader)
            features, _ = next(warmup_loader)
        with torch.no_grad():
            _ = model(features.to(device))

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
        "throughput": float(measured_samples / total_seconds),
    }


def count_nonzero(model: nn.Module) -> int:
    return sum(int(torch.count_nonzero(p).item()) for p in model.parameters() if p.requires_grad)


def train_model(
    hidden_dim: int,
    data_dir: str,
    epochs: int,
    batch_size: int,
    bidirectional: bool,
    device: torch.device,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[DSC_CBAM_GRU, dict[str, Any], float]:
    """Train a model with given hidden_dim. Returns model, test metrics, training time."""
    x_train, y_train, x_val, y_val, x_test, y_test = load_npz_data(data_dir)

    train_loader, val_loader, test_loader = create_dataloaders(
        x_train, y_train, x_val, y_val, x_test, y_test,
        batch_size=batch_size, num_workers=0, pin_memory=False,
    )

    model = DSC_CBAM_GRU(
        input_dim=18, num_classes=3,
        hidden_dim=hidden_dim, bidirectional=bidirectional, dropout=0.3,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, "adamw", lr=1e-3, weight_decay=1e-2)

    best_val_acc = 0.0
    best_state = None
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features.to(device))
                preds = outputs.argmax(dim=1)
                correct += (preds == labels.to(device)).sum().item()
                total += labels.size(0)
        val_acc = correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: loss={total_loss/len(train_loader):.4f}, val_acc={val_acc:.4f}")

    train_time = time.time() - start_time

    # Load best model
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # Evaluate on test set
    test_metrics = evaluate_model(model, test_loader, device)

    return model, test_metrics, train_time


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.num_threads)
    device = torch.device("cpu")

    print(f"PyTorch: {torch.__version__}")
    print(f"Threads: {torch.get_num_threads()}")

    output_subdir = os.path.join(args.output_dir, "compression")
    os.makedirs(output_subdir, exist_ok=True)

    test_loader, _, _ = create_test_loader(args.data_dir, args.batch_size)

    all_results = {}

    # ─── FP32 BASELINE (hidden_dim=64, from checkpoint) ───
    print("\n" + "="*60)
    print("BASELINE: hidden_dim=64 (from trained checkpoint)")
    print("="*60)
    model_base = DSC_CBAM_GRU(
        input_dim=args.input_dim, num_classes=args.num_classes,
        hidden_dim=args.hidden_dim_base, bidirectional=args.bidirectional, dropout=args.dropout
    )
    model_base = load_checkpoint(model_base, args.checkpoint, device)
    n_params_base = count_parameters(model_base)
    acc_base = evaluate_model(model_base, test_loader, device)
    lat_base = benchmark_inference(model_base, test_loader, device,
                                    num_batches=args.benchmark_batches,
                                    warmup_batches=args.warmup_batches)
    all_results["fp32_hdim64"] = {
        "hidden_dim": args.hidden_dim_base, "method": "fp32",
        "accuracy": acc_base["accuracy"], "f1": acc_base["f1"],
        "n_params": n_params_base, **lat_base,
        "latency_reduction": 0.0
    }
    print(f"  Baseline: {lat_base['latency_ms_per_sample']:.6f} ms/sample, "
          f"acc={acc_base['accuracy']:.4f}, params={n_params_base:,}")
    baseline_latency = lat_base["latency_ms_per_sample"]
    target_latency = baseline_latency * 0.75

    # ─── JIT BASELINE ───
    print("\n" + "="*60)
    print("BASELINE: JIT FP32 (hidden_dim=64)")
    print("="*60)
    scripted_base = torch.jit.script(model_base)
    lat_jit_base = benchmark_inference(scripted_base, test_loader, device,
                                        num_batches=args.benchmark_batches,
                                        warmup_batches=args.warmup_batches)
    all_results["jit_fp32_hdim64"] = {
        "hidden_dim": args.hidden_dim_base, "method": "jit_fp32",
        "accuracy": acc_base["accuracy"], "f1": acc_base["f1"],
        "n_params": n_params_base, **lat_jit_base,
        "latency_reduction": 1.0 - lat_jit_base['latency_ms_per_sample']/baseline_latency
    }
    print(f"  JIT FP32: {lat_jit_base['latency_ms_per_sample']:.6f} ms/sample, "
          f"Δ={all_results['jit_fp32_hdim64']['latency_reduction']*100:+.1f}%")

    # ─── TRAIN SMALLER MODELS ───
    print("\n" + "="*60)
    print("TRAINING SMALLER MODELS")
    print("="*60)

    trained_models = {}
    for hdim in [48, 40, 32]:
        print(f"\n--- Training hidden_dim={hdim} ---")
        torch.set_num_threads(args.num_threads)
        m, metrics, t_time = train_model(
            hidden_dim=hdim,
            data_dir=args.data_dir,
            epochs=args.train_epochs,
            batch_size=args.batch_size,
            bidirectional=args.bidirectional,
            device=device,
        )
        n_params_hdim = count_parameters(m)
        lat_hdim = benchmark_inference(m, test_loader, device,
                                        num_batches=args.benchmark_batches,
                                        warmup_batches=args.warmup_batches)
        lat_red_hdim = 1.0 - lat_hdim['latency_ms_per_sample'] / baseline_latency

        trained_models[hdim] = {
            "model": m,
            "metrics": metrics,
            "latency": lat_hdim,
            "n_params": n_params_hdim,
            "train_time": t_time,
            "lat_red": lat_red_hdim,
        }
        all_results[f"fp32_hdim{hdim}"] = {
            "hidden_dim": hdim, "method": f"fp32",
            "accuracy": metrics["accuracy"], "f1": metrics["f1"],
            "n_params": n_params_hdim, "train_time": t_time,
            **lat_hdim, "latency_reduction": lat_red_hdim
        }
        print(f"  hdim={hdim}: {lat_hdim['latency_ms_per_sample']:.6f} ms/sample, "
              f"Δ={lat_red_hdim*100:+.1f}%, acc={metrics['accuracy']:.4f}, "
              f"params={n_params_hdim:,}")

        # JIT version
        scripted_hdim = torch.jit.script(m)
        lat_jit_hdim = benchmark_inference(scripted_hdim, test_loader, device,
                                             num_batches=args.benchmark_batches,
                                             warmup_batches=args.warmup_batches)
        lat_red_jit = 1.0 - lat_jit_hdim['latency_ms_per_sample'] / baseline_latency
        all_results[f"jit_fp32_hdim{hdim}"] = {
            "hidden_dim": hdim, "method": f"jit_fp32",
            "accuracy": metrics["accuracy"], "f1": metrics["f1"],
            "n_params": n_params_hdim, **lat_jit_hdim, "latency_reduction": lat_red_jit
        }
        print(f"  JIT hdim={hdim}: {lat_jit_hdim['latency_ms_per_sample']:.6f} ms/sample, "
              f"Δ={lat_red_jit*100:+.1f}%")

        del m, scripted_hdim
        gc.collect()

    # ─── TRAINED MODELS + JIT + INT8 COMBINED ───
    print("\n" + "="*60)
    print("BEST TRAINED MODELS + INT8")
    print("="*60)
    # Find the best trained model by latency reduction
    best_trained_hdim = max(trained_models.keys(),
                            key=lambda h: trained_models[h]["lat_red"])
    print(f"Best trained: hidden_dim={best_trained_hdim} "
          f"(Δ={trained_models[best_trained_hdim]['lat_red']*100:+.1f}%)")

    # Try INT8 on the best trained model
    best_model_data = trained_models[best_trained_hdim]
    model_int8 = DSC_CBAM_GRU(
        input_dim=args.input_dim, num_classes=args.num_classes,
        hidden_dim=best_trained_hdim, bidirectional=args.bidirectional, dropout=args.dropout
    )
    model_int8.load_state_dict({k: v.cpu() for k, v in best_model_data["model"].state_dict().items()})
    model_int8.to(device)
    model_int8 = torch.quantization.quantize_dynamic(
        model_int8, {nn.Linear, nn.GRU}, dtype=torch.qint8
    )
    lat_int8_trained = benchmark_inference(model_int8, test_loader, device,
                                             num_batches=args.benchmark_batches,
                                             warmup_batches=args.warmup_batches)
    lat_red_int8_trained = 1.0 - lat_int8_trained['latency_ms_per_sample'] / baseline_latency
    acc_int8_trained = evaluate_model(model_int8, test_loader, device)
    all_results[f"int8_hdim{best_trained_hdim}"] = {
        "hidden_dim": best_trained_hdim, "method": f"int8",
        "accuracy": acc_int8_trained["accuracy"], "f1": acc_int8_trained["f1"],
        "n_params": best_model_data["n_params"], **lat_int8_trained,
        "latency_reduction": lat_red_int8_trained
    }
    print(f"  INT8 hdim={best_trained_hdim}: {lat_int8_trained['latency_ms_per_sample']:.6f} ms/sample, "
          f"Δ={lat_red_int8_trained*100:+.1f}%, acc={acc_int8_trained['accuracy']:.4f}")

    # ─── FINAL SUMMARY ───
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print(f"Baseline FP32 (hdim=64): {baseline_latency:.6f} ms/sample")
    print(f"Target (25% reduction): {target_latency:.6f} ms/sample")
    print()
    print(f"{'Method':<25} {'hdim':>5} {'Latency':>12} {'Reduction':>10} {'Accuracy':>10} {'Params':>8}")
    print("-"*70)

    sorted_results = sorted(all_results.items(),
                           key=lambda x: x[1].get("latency_reduction", -999), reverse=True)
    for key, r in sorted_results:
        lat = r["latency_ms_per_sample"]
        red = r.get("latency_reduction", 0)
        acc = r["accuracy"]
        hdim = r.get("hidden_dim", "?")
        nparams = r.get("n_params", 0)
        meets = "✓" if lat <= target_latency and acc >= 0.95 else "✗"
        print(f"{r['method']:<25} {hdim:>5} {lat:>12.6f} {red*100:>+9.1f}% {acc:>9.4f} "
              f"{nparams:>8,} {meets}")

    # Save best models
    best_result = sorted_results[0]
    print(f"\nBest overall: {best_result[0]} with {best_result[1].get('latency_reduction', 0)*100:+.1f}% reduction")

    # Save results
    serializable_results = {}
    for k, v in all_results.items():
        v_copy = {kk: vv for kk, vv in v.items() if kk != "model"}
        if "train_time" in v_copy:
            v_copy["train_time"] = float(v_copy["train_time"])
        serializable_results[k] = v_copy

    output_json = os.path.join(output_subdir, "final_optimization_results.json")
    with open(output_json, "w") as f:
        json.dump({
            "baseline_latency_ms": float(baseline_latency),
            "target_latency_ms": float(target_latency),
            "target_reduction": 0.25,
            "results": serializable_results,
            "best_config": best_result[0],
            "best_latency_reduction": float(best_result[1].get("latency_reduction", 0)),
        }, f, indent=2, default=str)
    print(f"\nSaved: {output_json}")


if __name__ == "__main__":
    main()
