#!/usr/bin/env python3
"""Evaluate fixed-structure pruning + FP16 deployment on GPU."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.utils.prune as prune

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from compression_utils import (
    build_model,
    checkpoint_size_mb,
    count_nonzero_weights,
    create_test_loader,
    evaluate_model,
    load_model_checkpoint,
    save_json,
    summarize_dense_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-structure pruning + FP16 GPU benchmark.")
    parser.add_argument("--data_dir", default="../dataset_cicids17")
    parser.add_argument("--checkpoint", default="checkpoints_gru/cicids17_gru_best.pt")
    parser.add_argument("--output_dir", default="experiments/compression/fixed_fp16_gpu")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batches", type=int, default=100)
    parser.add_argument("--benchmark_batch_size", type=int, default=1024)
    parser.add_argument("--benchmark_steps", type=int, default=400)
    parser.add_argument("--benchmark_warmup", type=int, default=80)
    parser.add_argument("--input_dim", type=int, default=18)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--bidirectional", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--gru_prune_amount", type=float, default=0.35)
    parser.add_argument("--fc_prune_amount", type=float, default=0.20)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def apply_targeted_pruning(model: torch.nn.Module, gru_amount: float, fc_amount: float) -> None:
    prunable = [
        (model.gru, "weight_ih_l0", gru_amount),
        (model.gru, "weight_hh_l0", gru_amount),
        (model.fc[0], "weight", fc_amount),
    ]
    if model.bidirectional:
        prunable.extend(
            [
                (model.gru, "weight_ih_l0_reverse", gru_amount),
                (model.gru, "weight_hh_l0_reverse", gru_amount),
            ]
        )

    for module, parameter_name, amount in prunable:
        prune.l1_unstructured(module, name=parameter_name, amount=amount)
        prune.remove(module, parameter_name)


def benchmark_gpu(
    model: torch.nn.Module,
    batch_size: int,
    steps: int,
    warmup: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    if dtype == torch.float16:
        model = model.half()
        example = torch.randn(batch_size, 10, 18, device=device, dtype=torch.float16)
    else:
        example = torch.randn(batch_size, 10, 18, device=device, dtype=torch.float32)

    torch.cuda.empty_cache()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(example)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(steps):
            _ = model(example)
        torch.cuda.synchronize(device)
        end = time.perf_counter()

    total = end - start
    return {
        "latency_ms_per_batch": float(total * 1000.0 / steps),
        "latency_ms_per_sample": float(total * 1000.0 / steps / batch_size),
        "throughput_samples_per_sec": float(batch_size * steps / total),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This script is intended for GPU FP16 evaluation. Use --device cuda.")

    os.makedirs(args.output_dir, exist_ok=True)
    test_loader = create_test_loader(args.data_dir, args.batch_size)

    baseline_model = load_model_checkpoint(build_model(args), args.checkpoint, device)
    baseline_metrics = evaluate_model(baseline_model, test_loader, device)
    baseline_latency = benchmark_gpu(
        baseline_model,
        batch_size=args.benchmark_batch_size,
        steps=args.benchmark_steps,
        warmup=args.benchmark_warmup,
        dtype=torch.float32,
        device=device,
    )
    baseline_summary = summarize_dense_model(baseline_model.cpu())

    compressed_model = load_model_checkpoint(build_model(args), args.checkpoint, device)
    apply_targeted_pruning(compressed_model, args.gru_prune_amount, args.fc_prune_amount)
    compressed_cpu = compressed_model.cpu()
    total_elements, nonzero_elements = count_nonzero_weights(compressed_cpu)
    compressed_model = compressed_cpu.to(device)

    compressed_metrics = evaluate_model(compressed_model, test_loader, device)
    compressed_latency = benchmark_gpu(
        compressed_model,
        batch_size=args.benchmark_batch_size,
        steps=args.benchmark_steps,
        warmup=args.benchmark_warmup,
        dtype=torch.float16,
        device=device,
    )

    output_checkpoint = os.path.join(args.output_dir, "cicids17_gru_fixed_prune_fp16.pt")
    torch.save(compressed_model.cpu().state_dict(), output_checkpoint)

    parameter_reduction_ratio = float(1.0 - (nonzero_elements / total_elements if total_elements else 0.0))
    latency_reduction_ratio = float(
        1.0
        - (
            compressed_latency["latency_ms_per_sample"]
            / baseline_latency["latency_ms_per_sample"]
            if baseline_latency["latency_ms_per_sample"] > 0
            else 0.0
        )
    )

    summary = {
        "compression": "fixed_structure_targeted_pruning_plus_fp16_gpu",
        "checkpoint_in": args.checkpoint,
        "checkpoint_out": output_checkpoint,
        "baseline": {
            **baseline_metrics,
            **baseline_latency,
            **baseline_summary,
            "dtype": "fp32",
            "checkpoint_size_mb": checkpoint_size_mb(args.checkpoint),
        },
        "compressed": {
            **compressed_metrics,
            **compressed_latency,
            "dtype": "fp16",
            "effective_parameter_count": int(nonzero_elements),
            "effective_sparsity": parameter_reduction_ratio,
            "parameter_reduction_ratio": parameter_reduction_ratio,
            "checkpoint_size_mb": checkpoint_size_mb(output_checkpoint),
        },
        "config": {
            "gru_prune_amount": args.gru_prune_amount,
            "fc_prune_amount": args.fc_prune_amount,
            "benchmark_batch_size": args.benchmark_batch_size,
            "benchmark_steps": args.benchmark_steps,
            "benchmark_warmup": args.benchmark_warmup,
            "benchmark_backend": "gpu_eager",
        },
        "meets_target": {
            "accuracy_gte_0_95": bool(compressed_metrics["accuracy"] >= 0.95),
            "parameter_reduction_gte_0_25": bool(parameter_reduction_ratio >= 0.25),
            "latency_reduction_gte_0_20": bool(latency_reduction_ratio >= 0.20),
        },
        "latency_reduction_ratio": latency_reduction_ratio,
    }

    output_json = os.path.join(args.output_dir, "cicids17_gru_fixed_prune_fp16_summary.json")
    save_json(output_json, summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
