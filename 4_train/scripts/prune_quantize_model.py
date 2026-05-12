#!/usr/bin/env python3
"""Joint pruning + dynamic quantization for DSC-CBAM-GRU."""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from compression_utils import (
    benchmark_inference,
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
    parser = argparse.ArgumentParser(
        description="Apply targeted pruning followed by dynamic INT8 quantization to DSC-CBAM-GRU"
    )
    parser.add_argument("--data_dir", default="../dataset_cicids17")
    parser.add_argument("--checkpoint", default="checkpoints_gru/cicids17_gru_best.pt")
    parser.add_argument("--output_dir", default="experiments/compression/combined")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--benchmark_batches", type=int, default=100)
    parser.add_argument("--benchmark_batch_size", type=int, default=512)
    parser.add_argument("--benchmark_threads", type=int, default=4)
    parser.add_argument("--input_dim", type=int, default=18)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--bidirectional", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--gru_prune_amount",
        type=float,
        default=0.35,
        help="Unstructured pruning ratio applied to GRU input/hidden weights",
    )
    parser.add_argument(
        "--fc_prune_amount",
        type=float,
        default=0.20,
        help="Unstructured pruning ratio applied to the first FC layer",
    )
    return parser.parse_args()


def apply_targeted_pruning(model: torch.nn.Module, gru_amount: float, fc_amount: float) -> None:
    """Prune the dominant dense weights while leaving sensitive conv/attention layers intact."""
    prunable: list[tuple[torch.nn.Module, str, float]] = [
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


def benchmark_torchscript_cpu(
    model: torch.nn.Module,
    data_loader,
    example_input: torch.Tensor,
    num_batches: int = 100,
    warmup_batches: int = 10,
) -> dict[str, float]:
    """Benchmark a frozen TorchScript model on CPU for deployment-oriented latency."""
    scripted = torch.jit.freeze(torch.jit.trace(model.eval(), example_input).eval())
    total_seconds = 0.0
    measured_batches = 0
    measured_samples = 0

    with torch.no_grad():
        for batch_idx, (features, _) in enumerate(data_loader):
            start = time.perf_counter()
            _ = scripted(features)
            elapsed = time.perf_counter() - start

            if batch_idx >= warmup_batches:
                total_seconds += elapsed
                measured_batches += 1
                measured_samples += features.size(0)

            if measured_batches >= num_batches:
                break

    return {
        "inference_time_sec": float(total_seconds),
        "latency_ms_per_batch": float(total_seconds * 1000.0 / measured_batches),
        "latency_ms_per_sample": float(total_seconds * 1000.0 / measured_samples),
        "throughput_samples_per_sec": float(measured_samples / total_seconds),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    test_loader = create_test_loader(args.data_dir, args.batch_size)
    benchmark_loader = create_test_loader(args.data_dir, args.benchmark_batch_size)
    torch.set_num_threads(args.benchmark_threads)
    example_input = torch.randn(args.benchmark_batch_size, 10, args.input_dim)

    baseline_model = build_model(args)
    baseline_model = load_model_checkpoint(baseline_model, args.checkpoint, device)
    baseline_metrics = evaluate_model(baseline_model, test_loader, device)
    baseline_latency = benchmark_torchscript_cpu(
        baseline_model,
        benchmark_loader,
        example_input,
        num_batches=args.benchmark_batches,
    )
    baseline_summary = summarize_dense_model(baseline_model)

    compressed_model = build_model(args)
    compressed_model = load_model_checkpoint(compressed_model, args.checkpoint, device)
    apply_targeted_pruning(
        compressed_model,
        gru_amount=args.gru_prune_amount,
        fc_amount=args.fc_prune_amount,
    )
    total_elements, nonzero_elements = count_nonzero_weights(compressed_model)

    quantized_model = torch.quantization.quantize_dynamic(
        compressed_model,
        {nn.Linear, nn.GRU},
        dtype=torch.qint8,
    )
    quantized_model.eval()

    output_checkpoint = os.path.join(args.output_dir, "cicids17_gru_prune_quant_int8.pt")
    torch.save(
        {
            "compression": "targeted_pruning_plus_dynamic_quantization",
            "config": {
                "input_dim": args.input_dim,
                "num_classes": args.num_classes,
                "hidden_dim": args.hidden_dim,
                "bidirectional": args.bidirectional,
                "dropout": args.dropout,
                "gru_prune_amount": args.gru_prune_amount,
                "fc_prune_amount": args.fc_prune_amount,
            },
            "state_dict": quantized_model.state_dict(),
        },
        output_checkpoint,
    )

    compressed_metrics = evaluate_model(quantized_model, test_loader, device)
    compressed_latency = benchmark_torchscript_cpu(
        quantized_model,
        benchmark_loader,
        example_input,
        num_batches=args.benchmark_batches,
    )

    effective_parameter_count = int(nonzero_elements)
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
        "compression": "targeted_pruning_plus_dynamic_quantization",
        "checkpoint_in": args.checkpoint,
        "checkpoint_out": output_checkpoint,
        "baseline": {
            **baseline_metrics,
            **baseline_latency,
            **baseline_summary,
            "checkpoint_size_mb": checkpoint_size_mb(args.checkpoint),
        },
        "compressed": {
            **compressed_metrics,
            **compressed_latency,
            "effective_parameter_count": effective_parameter_count,
            "effective_sparsity": parameter_reduction_ratio,
            "parameter_reduction_ratio": parameter_reduction_ratio,
            "checkpoint_size_mb": checkpoint_size_mb(output_checkpoint),
        },
        "config": {
            "gru_prune_amount": args.gru_prune_amount,
            "fc_prune_amount": args.fc_prune_amount,
            "benchmark_batch_size": args.benchmark_batch_size,
            "benchmark_threads": args.benchmark_threads,
            "benchmark_backend": "torchscript_cpu",
        },
        "meets_target": {
            "accuracy_gte_0_95": bool(compressed_metrics["accuracy"] >= 0.95),
            "parameter_reduction_gte_0_25": bool(parameter_reduction_ratio >= 0.25),
            "latency_reduction_gte_0_20": bool(latency_reduction_ratio >= 0.20),
        },
        "latency_reduction_ratio": latency_reduction_ratio,
    }

    output_json = os.path.join(args.output_dir, "cicids17_gru_prune_quant_int8_summary.json")
    save_json(output_json, summary)

    print("=" * 60)
    print("Joint Compression Summary")
    print("=" * 60)
    print(f"Compressed accuracy         : {compressed_metrics['accuracy']:.4f}")
    print(f"Compressed F1               : {compressed_metrics['f1']:.4f}")
    print(f"Effective sparsity          : {parameter_reduction_ratio:.4f}")
    print(f"Latency reduction ratio     : {latency_reduction_ratio:.4f}")
    print(f"Compressed checkpoint MB    : {summary['compressed']['checkpoint_size_mb']:.4f}")
    print(f"Saved checkpoint            : {output_checkpoint}")
    print(f"Saved summary               : {output_json}")


if __name__ == "__main__":
    main()
