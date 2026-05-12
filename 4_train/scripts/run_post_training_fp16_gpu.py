#!/usr/bin/env python3
"""Formal post-training pruning + FP16 GPU experiment based on an existing checkpoint."""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from compression_utils import (
    apply_targeted_pruning,
    benchmark_gpu_eager,
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
    parser = argparse.ArgumentParser(description="Post-training pruning + FP16 experiment on GPU.")
    parser.add_argument("--data_dir", default="../dataset_cicids17")
    parser.add_argument("--checkpoint", default="checkpoints_gru/cicids17_gru_best.pt")
    parser.add_argument("--output_dir", default="experiments/compression/post_training_fp16_gpu")
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--benchmark_batch_size", type=int, default=1024)
    parser.add_argument("--benchmark_steps", type=int, default=400)
    parser.add_argument("--benchmark_warmup", type=int, default=80)
    parser.add_argument("--gru_prune_amount", type=float, default=0.35)
    parser.add_argument("--fc_prune_amount", type=float, default=0.20)
    parser.add_argument("--input_dim", type=int, default=18)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--bidirectional", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("run_post_training_fp16_gpu.py requires a CUDA device")

    os.makedirs(args.output_dir, exist_ok=True)
    test_loader = create_test_loader(args.data_dir, args.eval_batch_size)

    baseline_model = load_model_checkpoint(build_model(args), args.checkpoint, device)
    baseline_metrics = evaluate_model(baseline_model, test_loader, device)
    baseline_latency = benchmark_gpu_eager(
        baseline_model,
        batch_size=args.benchmark_batch_size,
        input_dim=args.input_dim,
        seq_len=args.seq_len,
        dtype=torch.float32,
        device=device,
        steps=args.benchmark_steps,
        warmup=args.benchmark_warmup,
    )
    baseline_summary = summarize_dense_model(baseline_model.cpu())

    compressed_model = load_model_checkpoint(build_model(args), args.checkpoint, device)
    apply_targeted_pruning(
        compressed_model,
        gru_amount=args.gru_prune_amount,
        fc_amount=args.fc_prune_amount,
    )
    compressed_cpu = compressed_model.cpu()
    total_elements, nonzero_elements = count_nonzero_weights(compressed_cpu)
    compressed_model = compressed_cpu.to(device)

    compressed_metrics = evaluate_model(compressed_model, test_loader, device)
    compressed_latency = benchmark_gpu_eager(
        compressed_model,
        batch_size=args.benchmark_batch_size,
        input_dim=args.input_dim,
        seq_len=args.seq_len,
        dtype=torch.float16,
        device=device,
        steps=args.benchmark_steps,
        warmup=args.benchmark_warmup,
    )

    output_checkpoint = os.path.join(args.output_dir, "cicids17_gru_post_training_fp16.pt")
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
        "experiment": "post_training_fp16_gpu",
        "checkpoint_in": args.checkpoint,
        "checkpoint_out": output_checkpoint,
        "baseline": {
            **baseline_metrics,
            **baseline_latency,
            **baseline_summary,
            "dtype": "fp32",
            "device": "cuda",
            "checkpoint_size_mb": checkpoint_size_mb(args.checkpoint),
        },
        "compressed": {
            **compressed_metrics,
            **compressed_latency,
            "dtype": "fp16",
            "device": "cuda",
            "effective_parameter_count": int(nonzero_elements),
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
        "latency_reduction_ratio": latency_reduction_ratio,
        "meets_target": {
            "accuracy_gte_0_95": bool(compressed_metrics["accuracy"] >= 0.95),
            "parameter_reduction_gte_0_25": bool(parameter_reduction_ratio >= 0.25),
            "latency_reduction_gte_0_20": bool(latency_reduction_ratio >= 0.20),
        },
    }

    output_json = os.path.join(args.output_dir, "cicids17_gru_post_training_fp16_summary.json")
    save_json(output_json, summary)
    print(f"Saved checkpoint: {output_checkpoint}")
    print(f"Saved summary   : {output_json}")


if __name__ == "__main__":
    main()
