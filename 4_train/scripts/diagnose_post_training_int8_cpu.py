#!/usr/bin/env python3
"""Diagnose CPU latency regression for formal tuned DSC-CBAM-GRU compression.

Measures these variants on the same tuned formal checkpoint:
- fp32 baseline
- pruned-only fp32
- dynamic INT8 only
- pruned + dynamic INT8

Across requested quant engines.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from compression_utils import (  # noqa: E402
    apply_targeted_pruning,
    benchmark_torchscript_cpu,
    build_model,
    create_test_loader,
    evaluate_model,
    load_model_checkpoint,
    save_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose CPU latency regression")
    p.add_argument("--data_dir", default="../dataset_cicids17")
    p.add_argument("--checkpoint", default="checkpoints_gru_formal_tuned/cicids17_gru_best.pt")
    p.add_argument("--output_dir", default="experiments/compression/post_training_int8_cpu_formal_tuned")
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--benchmark_batch_size", type=int, default=512)
    p.add_argument("--benchmark_steps", type=int, default=200)
    p.add_argument("--benchmark_warmup", type=int, default=40)
    p.add_argument("--benchmark_threads", type=int, default=4)
    p.add_argument("--gru_prune_amount", type=float, default=0.35)
    p.add_argument("--fc_prune_amount", type=float, default=0.20)
    p.add_argument("--input_dim", type=int, default=18)
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--bidirectional", action="store_true", default=False)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--conv_dim", type=int, default=16)
    p.add_argument("--dsc_dim", type=int, default=48)
    p.add_argument("--seq_len", type=int, default=10)
    p.add_argument("--engines", nargs="+", default=["x86", "onednn", "fbgemm"])
    return p.parse_args()


def load_fp32_model(args):
    return load_model_checkpoint(build_model(args), args.checkpoint, torch.device("cpu"))


def bench(model, args):
    return benchmark_torchscript_cpu(
        model,
        batch_size=args.benchmark_batch_size,
        input_dim=args.input_dim,
        seq_len=args.seq_len,
        threads=args.benchmark_threads,
        steps=args.benchmark_steps,
        warmup=args.benchmark_warmup,
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    test_loader = create_test_loader(args.data_dir, args.eval_batch_size)

    results: dict[str, dict] = {}

    # Baseline
    baseline = load_fp32_model(args)
    results["fp32_baseline"] = {
        "metrics": evaluate_model(baseline, test_loader, torch.device("cpu")),
        "benchmark": bench(baseline, args),
    }

    # Pruned only
    pruned = load_fp32_model(args)
    apply_targeted_pruning(pruned, gru_amount=args.gru_prune_amount, fc_amount=args.fc_prune_amount)
    results["pruned_fp32"] = {
        "metrics": evaluate_model(pruned, test_loader, torch.device("cpu")),
        "benchmark": bench(pruned, args),
    }

    # Quantized only / pruned+quantized per engine
    for engine in args.engines:
        if engine not in torch.backends.quantized.supported_engines:
            continue
        torch.backends.quantized.engine = engine

        quant_only = load_fp32_model(args)
        quant_only = torch.quantization.quantize_dynamic(quant_only, {nn.Linear, nn.GRU}, dtype=torch.qint8)
        results[f"quant_only_{engine}"] = {
            "metrics": evaluate_model(quant_only, test_loader, torch.device("cpu")),
            "benchmark": bench(quant_only, args),
        }

        pruned_quant = load_fp32_model(args)
        apply_targeted_pruning(pruned_quant, gru_amount=args.gru_prune_amount, fc_amount=args.fc_prune_amount)
        pruned_quant = torch.quantization.quantize_dynamic(pruned_quant, {nn.Linear, nn.GRU}, dtype=torch.qint8)
        results[f"pruned_quant_{engine}"] = {
            "metrics": evaluate_model(pruned_quant, test_loader, torch.device("cpu")),
            "benchmark": bench(pruned_quant, args),
        }

    out = Path(args.output_dir) / "cpu_latency_diagnosis.json"
    save_json(str(out), results)
    print(f"saved {out}")
    for name, payload in results.items():
        print(name, payload['benchmark']['latency_ms_per_sample'], payload['metrics']['accuracy'])


if __name__ == "__main__":
    main()
