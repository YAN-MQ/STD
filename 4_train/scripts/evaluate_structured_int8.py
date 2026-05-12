#!/usr/bin/env python3
"""Evaluate structured lightweight checkpoints with dynamic INT8 benchmarking."""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import create_dataloaders, load_npz_data
from src.models import DSC_CBAM_GRU, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate structured model + INT8 latency.")
    parser.add_argument("--dataset_dir", default="../dataset_cicids17")
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--small_ckpt", required=True)
    parser.add_argument("--small_hidden_dim", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--engine", default="onednn")
    return parser.parse_args()


def load_model(hidden_dim: int, ckpt: str) -> nn.Module:
    model = DSC_CBAM_GRU(
        input_dim=18,
        num_classes=3,
        hidden_dim=hidden_dim,
        bidirectional=False,
        dropout=0.3,
    )
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate(model: nn.Module, test_loader) -> tuple[float, float]:
    y_true = []
    y_pred = []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb).argmax(1)
            y_true.append(yb)
            y_pred.append(preds)
    y = torch.cat(y_true).numpy()
    p = torch.cat(y_pred).numpy()
    acc = float((y == p).mean())
    f1_scores = []
    for c in np.unique(y):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1_scores.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return acc, float(np.mean(f1_scores))


def benchmark(model: nn.Module, batch_size: int, threads: int, repeats: int, warmup: int) -> float:
    torch.set_num_threads(threads)
    example = torch.randn(batch_size, 10, 18)
    with torch.no_grad():
        scripted = torch.jit.trace(model, example)
        scripted = torch.jit.optimize_for_inference(scripted)
        for _ in range(warmup):
            scripted(example)
        start = time.perf_counter()
        for _ in range(repeats):
            scripted(example)
        end = time.perf_counter()
    return (end - start) / repeats / batch_size * 1000.0


def main() -> None:
    args = parse_args()
    x_train, y_train, x_val, y_val, x_test, y_test = load_npz_data(args.dataset_dir)
    _, _, test_loader = create_dataloaders(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
    )

    base = load_model(64, args.base_ckpt)
    small = load_model(args.small_hidden_dim, args.small_ckpt)

    if args.engine not in torch.backends.quantized.supported_engines:
        raise ValueError(f"Unsupported quantized engine: {args.engine}")

    torch.backends.quantized.engine = args.engine
    small_q = torch.quantization.quantize_dynamic(
        load_model(args.small_hidden_dim, args.small_ckpt),
        {nn.GRU, nn.Linear},
        dtype=torch.qint8,
    )

    base_acc, base_f1 = evaluate(base, test_loader)
    small_acc, small_f1 = evaluate(small, test_loader)
    quant_acc, quant_f1 = evaluate(small_q, test_loader)

    base_lat = benchmark(base, args.batch_size, args.threads, args.repeats, args.warmup)
    quant_lat = benchmark(small_q, args.batch_size, args.threads, args.repeats, args.warmup)
    reduction = 1.0 - quant_lat / base_lat

    print(f"BASE_PARAMS={count_parameters(base)}")
    print(f"SMALL_PARAMS={count_parameters(small)}")
    print(f"PARAM_REDUCTION={1.0 - count_parameters(small) / count_parameters(base):.6f}")
    print(f"BASE_ACC={base_acc:.6f}")
    print(f"BASE_F1={base_f1:.6f}")
    print(f"SMALL_ACC={small_acc:.6f}")
    print(f"SMALL_F1={small_f1:.6f}")
    print(f"QUANT_ACC={quant_acc:.6f}")
    print(f"QUANT_F1={quant_f1:.6f}")
    print(f"ENGINE={args.engine}")
    print(f"BATCH_SIZE={args.batch_size}")
    print(f"THREADS={args.threads}")
    print(f"BASE_LAT_MS={base_lat:.9f}")
    print(f"QUANT_LAT_MS={quant_lat:.9f}")
    print(f"LAT_REDUCTION={reduction:.6f}")


if __name__ == "__main__":
    main()
