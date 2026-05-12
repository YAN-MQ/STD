#!/usr/bin/env python3
"""Search formal post-training compression settings for CPU INT8 and GPU FP16 routes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search post-training compression settings.")
    parser.add_argument("--checkpoint", default="checkpoints_gru/cicids17_gru_best.pt")
    parser.add_argument("--data_dir", default="../dataset_cicids17")
    parser.add_argument("--route", choices=["cpu_int8", "gpu_fp16", "both"], default="both")
    parser.add_argument("--gru_amounts", default="0.25,0.30,0.35,0.40")
    parser.add_argument("--fc_amounts", default="0.10,0.15,0.20,0.25")
    parser.add_argument("--cpu_batch_sizes", default="256,512,1024")
    parser.add_argument("--gpu_batch_sizes", default="512,1024,2048")
    parser.add_argument("--cpu_threads", default="1,4,8")
    parser.add_argument("--gpu_steps", type=int, default=300)
    parser.add_argument("--cpu_steps", type=int, default=120)
    parser.add_argument("--output_dir", default="experiments/compression/search")
    parser.add_argument("--python_bin", default="/home/lithic/final/ns3-gpu-venv/bin/python")
    return parser.parse_args()


def run_and_collect(cmd: list[str], cwd: str, summary_path: Path) -> dict:
    subprocess.run(cmd, cwd=cwd, check=True)
    with open(summary_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_result(route: str, gru_amount: float, fc_amount: float, summary: dict, extra: dict) -> dict:
    compressed = summary["compressed"]
    baseline = summary["baseline"]
    return {
        "route": route,
        "gru_prune_amount": gru_amount,
        "fc_prune_amount": fc_amount,
        "accuracy": compressed["accuracy"],
        "f1": compressed["f1"],
        "parameter_reduction_ratio": compressed["parameter_reduction_ratio"],
        "latency_reduction_ratio": summary["latency_reduction_ratio"],
        "baseline_latency_ms_per_sample": baseline["latency_ms_per_sample"],
        "compressed_latency_ms_per_sample": compressed["latency_ms_per_sample"],
        "baseline_checkpoint_size_mb": baseline["checkpoint_size_mb"],
        "compressed_checkpoint_size_mb": compressed["checkpoint_size_mb"],
        **extra,
    }


def main() -> None:
    args = parse_args()
    cwd = os.path.dirname(__file__)
    project_dir = os.path.abspath(os.path.join(cwd, ".."))
    output_dir = Path(project_dir) / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    gru_amounts = parse_list(args.gru_amounts)
    fc_amounts = parse_list(args.fc_amounts)
    cpu_batch_sizes = parse_int_list(args.cpu_batch_sizes)
    gpu_batch_sizes = parse_int_list(args.gpu_batch_sizes)
    cpu_threads = parse_int_list(args.cpu_threads)

    rows: list[dict] = []

    if args.route in {"cpu_int8", "both"}:
        for gru_amount in gru_amounts:
            for fc_amount in fc_amounts:
                for batch_size in cpu_batch_sizes:
                    for threads in cpu_threads:
                        tag = f"cpu_g{int(gru_amount*100):02d}_f{int(fc_amount*100):02d}_b{batch_size}_t{threads}"
                        route_dir = output_dir / tag
                        cmd = [
                            args.python_bin,
                            "scripts/run_post_training_int8_cpu.py",
                            "--checkpoint",
                            args.checkpoint,
                            "--data_dir",
                            args.data_dir,
                            "--output_dir",
                            str(route_dir),
                            "--gru_prune_amount",
                            str(gru_amount),
                            "--fc_prune_amount",
                            str(fc_amount),
                            "--benchmark_batch_size",
                            str(batch_size),
                            "--benchmark_threads",
                            str(threads),
                            "--benchmark_steps",
                            str(args.cpu_steps),
                        ]
                        summary = run_and_collect(
                            cmd,
                            cwd=project_dir,
                            summary_path=route_dir / "cicids17_gru_post_training_int8_summary.json",
                        )
                        rows.append(
                            flatten_result(
                                "cpu_int8",
                                gru_amount,
                                fc_amount,
                                summary,
                                {"benchmark_batch_size": batch_size, "benchmark_threads": threads},
                            )
                        )

    if args.route in {"gpu_fp16", "both"}:
        for gru_amount in gru_amounts:
            for fc_amount in fc_amounts:
                for batch_size in gpu_batch_sizes:
                    tag = f"gpu_g{int(gru_amount*100):02d}_f{int(fc_amount*100):02d}_b{batch_size}"
                    route_dir = output_dir / tag
                    cmd = [
                        args.python_bin,
                        "scripts/run_post_training_fp16_gpu.py",
                        "--checkpoint",
                        args.checkpoint,
                        "--data_dir",
                        args.data_dir,
                        "--output_dir",
                        str(route_dir),
                        "--gru_prune_amount",
                        str(gru_amount),
                        "--fc_prune_amount",
                        str(fc_amount),
                        "--benchmark_batch_size",
                        str(batch_size),
                        "--benchmark_steps",
                        str(args.gpu_steps),
                        "--device",
                        "cuda",
                    ]
                    summary = run_and_collect(
                        cmd,
                        cwd=project_dir,
                        summary_path=route_dir / "cicids17_gru_post_training_fp16_summary.json",
                    )
                    rows.append(
                        flatten_result(
                            "gpu_fp16",
                            gru_amount,
                            fc_amount,
                            summary,
                            {"benchmark_batch_size": batch_size, "benchmark_threads": ""},
                        )
                    )

    rows.sort(
        key=lambda row: (
            row["accuracy"] >= 0.95 and row["parameter_reduction_ratio"] >= 0.25,
            row["latency_reduction_ratio"],
            row["accuracy"],
        ),
        reverse=True,
    )

    summary_csv = output_dir / "compression_search_summary.csv"
    fieldnames = [
        "route",
        "gru_prune_amount",
        "fc_prune_amount",
        "accuracy",
        "f1",
        "parameter_reduction_ratio",
        "latency_reduction_ratio",
        "baseline_latency_ms_per_sample",
        "compressed_latency_ms_per_sample",
        "baseline_checkpoint_size_mb",
        "compressed_checkpoint_size_mb",
        "benchmark_batch_size",
        "benchmark_threads",
    ]
    with open(summary_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        best_json = output_dir / "compression_search_best.json"
        with open(best_json, "w", encoding="utf-8") as handle:
            json.dump(rows[0], handle, indent=2, ensure_ascii=False)
        print(f"Best result saved to: {best_json}")
    print(f"Search summary saved to: {summary_csv}")


if __name__ == "__main__":
    main()
