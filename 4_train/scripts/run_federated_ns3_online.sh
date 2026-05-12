#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/lithic/final/ns3-gpu-venv/bin/python}"

"${PYTHON_BIN}" scripts/train_federated_ns3_online.py \
  --dataset cicids17 \
  --method full \
  --device cuda
