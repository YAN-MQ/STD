#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${TRAIN_DIR}"
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_federated_ns3_libtorch.py \
  --dataset cicids17 \
  --rounds 20 \
  --local_epochs 1 \
  --batch_size 512 \
  --device cuda \
  "$@"
