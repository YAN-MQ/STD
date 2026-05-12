"""Python training bridge for Level 4A ns-3 in-the-loop orchestration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch

from .client import ClientTrainResult, FederatedClient
from .runtime_protocol import RuntimeTrainingTask


@dataclass(frozen=True)
class RuntimeTrainingResult:
    """Serialized output returned to the Level 4A runtime."""

    client_id: str
    plane_id: int
    round_idx: int
    sample_count: int
    average_loss: float
    checkpoint_path: str
    result_path: str
    state_size_bytes: int


def _state_size_bytes(state_dict: dict[str, torch.Tensor]) -> int:
    """Estimate serialized tensor size in bytes."""
    return int(sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values()))


def run_local_training_task(client: FederatedClient, task: RuntimeTrainingTask) -> RuntimeTrainingResult:
    """Execute one local training task and persist its outputs for the runtime."""
    os.makedirs(task.output_dir, exist_ok=True)
    if task.checkpoint_path:
        weights = torch.load(task.checkpoint_path, map_location="cpu")
        client.set_weights(weights)

    result: ClientTrainResult = client.local_train(
        local_epochs=task.local_epochs,
        lr=task.lr,
        weight_decay=task.weight_decay,
        max_local_batches=task.max_local_batches,
    )

    checkpoint_path = os.path.join(task.output_dir, f"{task.client_id}_round_{task.round_idx:04d}.pt")
    torch.save(result.weights, checkpoint_path)

    result_path = os.path.join(task.output_dir, f"{task.client_id}_round_{task.round_idx:04d}.json")
    payload = {
        "client_id": task.client_id,
        "plane_id": task.plane_id,
        "round_idx": task.round_idx,
        "sample_count": int(result.sample_count),
        "average_loss": float(result.average_loss),
        "checkpoint_path": checkpoint_path,
        "state_size_bytes": _state_size_bytes(result.weights),
    }
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    return RuntimeTrainingResult(
        client_id=task.client_id,
        plane_id=task.plane_id,
        round_idx=task.round_idx,
        sample_count=int(result.sample_count),
        average_loss=float(result.average_loss),
        checkpoint_path=checkpoint_path,
        result_path=result_path,
        state_size_bytes=_state_size_bytes(result.weights),
    )
