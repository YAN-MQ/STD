"""Protocol definitions for Level 4A ns-3 in-the-loop federated runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


TaskType = Literal["client_to_plane", "plane_to_plane"]


@dataclass(frozen=True)
class RuntimeTrainingTask:
    """Describe one local training request issued by the ns-3 runtime."""

    client_id: str
    plane_id: int
    round_idx: int
    local_epochs: int
    lr: float
    weight_decay: float
    max_local_batches: int | None
    checkpoint_path: str
    output_dir: str


@dataclass(frozen=True)
class RuntimeTransferTask:
    """Describe one parameter-transfer task that must be evaluated by the runtime."""

    task_id: str
    task_type: TaskType
    src: str
    dst: str
    round_idx: int
    payload_size_bytes: int
    plane_pair: tuple[int, int] | None = None
    deadline_s: float | None = None


@dataclass(frozen=True)
class RuntimeTransferResult:
    """Describe the runtime decision for one transfer task."""

    task_id: str
    success: bool
    delay_ms: float
    bandwidth_mbps: float
    packet_loss: float
    contact_duration_s: float
    reason: str = "ok"


def dataclass_to_dict(value: object) -> dict[str, object]:
    """Serialize a protocol dataclass to a JSON-ready dictionary."""
    return asdict(value)
