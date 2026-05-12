"""State containers for Level 4A ns-3 in-the-loop federated runtime."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SatelliteRuntimeState:
    """Mutable runtime state for one satellite client."""

    client_id: str
    plane_id: int
    model_version: int = 0
    reputation: float = 1.0
    last_sync_round: int = 0
    last_train_finish_time_s: float = 0.0
    last_checkpoint_path: str | None = None
    last_result_path: str | None = None


@dataclass
class PlaneRuntimeState:
    """Mutable runtime state for one orbital plane."""

    plane_id: int
    model_version: int = 0
    last_sync_round: int = 0
    current_checkpoint_path: str | None = None
    member_clients: list[str] = field(default_factory=list)
