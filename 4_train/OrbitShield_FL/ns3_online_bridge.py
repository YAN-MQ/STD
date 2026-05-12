"""Online ns-3 co-simulation bridge for Level 3 OrbitShield_FL training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .ns3_bridge import Ns3RoundTrace, load_ns3_round_trace, run_federated_constellation


@dataclass(frozen=True)
class Ns3OnlineRoundResult:
    """Container for one online-generated ns-3 trace round."""

    round_idx: int
    trace_dir: Path
    round_trace: Ns3RoundTrace


def generate_online_round_trace(
    *,
    binary_path: str | Path,
    trace_root_dir: str | Path,
    round_idx: int,
    num_planes: int,
    sats_per_plane: int,
    round_duration: float,
    seed: int,
    extra_args: list[str] | None = None,
    force_regenerate: bool = False,
) -> Ns3OnlineRoundResult:
    """Generate or load one ns-3 round trace for online co-simulation.

    The generated trace is isolated under ``trace_root_dir/round_xxxx`` and uses
    the global ``round_idx`` as both the exported round number and the contact
    pattern phase input.
    """

    trace_root = Path(trace_root_dir).resolve()
    trace_dir = trace_root / f"round_{round_idx:04d}"
    trace_dir.mkdir(parents=True, exist_ok=True)

    round_file = trace_dir / f"round_{round_idx:04d}.json"
    if force_regenerate or not round_file.exists():
        bundle = run_federated_constellation(
            binary_path=binary_path,
            output_dir=trace_dir,
            num_planes=num_planes,
            sats_per_plane=sats_per_plane,
            rounds=1,
            round_duration=round_duration,
            seed=seed + round_idx,
            extra_args=[f"--start-round={round_idx}", *(extra_args or [])],
        )
        round_trace = bundle.rounds[0]
    else:
        round_trace = load_ns3_round_trace(trace_dir, round_idx)

    return Ns3OnlineRoundResult(round_idx=round_idx, trace_dir=trace_dir, round_trace=round_trace)
