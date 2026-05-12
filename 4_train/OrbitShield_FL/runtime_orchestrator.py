"""Level 4A runtime orchestrator: ns-3-scheduled, Python-executed federated training."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from dataclasses import asdict

import torch

from .aggregators import (
    clone_state_dict,
    compute_staleness,
    estimate_link_quality,
    state_dict_num_bytes,
    weighted_average_state_dict,
    intra_plane_aggregate,
)
from .config import FederatedConfig
from .metrics_fl import evaluate_global_model
from .ns3_online_bridge import generate_online_round_trace
from .runtime_bridge import RuntimeTrainingResult, run_local_training_task
from .runtime_protocol import RuntimeTrainingTask, RuntimeTransferResult, RuntimeTransferTask, dataclass_to_dict
from .runtime_state import PlaneRuntimeState, SatelliteRuntimeState
from .serverless_orchestrator import ServerlessOrchestrator
from .topology_ns3 import convert_ns3_round_trace
from .transfer_scheduler import build_transfer_plan_from_link


class RuntimeOrchestrator(ServerlessOrchestrator):
    """Level 4A orchestrator with ns-3-owned round scheduling and communication tasks."""

    def __init__(self, config: FederatedConfig) -> None:
        super().__init__(config)
        self.runtime_output_dir = os.path.join(self.output_dir, "runtime")
        self.training_output_dir = os.path.join(self.runtime_output_dir, "local_training")
        self.online_trace_root = os.path.join(self.runtime_output_dir, "ns3_round_trace")
        os.makedirs(self.training_output_dir, exist_ok=True)
        os.makedirs(self.online_trace_root, exist_ok=True)

        self.satellite_runtime: dict[str, SatelliteRuntimeState] = {}
        for client_id, client in self.clients.items():
            self.satellite_runtime[client_id] = SatelliteRuntimeState(
                client_id=client_id,
                plane_id=client.plane_id,
                last_checkpoint_path=self.config.init_checkpoint,
            )

        self.plane_runtime: dict[int, PlaneRuntimeState] = {}
        for plane_id in range(self.config.num_planes):
            members = [cid for cid, pid in self.plane_assignments.items() if pid == plane_id]
            self.plane_runtime[plane_id] = PlaneRuntimeState(
                plane_id=plane_id,
                member_clients=members,
                current_checkpoint_path=self.config.init_checkpoint,
            )

        self.runtime_task_history: list[dict[str, object]] = []

    def _generate_round_topology(self, round_idx: int) -> dict[str, object]:
        """Generate one fresh ns-3 round and convert it to a topology snapshot."""
        round_result = generate_online_round_trace(
            binary_path=self.config.ns3_binary,
            trace_root_dir=self.online_trace_root,
            round_idx=round_idx,
            num_planes=self.config.num_planes,
            sats_per_plane=self.config.sats_per_plane,
            round_duration=self.config.ns3_round_duration,
            seed=self.config.seed,
            extra_args=[
                f"--contact-period={self.config.inter_plane_contact_period}",
                f"--contact-duration-rounds={self.config.inter_plane_contact_duration}",
                f"--intra-success-prob={self.config.intra_plane_success_prob}",
                f"--inter-success-prob={self.config.inter_plane_success_prob}",
                f"--inter-loss={self.config.packet_loss_prob}",
                f"--inter-delay={self.config.link_delay_mean}ms",
            ],
            force_regenerate=self.config.ns3_force_regenerate,
        )
        return convert_ns3_round_trace(round_result.round_trace)

    def _build_training_tasks(self, round_idx: int) -> list[RuntimeTrainingTask]:
        """Build per-client local training tasks for one runtime round."""
        tasks: list[RuntimeTrainingTask] = []
        for client_id, client in self.clients.items():
            state = self.satellite_runtime[client_id]
            checkpoint_path = state.last_checkpoint_path or self.config.init_checkpoint
            tasks.append(
                RuntimeTrainingTask(
                    client_id=client_id,
                    plane_id=client.plane_id,
                    round_idx=round_idx,
                    local_epochs=self.config.local_epochs,
                    lr=self.config.lr,
                    weight_decay=self.config.weight_decay,
                    max_local_batches=self.config.max_local_batches,
                    checkpoint_path=checkpoint_path or "",
                    output_dir=self.training_output_dir,
                )
            )
        return tasks

    def _run_training_tasks(self, tasks: list[RuntimeTrainingTask]) -> dict[str, RuntimeTrainingResult]:
        """Execute all local training tasks sequentially and collect results."""
        results: dict[str, RuntimeTrainingResult] = {}
        for task in tasks:
            result = run_local_training_task(self.clients[task.client_id], task)
            results[task.client_id] = result
            sat_state = self.satellite_runtime[task.client_id]
            sat_state.last_checkpoint_path = result.checkpoint_path
            sat_state.last_result_path = result.result_path
        return results

    def _evaluate_transfer_task(
        self,
        task: RuntimeTransferTask,
        link_state,
        bandwidth_mbps: float,
    ) -> RuntimeTransferResult:
        """Evaluate one transfer task against the current round link state."""
        plan = build_transfer_plan_from_link(
            model_size_bytes=task.payload_size_bytes,
            link_state=link_state,
            bandwidth_mbps=bandwidth_mbps,
        )
        return RuntimeTransferResult(
            task_id=task.task_id,
            success=bool(plan.can_finish),
            delay_ms=float(link_state.delay),
            bandwidth_mbps=float(plan.effective_bandwidth_mbps),
            packet_loss=float(link_state.packet_loss),
            contact_duration_s=float(link_state.contact_duration),
            reason="ok" if plan.can_finish else "window_timeout",
        )

    def _write_runtime_outputs(self) -> None:
        """Persist Level 4A runtime task history."""
        path = os.path.join(self.runtime_output_dir, "runtime_task_history.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.runtime_task_history, handle, indent=2, ensure_ascii=False)

    def train_one_federated_round(self, round_idx: int) -> dict[str, object]:
        """Execute one Level 4A runtime-controlled federated round."""
        topology = self._generate_round_topology(round_idx)
        self.topology_history.append(self._serialize_topology(topology, round_idx))

        training_tasks = self._build_training_tasks(round_idx)
        training_results = self._run_training_tasks(training_tasks)

        client_payloads_by_plane: dict[int, list[dict[str, object]]] = defaultdict(list)
        total_attempted_uploads = 0
        failed_uploads = 0
        stale_contributors = 0
        communication_bytes = 0
        previous_global = clone_state_dict(self.global_weights)

        for task in training_tasks:
            total_attempted_uploads += 1
            result = training_results[task.client_id]
            client = self.clients[task.client_id]
            intra_link = topology["intra_plane_links"][(client.plane_id, client.plane_id)]
            intra_bandwidth = float(topology.get("intra_plane_bandwidth_mbps", {}).get(client.plane_id, 0.0))

            transfer_task = RuntimeTransferTask(
                task_id=f"{task.client_id}_to_plane_{client.plane_id}_r{round_idx}",
                task_type="client_to_plane",
                src=task.client_id,
                dst=f"plane_{client.plane_id}",
                round_idx=round_idx,
                payload_size_bytes=result.state_size_bytes,
            )
            transfer_result = self._evaluate_transfer_task(transfer_task, intra_link, intra_bandwidth)

            self.runtime_task_history.append(
                {
                    "round": round_idx,
                    "training_task": dataclass_to_dict(task),
                    "training_result": dataclass_to_dict(result),
                    "transfer_task": dataclass_to_dict(transfer_task),
                    "transfer_result": dataclass_to_dict(transfer_result),
                }
            )

            if not transfer_result.success:
                failed_uploads += 1
                continue

            weights = torch.load(result.checkpoint_path, map_location="cpu")
            before_state = previous_global if client.last_sync_round == 0 else self.plane_model_cache[client.plane_id]
            update = {k: weights[k] - before_state[k] for k in weights}
            staleness = compute_staleness(client.last_sync_round, round_idx)
            stale_contributors += int(staleness > 1)
            link_quality = estimate_link_quality(
                success_rate=self.config.intra_plane_success_prob,
                contact_duration=intra_link.contact_duration,
                delay=intra_link.delay,
                packet_loss=intra_link.packet_loss,
            )
            client_payloads_by_plane[client.plane_id].append(
                {
                    "client_id": task.client_id,
                    "weights": weights,
                    "update": update,
                    "loss": result.average_loss,
                    "sample_count": result.sample_count,
                    "reputation": self.clients[task.client_id].reputation,
                    "last_sync_round": self.clients[task.client_id].last_sync_round,
                    "link_quality": link_quality,
                }
            )
            communication_bytes += int(result.state_size_bytes)

        plane_models = {}
        plane_meta = {}
        plane_sample_counts = {}
        for plane_id in range(self.config.num_planes):
            payloads = client_payloads_by_plane.get(plane_id, [])
            if payloads:
                plane_model, metadata = intra_plane_aggregate(
                    payloads,
                    current_round=round_idx,
                    lambda_s=self.config.lambda_s,
                    method="full",
                )
                plane_models[plane_id] = plane_model
                plane_meta[plane_id] = metadata
                plane_sample_counts[plane_id] = int(sum(payload["sample_count"] for payload in payloads))
                self.plane_last_sync[plane_id] = round_idx
            else:
                plane_models[plane_id] = clone_state_dict(self.plane_model_cache[plane_id])
                plane_meta[plane_id] = {"participant_count": 0, "client_weights": {}, "stale_clients": 0}
                plane_sample_counts[plane_id] = 0

        gossiped_models = {}
        plane_gossip_meta = {}
        total_inter_attempts = 0
        failed_inter_links = 0

        for plane_id in range(self.config.num_planes):
            neighbors = topology["plane_neighbors"][plane_id]
            available_models = {}
            available_scores = {}
            plane_meta_log: list[dict[str, object]] = []

            for neighbor in neighbors:
                total_inter_attempts += 1
                pair = tuple(sorted((plane_id, neighbor)))
                inter_link = topology["inter_plane_links"][pair]
                inter_bandwidth = float(topology.get("inter_plane_bandwidth_mbps", {}).get(pair, 0.0))
                payload_size = state_dict_num_bytes(plane_models[neighbor])
                transfer_task = RuntimeTransferTask(
                    task_id=f"plane_{plane_id}_to_plane_{neighbor}_r{round_idx}",
                    task_type="plane_to_plane",
                    src=f"plane_{plane_id}",
                    dst=f"plane_{neighbor}",
                    round_idx=round_idx,
                    payload_size_bytes=payload_size,
                    plane_pair=pair,
                )
                transfer_result = self._evaluate_transfer_task(transfer_task, inter_link, inter_bandwidth)
                plane_meta_log.append(
                    {
                        "transfer_task": dataclass_to_dict(transfer_task),
                        "transfer_result": dataclass_to_dict(transfer_result),
                    }
                )
                if transfer_result.success:
                    available_models[neighbor] = plane_models[neighbor]
                    available_scores[neighbor] = estimate_link_quality(
                        success_rate=self.config.inter_plane_success_prob,
                        contact_duration=inter_link.contact_duration,
                        delay=inter_link.delay,
                        packet_loss=inter_link.packet_loss,
                    )
                    communication_bytes += payload_size
                else:
                    failed_inter_links += 1

            from .gossip import inter_plane_gossip

            gossiped_model, gossip_meta = inter_plane_gossip(
                plane_id=plane_id,
                self_model=plane_models[plane_id],
                plane_neighbors=neighbors,
                available_models=available_models,
                available_scores=available_scores,
                plane_model_cache=self.plane_model_cache,
                plane_staleness={
                    other_plane: compute_staleness(self.plane_last_sync.get(other_plane, 0), round_idx)
                    for other_plane in range(self.config.num_planes)
                },
                beta=self.config.beta,
                beta_floor=self.config.beta_floor,
                lambda_s=self.config.lambda_s,
                method="full" if round_idx > self.config.warmup_rounds else "intra_only",
                rho=self.config.rho,
            )
            gossiped_models[plane_id] = gossiped_model
            gossip_meta["runtime_transfer_log"] = plane_meta_log
            plane_gossip_meta[plane_id] = gossip_meta
            self.plane_model_cache[plane_id] = clone_state_dict(gossiped_model)

        merged_weights = weighted_average_state_dict(
            list(gossiped_models.values()),
            [max(plane_sample_counts[plane_id], 1) for plane_id in range(self.config.num_planes)],
        )
        if round_idx > self.config.warmup_rounds:
            merged_weights = weighted_average_state_dict(
                [previous_global, merged_weights],
                [self.config.global_momentum, 1.0 - self.config.global_momentum],
            )
        self._set_global_weights(merged_weights)

        for client_id, client in self.clients.items():
            client.set_weights(self.plane_model_cache[client.plane_id])
            client.last_sync_round = round_idx
            client.successful_sync_count += 1
            self.satellite_runtime[client_id].model_version = round_idx
            self.satellite_runtime[client_id].last_sync_round = round_idx

        round_metrics = self._evaluate_round_metrics(
            round_idx=round_idx,
            communication_bytes=communication_bytes,
            stale_contributors=stale_contributors,
            total_attempted_uploads=total_attempted_uploads,
            failed_uploads=failed_uploads,
            failed_inter_links=failed_inter_links,
            total_inter_attempts=total_inter_attempts,
        )
        round_metrics["plane_meta"] = plane_meta
        round_metrics["plane_gossip_meta"] = plane_gossip_meta
        self.round_history.append(round_metrics)
        return round_metrics

    def run_federated_training(self) -> dict[str, object]:
        """Run full Level 4A training and persist runtime outputs."""
        result = super().run_federated_training()
        self._write_runtime_outputs()
        return result


def run_runtime_federated_training(config: FederatedConfig) -> dict[str, object]:
    """Public API for Level 4A runtime-driven training."""
    orchestrator = RuntimeOrchestrator(config)
    return orchestrator.run_federated_training()
