"""Causal M-10 task-allocation environment.

This module is deliberately independent from the frozen M-09 GPPO environment.
It connects the M-10 lifecycle, service clock, received-only policy view and
task-level ACK/lease executor into one small, reproducible simulator.  The
policy sees only delivered telemetry snapshots; truth is retained by the
executor and simulator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Iterable

import numpy as np

from .service_clock import ServiceClock, ServiceEvent, ServiceResource
from .task_decision_bridge import TaskDecisionBridge
from .task_execution import TaskCommand, TaskExecution
from .task_lifecycle import TaskLifecycle, TaskState
from .task_policy_view import TaskPolicySnapshot, TaskPolicyView
from .telemetry import Telemetry
from .m10_communication import CommunicationProfile
from .m10_communication import weak_communication_profile
from .m10_contract_v1 import (
    ContractV1Config, ContractTimeLedger, EmergencyTaskAnnouncement,
    SendBudgetExecutor, canonical_audit_bytes,
)
from .completion_control_eligibility_candidate import (
    CompletionControlEligibilityConfig,
    request_completion_candidate,
    service_completion_candidate,
)


UAV_STATE_FIELDS = frozenset({"x", "y", "energy", "alive", "connected", "idle"})


@dataclass(frozen=True)
class M10Config:
    uav_count: int = 4
    task_capacity: int = 6
    region_count: int = 3
    target_count: int = 4
    event_capacity: int = 4
    relation_width: int = 4
    horizon: float = 18.0
    decision_interval: float = 1.0
    ack_latency: float = 0.0
    command_ttl: float = 0.5
    lease_ttl: float = 2.0
    telemetry_max_age: float = 2.5
    telemetry_delay: float = 0.0
    service_rate: float = 1.0
    service_power: float = 1.0
    travel_speed: float = 1.0
    travel_power: float = 0.35
    idle_power: float = 0.05
    initial_energy: float = 9.0
    reward_completion: float = 10.0
    reward_priority_scale: float = 2.0
    penalty_expired: float = 4.0
    penalty_rejected: float = 0.25
    energy_cost_weight: float = 0.08
    seed: int = 0
    task_completion_mode: str = "continuous_service_until_deadline"
    deadline_basis: str = "physical_service"
    arrival_radius: float = 0.0
    # Explicit opt-in.  None preserves the historical M-10 transport exactly.
    contract_v1: ContractV1Config | None = None

    def __post_init__(self) -> None:
        if self.uav_count <= 0 or self.task_capacity <= 0 or self.region_count <= 0 or self.target_count <= 0 or self.event_capacity <= 0:
            raise ValueError("positive fleet and task capacity required")
        if not all(math.isfinite(float(v)) for v in self.__dict__.values() if isinstance(v, (int, float))):
            raise ValueError("M10 configuration must be finite")
        if self.horizon <= 0 or self.decision_interval <= 0:
            raise ValueError("positive horizon and decision interval required")
        if self.telemetry_delay < 0:
            raise ValueError("telemetry delay must be nonnegative")
        if self.task_capacity > 32:
            raise ValueError("task capacity is intentionally bounded")
        if self.task_completion_mode not in ("continuous_service_until_deadline", "arrival_to_region"):
            raise ValueError("Unsupported task completion mode")
        if self.deadline_basis not in ("physical_service", "physical_arrival", "host_confirmation"):
            raise ValueError("Unsupported deadline basis")
        if not math.isfinite(self.arrival_radius) or self.arrival_radius < 0:
            raise ValueError("arrival_radius must be finite and nonnegative")

    @property
    def action_count(self) -> int:
        return self.uav_count * self.task_capacity + 1


@dataclass(frozen=True)
class M10TaskSpec:
    task_id: str
    arrival: float
    x: float
    y: float
    deadline: float
    service: float
    priority: float
    region_id: int = 0
    target_id: int = 0
    emergency: bool = False
    source: str = "scenario"

    def __post_init__(self) -> None:
        if self.emergency and self.source != "external_task_publisher":
            raise ValueError("emergency tasks require the external publisher source")

    def announcement(self) -> EmergencyTaskAnnouncement | None:
        if not self.emergency:
            return None
        return EmergencyTaskAnnouncement(
            task_id=self.task_id, actual_arrival=self.arrival, deadline=self.deadline,
            priority=self.priority, x=self.x, y=self.y, service=self.service,
            source=self.source,
        )


@dataclass(frozen=True)
class M10Scenario:
    name: str
    tasks: tuple[M10TaskSpec, ...]
    events: tuple[ServiceEvent, ...] = ()
    seed: int = 0
    split: str = "regression"
    tape_id: str = "regression-seed-0"
    communication: CommunicationProfile = CommunicationProfile()


def default_scenario(name: str = "mixed", seed: int = 0, *, split: str = "regression") -> M10Scenario:
    """Return a reproducible scenario with all four required perturbation classes.

    The event schedule is public to the simulator but is not injected into the
    policy view.  A separate tape generator can choose which events are sent as
    received telemetry after they occur.
    """
    rng = np.random.default_rng(seed)
    positions = ((1.5, 0.5), (2.5, 2.0), (0.5, 2.5), (3.0, 0.2), (1.0, 3.0), (3.2, 2.8))
    tasks = tuple(
        M10TaskSpec(
            task_id=f"task-{i}",
            arrival=float((0.0, 0.0, 2.0, 3.0, 5.0, 7.0)[i]),
            x=positions[i][0],
            y=positions[i][1],
            deadline=float((7.0, 8.0, 11.0, 12.0, 15.0, 17.0)[i]),
            service=float((1.3, 1.8, 1.0, 2.0, 1.2, 1.6)[i]),
            priority=float((1.0, 1.6, 1.2, 2.0, 1.1, 1.8)[i]),
            region_id=i % 3,
            target_id=i % 4,
        )
        for i in range(6)
    )
    events = (
        ServiceEvent(4.0, "uav-1", "disconnect"),
        ServiceEvent(6.0, "uav-1", "reconnect"),
        ServiceEvent(8.0, "uav-2", "damage"),
    )
    if seed != 0 and split != "regression":
        # Seed-addressed tapes alter both task parameters and the fault tape.
        # The seed-0 regression scenario above remains byte-for-byte stable.
        jittered = []
        for task in tasks:
            arrival = task.arrival if task.arrival == 0 else max(0.0, task.arrival + float(rng.uniform(-0.45, 0.45)))
            deadline = max(arrival + 1.0, task.deadline + float(rng.uniform(-0.65, 0.65)))
            jittered.append(M10TaskSpec(
                task_id=task.task_id,
                arrival=arrival,
                x=max(0.05, task.x + float(rng.normal(0.0, 0.18))),
                y=max(0.05, task.y + float(rng.normal(0.0, 0.18))),
                deadline=deadline,
                service=max(0.35, task.service * float(rng.uniform(0.8, 1.25))),
                priority=max(0.1, task.priority * float(rng.uniform(0.85, 1.15))),
                region_id=int(rng.integers(0, 3)),
                target_id=int(rng.integers(0, 4)),
            ))
        tasks = tuple(jittered)

    if name == "normal":
        events = ()
    elif name == "energy_insufficient":
        tasks = tuple(M10TaskSpec(**{**task.__dict__, "service": task.service * 2.5}) for task in tasks)
    elif name == "uav_damage":
        event_time = 3.0 if seed == 0 else float(rng.uniform(2.0, 5.0))
        events = (ServiceEvent(event_time, f"uav-{0 if seed == 0 else int(rng.integers(0, 4))}", "damage"),)
    elif name == "communication_interrupt":
        uid = "uav-1" if seed == 0 else f"uav-{int(rng.integers(0, 4))}"
        disconnect = 3.0 if seed == 0 else float(rng.uniform(2.0, 5.0))
        events = (ServiceEvent(disconnect, uid, "disconnect"), ServiceEvent(disconnect + 4.0, uid, "reconnect"))
    elif name == "composite":
        pass
    elif name != "mixed":
        raise ValueError(f"unknown M10 scenario: {name}")
    if name == "mixed" and seed != 0 and split != "regression":
        disconnect = float(rng.uniform(2.5, 5.0))
        reconnect = min(10.0, disconnect + float(rng.uniform(2.0, 4.0)))
        link_resource = f"uav-{int(rng.integers(0, 4))}"
        events = (
            ServiceEvent(disconnect, link_resource, "disconnect"),
            ServiceEvent(reconnect, link_resource, "reconnect"),
            ServiceEvent(float(rng.uniform(6.5, 9.5)), f"uav-{int(rng.integers(0, 4))}", "damage"),
        )
    tape_id = f"{split}-{name}-seed-{seed}"
    return M10Scenario(name=name, tasks=tasks, events=events, seed=seed, split=split, tape_id=tape_id)


_TAPE_OFFSETS = {"train": 0, "validation": 1_000_000, "test": 2_000_000, "ood": 3_000_000}


def scenario_tape(split: str, *, count: int, base_seed: int = 7001, name: str = "mixed") -> tuple[M10Scenario, ...]:
    """Freeze a distinct, serializable scenario collection for one split."""
    if split not in _TAPE_OFFSETS:
        raise ValueError(f"unknown tape split: {split}")
    return tuple(default_scenario(name, seed=base_seed + _TAPE_OFFSETS[split] + i, split=split) for i in range(count))


def weak_communication_tape(split: str, *, count: int, base_seed: int = 7001,
                           level: str = "composite", name: str = "mixed") -> tuple[M10Scenario, ...]:
    """Return a distinct, serializable tape with one frozen link protocol."""
    profile = weak_communication_profile(level)
    return tuple(replace(scenario, communication=profile)
                 for scenario in scenario_tape(split, count=count, base_seed=base_seed, name=name))


def scenario_to_dict(scenario: M10Scenario) -> dict[str, Any]:
    return {
        "name": scenario.name,
        "seed": scenario.seed,
        "split": scenario.split,
        "tape_id": scenario.tape_id,
        "tasks": [asdict(task) for task in scenario.tasks],
        "events": [asdict(event) for event in scenario.events],
        "communication": scenario.communication.to_dict(),
    }


def scenario_from_dict(payload: dict[str, Any]) -> M10Scenario:
    return M10Scenario(
        name=str(payload["name"]),
        tasks=tuple(M10TaskSpec(**item) for item in payload["tasks"]),
        events=tuple(ServiceEvent(**item) for item in payload.get("events", [])),
        seed=int(payload.get("seed", 0)), split=str(payload.get("split", "regression")),
        tape_id=str(payload.get("tape_id", "unknown")),
        communication=CommunicationProfile.from_dict(payload.get("communication")),
    )


def m10_config_to_dict(config: M10Config) -> dict[str, Any]:
    """Serialize the versioned environment contract without losing nesting."""
    return asdict(config)


def m10_config_from_dict(payload: dict[str, Any]) -> M10Config:
    values = dict(payload)
    if isinstance(values.get("contract_v1"), dict):
        values["contract_v1"] = ContractV1Config.from_dict(values["contract_v1"])
    return M10Config(**values)


class M10Environment:
    """Gym-like environment with a fixed global assignment action space."""

    def __init__(self, config: M10Config | None = None, scenario: M10Scenario | None = None, *, exogenous_key: str | None = None):
        self.config = config or M10Config()
        self.scenario = scenario or default_scenario(seed=self.config.seed)
        self.communication = self.scenario.communication
        # Counterfactual branches may opt into a shared random stream.  The
        # key is only used by the simulator-side label generator; it is never
        # exposed through the public observation.
        self._exogenous_key = exogenous_key
        if len(self.scenario.tasks) > self.config.task_capacity:
            raise ValueError("scenario exceeds public task capacity")
        self.uav_ids = tuple(f"uav-{i}" for i in range(self.config.uav_count))
        self._sequence: dict[tuple[str, str], int] = {}
        self._task_by_id = {task.task_id: task for task in self.scenario.tasks}
        self._last_visible_states: dict[str, str] = {}
        self._event_cursor = 0
        self._reset_state()

    def _random_identity(self, identity: str) -> str:
        if self._exogenous_key is None:
            return identity
        return f"{self._exogenous_key}|{identity.split('|')[0]}"

    def _reset_state(self) -> None:
        tasks = {
            spec.task_id: TaskLifecycle(
                task_id=spec.task_id,
                arrival=spec.arrival,
                deadline=spec.deadline,
                required_service=spec.service,
                priority=spec.priority,
                completion_mode=self.config.task_completion_mode,
                completion_radius=self.config.arrival_radius,
            )
            for spec in self.scenario.tasks
        }
        resources = {
            uid: ServiceResource(
                energy=self.config.initial_energy,
                service_rate=self.config.service_rate,
                service_power=self.config.service_power,
                position=(0.0, 0.0),
                speed=self.config.travel_speed,
                travel_power=self.config.travel_power,
                idle_power=self.config.idle_power,
            )
            for uid in self.uav_ids
        }
        positions = {spec.task_id: (spec.x, spec.y) for spec in self.scenario.tasks}
        self.clock = ServiceClock(
            tasks,
            resources,
            list(self.scenario.events),
            disconnect_interrupts=True,
            task_positions=positions,
        )
        self.execution = TaskExecution(self.clock, command_ttl=self.config.command_ttl, lease_ttl=self.config.lease_ttl)
        self.view = TaskPolicyView(self.uav_ids, task_capacity=self.config.task_capacity, max_age=self.config.telemetry_max_age)
        self.bridge = TaskDecisionBridge(self.view, self.execution)
        self._sequence = {}
        self._last_visible_states = {}
        self._event_cursor = 0
        self._command_index = 0
        self._step_index = 0
        self._episode_id = f"m10-{self.scenario.name}-{self.config.seed}"
        self._last_energy = sum(resource.energy for resource in self.clock.resources.values())
        self._last_completed = 0
        self._last_expired = 0
        self._feedback_log: list[dict[str, Any]] = []
        self._delivered_messages: list[dict[str, Any]] = []
        self._communication_log: list[dict[str, Any]] = []
        # (semantic kind, telemetry, delivery ordinal).  A duplicated packet
        # keeps the same message_id but has a distinct delivery ordinal.
        self._pending_messages: list[tuple[str, Telemetry, int]] = []
        # (delivery time, renewal id, command id, delivery ordinal, send time). Renewal
        # delivery is ordered by its scheduled transport time, not by the
        # order in which requests were created.
        self._pending_renewals: list[tuple[float, str, str, int, float]] = []
        self._lease_renewal_results: dict[str, str] = {}
        self._public_event_records: list[dict[str, Any]] = []
        self._last_public_event_values: dict[tuple[str, str], float] = {}
        self._public_task_entities: set[str] = set()
        self._trigger_flags: dict[str, bool] = {
            "task_arrival": False,
            "confirmed_fault": False,
            "link_recovery": False,
            "completion_or_invalidation": False,
            "safety_forced": False,
        }
        # Control-side continuation knowledge is a set.  TaskExecution owns
        # one fenced lease per command; retaining only one handle drops the
        # other acknowledged UAV/task executions on the next step.
        self._active_commands: dict[str, TaskCommand] = {}
        self._active_actions: dict[str, int] = {}
        self._completion_records: dict[str, dict[str, Any]] = {}
        self._completion_notice_ids: set[str] = set()
        self._emitted_arrival_events: set[tuple[str, float]] = set()
        self._contract_budget = (SendBudgetExecutor(self.config.contract_v1.budget)
                                 if self.config.contract_v1 is not None else None)
        self._completion_eligibility_config = CompletionControlEligibilityConfig(
            enable_completion_control_eligibility=bool(
                self.config.contract_v1 is not None
                and self.config.contract_v1.enable_completion_control_eligibility
            )
        )
        self._contract_pending_sends: list[dict[str, Any]] = []
        self._contract_ledger: dict[str, ContractTimeLedger] = {
            spec.task_id: ContractTimeLedger(task_id=spec.task_id, actual_arrival=float(spec.arrival))
            for spec in self.scenario.tasks
        }
        self._contract_event_log: list[dict[str, Any]] = []
        self._contract_last_reported: dict[tuple[str, str, str], tuple[float, ...]] = {}
        self._contract_last_uav_heartbeat = -math.inf
        self._contract_last_task_telemetry_heartbeat = -math.inf
        self._contract_published_emergencies: set[str] = set()
        self._contract_delivered_emergencies: set[str] = set()
        self._contract_completed_events: set[str] = set()
        self._contract_open_interruptions: dict[str, str] = {}
        self._deliver_observations()
        self._flush_messages()

    @property
    def _active_command(self) -> TaskCommand | None:
        """Compatibility view for older diagnostics; not a control path."""
        return next(iter(self._active_commands.values()), None)

    @property
    def _active_action(self) -> int | None:
        """Compatibility view for older diagnostics; not a control path."""
        return next(iter(self._active_actions.values()), None)

    def _remember_active(self, command: TaskCommand, action: int) -> None:
        self._active_commands[command.command_id] = command
        self._active_actions[command.command_id] = int(action)

    def _forget_active(self, command_id: str) -> None:
        self._active_commands.pop(command_id, None)
        self._active_actions.pop(command_id, None)

    def _apply_renewal(self, renewal_id: str, command_id: str, ordinal: int) -> str:
        command = self._active_commands.get(command_id)
        if command is None:
            return "unknown_control_handle"
        execution_result = self.execution.renew(
            command.command_id, command.uav_id, command.token,
        )
        ack_identity = f"{renewal_id}|ack|{ordinal}"
        ack_budget_ok = self._contract_control_send_allowed(
            self.clock.time, "uplink", "ack", ack_identity)
        ack_delivered = bool(ack_budget_ok and self.communication.ack_delivered(
            seed=self.scenario.seed, identity=self._random_identity(ack_identity),
        ))
        self._communication_log.append({
            "link": "ack", "kind": "lease_renewal",
            "status": "received" if ack_delivered else ("budget_rejected" if not ack_budget_ok else "dropped"),
            "command_id": command_id, "renewal_id": renewal_id,
            "delivery_ordinal": ordinal, "result": execution_result,
            "time": self.clock.time,
        })
        if not ack_delivered:
            result = "ack_lost"
        else:
            result = str(execution_result)
            if execution_result != "renewed":
                # Only a response delivered to the controller can retire its
                # known handle; executor truth is never polled directly.
                self._forget_active(command_id)
        self._lease_renewal_results[command_id] = result
        return result

    def _deliver_pending_renewals(self, now: float) -> None:
        ready = [item for item in self._pending_renewals if item[0] <= now]
        self._pending_renewals = [item for item in self._pending_renewals if item[0] > now]
        for delivery_time, renewal_id, command_id, ordinal, sent_time in sorted(ready):
            self._communication_log.append({
                "link": "command", "kind": "lease_renewal",
                "status": "received" if ordinal == 0 else "duplicate_received",
                "command_id": command_id, "renewal_id": renewal_id,
                "delivery_ordinal": ordinal, "sent_time": sent_time,
                "time": now,
            })
            self._apply_renewal(renewal_id, command_id, ordinal)

    def _advance_execution(self, end: float) -> None:
        while self._pending_renewals:
            delivery_time = min(item[0] for item in self._pending_renewals)
            if delivery_time > end:
                break
            self.execution.advance(delivery_time)
            self._deliver_pending_renewals(delivery_time)
            self._emit_completion_notices()
        self.execution.advance(end)
        self._emit_completion_notices()

    def _emit_completion_notices(self) -> None:
        """Emit one public completion notice for each physical arrival."""
        if self.config.task_completion_mode != "arrival_to_region":
            if self._contract_enabled() and self.config.contract_v1.reporting_mode == "event":
                for task_id, task in self.clock.tasks.items():
                    if task.state == TaskState.COMPLETED and task_id not in self._contract_completed_events:
                        self._contract_completed_events.add(task_id)
                        self._send_contract_event("task_physical_completion", task_id, self.clock.time,
                                                  {"source": "uav_local_observer"})
                        self._send("task", task_id, "pending", 0.0, message_kind="event")
            return
        for event in self.clock.log:
            if event.get("kind") != "arrival":
                continue
            key = (str(event["task"]), float(event["time"]))
            if key in self._emitted_arrival_events:
                continue
            self._emitted_arrival_events.add(key)
            task_id = str(event["task"])
            task = self.clock.tasks[task_id]
            record = {
                "task_id": task_id,
                "uav_id": str(event["resource"]),
                "physical_arrival_time": float(event["time"]),
                "deadline": float(task.deadline),
                "completion_message_id": None,
                "completion_message_send_time": None,
                "host_confirmation_time": None,
                "physical_arrival_before_deadline": bool(float(event["time"]) <= float(task.deadline)),
                "host_confirmation_before_deadline": None,
            }
            self._completion_records[task_id] = record
            identity = self._send("task", task_id, "pending", 0.0, message_kind="completion", completion_task_id=task_id)
            if self._contract_enabled() and self.config.contract_v1.reporting_mode == "event":
                self._send_contract_event("task_physical_completion", task_id, float(event["time"]),
                                          {"source": "uav_local_observer"})
            record["completion_message_id"] = identity
            record["completion_message_send_time"] = float(event["time"])
            if identity is not None:
                self._completion_notice_ids.add(identity)

    def _renew_active_leases(self, *, skip: set[str] | None = None) -> dict[str, str]:
        """Schedule one renewal per ACK-known lease through the command link."""
        skipped = skip or set()
        results: dict[str, str] = {}
        for command_id in sorted(tuple(self._active_commands)):
            if command_id in skipped:
                continue
            command = self._active_commands.get(command_id)
            if command is None:
                continue
            renewal_id = f"{command_id}|renew|{self._step_index + 1:05d}"
            if not self._contract_control_send_allowed(
                    self.clock.time, "downlink", "lease_renewal", renewal_id):
                results[command_id] = "budget_rejected"
                continue
            fate = self.communication.renewal(
                seed=self.scenario.seed, identity=self._random_identity(renewal_id),
            )
            self._communication_log.append({
                "link": "command", "kind": "lease_renewal",
                "status": "dropped" if fate["dropped"] else "sent",
                "command_id": command_id, "renewal_id": renewal_id,
                "time": self.clock.time, "delay": fate["delay"],
            })
            if fate["dropped"]:
                results[command_id] = "command_lost"
                continue
            ordinals = (0, 1) if fate["duplicate"] else (0,)
            for ordinal in ordinals:
                self._pending_renewals.append((
                    self.clock.time + float(fate["delay"]),
                    renewal_id, command_id, ordinal, self.clock.time,
                ))
            if fate["delay"] <= 0:
                self._deliver_pending_renewals(self.clock.time)
                results[command_id] = self._lease_renewal_results.get(command_id, "queued")
            else:
                results[command_id] = "queued"
        return results

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        if seed is not None and seed != self.config.seed:
            self.config = M10Config(**{**self.config.__dict__, "seed": seed})
            default = default_scenario(self.scenario.name, seed=seed)
            self.scenario = M10Scenario(
                name=default.name, tasks=default.tasks, events=default.events,
                seed=default.seed, split=default.split, tape_id=default.tape_id,
                communication=self.scenario.communication,
            )
            self.communication = self.scenario.communication
        self._reset_state()
        return self._observation(clear_trigger=True)

    def _next_sequence(self, entity: str, field: str) -> int:
        key = (entity, field)
        self._sequence[key] = self._sequence.get(key, 0) + 1
        return self._sequence[key]

    def _contract_enabled(self) -> bool:
        return self.config.contract_v1 is not None

    def _contract_message_kind_priority(self, kind: str) -> int:
        if self.config.contract_v1 is None:
            return 99
        priorities = dict(self.config.contract_v1.budget.priority)
        return int(priorities.get(kind, priorities.get("telemetry", 99)))

    def _contract_should_report(self, kind: str, entity: str, field: str,
                                value: float, now: float, *, force: bool = False) -> bool:
        """Return whether an event-mode report is generated from local state."""
        if not self._contract_enabled() or self.config.contract_v1.reporting_mode == "periodic":
            return True
        key = (kind, entity, field)
        previous = self._contract_last_reported.get(key)
        changed = previous is None or previous != (float(value),)
        if kind == "uav":
            heartbeat_interval = self.config.contract_v1.effective_uav_heartbeat_interval
            last_heartbeat = self._contract_last_uav_heartbeat
        else:
            heartbeat_interval = self.config.contract_v1.effective_task_telemetry_heartbeat_interval
            last_heartbeat = self._contract_last_task_telemetry_heartbeat
        heartbeat = now - last_heartbeat >= heartbeat_interval
        return bool(force or changed or heartbeat)

    def _contract_record_budget(self, decision: Any, *, identity: str) -> None:
        self._communication_log.append({
            "link": "budget", "status": decision.status, "direction": decision.direction,
            "message_kind": decision.message_kind, "identity": identity,
            "time": decision.time, "token_before": decision.token_before,
            "token_after": decision.token_after, "queue_depth": decision.queue_depth,
            "reason": decision.reason,
        })

    def _contract_control_send_allowed(self, now: float, direction: str,
                                       message_kind: str, identity: str) -> bool:
        """Reserve command/ACK/renewal sends without unsafe delayed execution.

        A command is bound to the public snapshot in which it was issued, so a
        no-token command is rejected rather than executed later with stale
        state.  The rejection remains distinct from a post-send link drop.
        """
        if not self._contract_enabled():
            return True
        decision = self._contract_budget.reserve_control(now, direction, message_kind)
        self._contract_record_budget(decision, identity=identity)
        if decision.status != "sent":
            self._communication_log.append({
                "link": direction, "status": "budget_rejected",
                "message_kind": message_kind, "identity": identity,
                "time": now, "reason": "no_token_control_message",
            })
            return False
        return True

    def _transmit_contract_telemetry(self, kind: str, message: Telemetry,
                                     message_kind: str, completion_task_id: str | None = None) -> str:
        """Apply the weak link after a budget token has already been reserved."""
        now = self.clock.time
        identity = message.message_id
        impairment = self.communication.telemetry(
            seed=self.scenario.seed, identity=self._random_identity(identity), now=now,
        )
        audit_payload = {
            "kind": kind, "entity": message.entity, "field": message.field,
            "value": message.value, "measured_at": message.measured_at,
            "message_id": identity, "message_kind": message_kind,
        }
        audit_bytes = canonical_audit_bytes(audit_payload)
        if impairment["dropped"]:
            self._communication_log.append({
                "link": "telemetry", "message_kind": message_kind,
                "status": "dropped_after_send", "identity": identity,
                "message_id": identity, "delivery_ordinal": 0, "time": now,
                "reason": "outage" if impairment["outage"] else "random_loss",
                "canonical_audit_bytes": audit_bytes,
            })
            return identity
        received_at = now + self.config.telemetry_delay + self.communication.telemetry_extra_delay + impairment["jitter"]
        message = Telemetry(message.entity, message.field, message.value, message.measured_at,
                            received_at, message.sequence, message.message_id)
        self._communication_log.append({
            "link": "telemetry", "message_kind": message_kind, "status": "sent",
            "identity": identity, "message_id": identity, "delivery_ordinal": 0,
            "time": now, "received_at": received_at,
            "canonical_audit_bytes": audit_bytes,
        })
        self._contract_last_reported[(kind, message.entity, message.field)] = (float(message.value),)
        if message.received_at > now:
            self._pending_messages.append((kind, message, 0))
            if impairment["duplicate"]:
                self._pending_messages.append((kind, message, 1))
        else:
            self._accept_message(kind, message, now)
        return identity

    def _expire_contract_pending_sends(self, now: float) -> None:
        """Remove queued messages whose measured value is no longer usable."""
        if not self._contract_enabled() or not self._contract_pending_sends:
            return
        ttl = self.config.contract_v1.budget.message_ttl
        remaining: list[dict[str, Any]] = []
        for item in self._contract_pending_sends:
            message = item["message"]
            if now - message.measured_at > ttl:
                self._contract_budget.dequeue("uplink")
                self._communication_log.append({
                    "link": "budget", "status": "expired_in_queue",
                    "identity": message.message_id, "message_kind": item["message_kind"],
                    "time": now, "measured_at": message.measured_at,
                    "reason": "message_ttl",
                })
            else:
                remaining.append(item)
        self._contract_pending_sends = remaining

    def _priority_admit_contract_pending(self, *, kind: str, message: Telemetry,
                                         message_kind: str, now: float) -> bool:
        """Apply the opt-in admission policy before asking the token bucket.

        Returns True when the new value was coalesced into an existing unsent
        ordinary telemetry item.  Sent messages are never withdrawn.
        """
        contract = self.config.contract_v1
        if contract is None or contract.queue_admission_policy != "priority_replace_merge":
            return False
        self._expire_contract_pending_sends(now)
        coalesce_uav_state = bool(
            contract.enable_uav_state_coalescing and
            kind == "uav" and
            message.field in UAV_STATE_FIELDS and
            message_kind == "event"
        )
        if message_kind == "telemetry" or coalesce_uav_state:
            for index, item in enumerate(self._contract_pending_sends):
                old = item["message"]
                if coalesce_uav_state:
                    same_slot = (
                        item["kind"] == kind and
                        item.get("direction", "uplink") == "uplink" and
                        item["message_kind"] == message_kind and
                        old.entity == message.entity and
                        old.field == message.field
                    )
                else:
                    # Keep the pre-existing telemetry merge predicate intact.
                    same_slot = (
                        item["kind"] == kind and
                        item.get("direction", "uplink") == "uplink" and
                        item["message_kind"] == "telemetry" and
                        old.entity == message.entity and
                        old.field == message.field
                    )
                if same_slot:
                    if coalesce_uav_state:
                        # A UAV state slot keeps its original queue age.  A
                        # stale or duplicate measurement cannot replace a
                        # newer payload, even when a caller constructs
                        # messages by hand.
                        if (message.sequence <= old.sequence or
                                message.measured_at < old.measured_at):
                            self._communication_log.append({
                                "link": "budget", "direction": "uplink",
                                "status": "coalesced_stale_ignored",
                                "identity": message.message_id,
                                "replaced_identity": old.message_id,
                                "message_kind": message_kind,
                                "time": now,
                                "old_measured_at": old.measured_at,
                                "new_measured_at": message.measured_at,
                                "original_enqueued_at": item["enqueued_at"],
                                "queue_depth": self._contract_budget.queue_depth["uplink"],
                                "reason": "older_or_duplicate_measurement",
                            })
                            return True
                        self._contract_pending_sends[index] = {
                            **item, "message": message,
                        }
                    else:
                        # Preserve the existing task telemetry contract.
                        self._contract_pending_sends[index] = {
                            **item, "message": message, "enqueued_at": now,
                        }
                    self._communication_log.append({
                        "link": "budget", "direction": "uplink",
                        "status": "coalesced_unsent",
                        "identity": message.message_id, "replaced_identity": old.message_id,
                        "message_kind": message_kind, "time": now,
                        "old_measured_at": old.measured_at,
                        "new_measured_at": message.measured_at,
                        "original_enqueued_at": item["enqueued_at"],
                        "queue_depth": self._contract_budget.queue_depth["uplink"],
                        "reason": ("latest_unsent_uav_state_same_source_entity_field"
                                   if coalesce_uav_state else
                                   "latest_unsent_same_source_entity_field"),
                    })
                    return True
        incoming_priority = self._contract_message_kind_priority(message_kind)
        # Give already queued higher/equal-priority messages the first chance
        # to consume newly available tokens.  Admission replacement alone is
        # insufficient when a fresh stream of messages drains the bucket
        # before the pending queue is serviced.
        if any(self._contract_message_kind_priority(item["message_kind"]) <= incoming_priority
               for item in self._contract_pending_sends):
            self._flush_contract_pending_sends()
        self._contract_budget.advance(now)
        limit = self.config.contract_v1.budget.uplink_queue_capacity
        # The incoming message is an ordinary (non-control) send.  A raw
        # token count of one is not sufficient when the same bucket has a
        # control reserve: the ordinary send must leave that reserve intact.
        # Without this check, an emergency announcement saw ``tokens ==
        # control_reserve`` as available, skipped replacement, and then was
        # rejected by ``reserve``/``enqueue`` as queue_full.
        ordinary_floor = float(self.config.contract_v1.budget.control_reserve_uplink)
        ordinary_send_available = (
            self._contract_budget.tokens["uplink"] >= 1.0 - 1e-12 and
            self._contract_budget.tokens["uplink"] - 1.0 >= ordinary_floor - 1e-12
        )
        if ordinary_send_available:
            return False
        if self._contract_budget.queue_depth["uplink"] < limit:
            return False
        candidates = [
            (self._contract_message_kind_priority(item["message_kind"]), item["enqueued_at"],
             item.get("service_sort_key", item["message"].message_id), index, item)
            for index, item in enumerate(self._contract_pending_sends)
            if self._contract_message_kind_priority(item["message_kind"]) > incoming_priority
        ]
        if not candidates:
            return False
        old_priority, _, old_identity, index, old_item = max(candidates)
        self._contract_pending_sends.pop(index)
        self._contract_budget.dequeue("uplink")
        self._communication_log.append({
            "link": "budget", "status": "replaced_in_queue",
            "identity": message.message_id, "replaced_identity": old_identity,
            "message_kind": message_kind, "replaced_message_kind": old_item["message_kind"],
            "priority": incoming_priority, "replaced_priority": old_priority,
            "time": now, "reason": "higher_priority_admission",
        })
        return False

    def _flush_contract_pending_sends(self) -> None:
        if not self._contract_enabled() or not self._contract_pending_sends:
            return
        now = self.clock.time
        self._expire_contract_pending_sends(now)
        ordered = sorted(self._contract_pending_sends,
                         key=lambda item: (self._contract_message_kind_priority(item["message_kind"]),
                                            item["enqueued_at"], item["message"].message_id))
        remaining: list[dict[str, Any]] = []
        ttl = self.config.contract_v1.budget.message_ttl
        for item in ordered:
            message = item["message"]
            decision = service_completion_candidate(
                self._contract_budget, now, "uplink", item["message_kind"],
                self._completion_eligibility_config,
            )
            if decision.status != "sent":
                remaining.append(item)
                continue
            self._contract_budget.dequeue("uplink")
            self._contract_record_budget(decision, identity=message.message_id)
            self._transmit_contract_telemetry(item["kind"], message, item["message_kind"],
                                               item.get("completion_task_id"))
        self._contract_pending_sends = remaining

    def _send_contract_event(self, event_type: str, entity: str, now: float,
                             payload: dict[str, Any]) -> None:
        """Record a separately typed event; its transport still uses the same budget."""
        if not self._contract_enabled() or not self.config.contract_v1.enable_event_reports:
            return
        record = {
            "event_type": event_type, "entity": entity, "actual_time": now,
            "payload": dict(payload), "source": payload.get("source", "local_observer"),
        }
        self._contract_event_log.append(record)

    def _send(self, kind: str, entity: str, field: str, value: float, *,
              message_kind: str = "telemetry", completion_task_id: str | None = None) -> str | None:
        now = self.clock.time
        sequence = self._next_sequence(entity, field)
        identity = f"{kind}|{entity}|{field}|{sequence}|{now:.9f}"
        if completion_task_id is not None:
            self._completion_notice_ids.add(identity)
        message = Telemetry(entity, field, float(value), now, now, sequence, identity)
        if not self._contract_enabled():
            impairment = self.communication.telemetry(seed=self.scenario.seed, identity=self._random_identity(identity), now=now)
            if impairment["dropped"]:
                self._communication_log.append({"link": "telemetry", "status": "dropped", "identity": identity,
                                                "message_id": identity, "delivery_ordinal": 0,
                                                "time": now, "reason": "outage" if impairment["outage"] else "random_loss"})
                return None
            received_at = now + self.config.telemetry_delay + self.communication.telemetry_extra_delay + impairment["jitter"]
            message = Telemetry(entity, field, float(value), now, received_at, sequence, identity)
            self._communication_log.append({"link": "telemetry", "message_kind": message_kind, "status": "sent", "identity": identity,
                                            "message_id": message.message_id, "delivery_ordinal": 0,
                                            "time": now, "received_at": received_at})
            if message.received_at > now:
                self._pending_messages.append((kind, message, 0))
                if impairment["duplicate"]:
                    self._pending_messages.append((kind, message, 1))
                return identity
            self._accept_message(kind, message, now)
            return identity

        if self._priority_admit_contract_pending(kind=kind, message=message,
                                                  message_kind=message_kind, now=now):
            return identity
        decision = request_completion_candidate(
            self._contract_budget, now, "uplink", message_kind,
            self._completion_eligibility_config,
        )
        self._contract_record_budget(decision, identity=identity)
        if decision.status == "budget_rejected":
            self._communication_log.append({"link": "telemetry", "status": "budget_rejected",
                                            "message_kind": message_kind, "identity": identity,
                                            "time": now, "reason": decision.reason})
            return None
        if decision.status == "queued":
            self._contract_pending_sends.append({
                "kind": kind, "message": message, "message_kind": message_kind,
                "completion_task_id": completion_task_id, "enqueued_at": now,
                "direction": "uplink",
                "service_sort_key": message.message_id,
            })
            self._communication_log.append({"link": "telemetry", "status": "queued",
                                            "message_kind": message_kind, "identity": identity,
                                            "time": now, "reason": decision.reason})
            return identity
        return self._transmit_contract_telemetry(kind, message, message_kind, completion_task_id)

    def _accept_message(self, kind: str, message: Telemetry, now: float, *, delivery_ordinal: int = 0) -> bool:
        accepted = self.view.receive(kind, message, now)
        if not accepted:
            self._communication_log.append({"link": "telemetry", "status": "stale_or_duplicate",
                                            "entity": message.entity, "field": message.field,
                                            "sequence": message.sequence, "message_id": message.message_id,
                                            "delivery_ordinal": delivery_ordinal, "time": now})
            return False
        self._communication_log.append({"link": "telemetry", "status": "received",
                                        "entity": message.entity, "field": message.field,
                                        "sequence": message.sequence, "message_id": message.message_id,
                                        "delivery_ordinal": delivery_ordinal, "time": now,
                                        "measured_at": message.measured_at,
                                        "received_at": message.received_at})
        if message.message_id in self._completion_notice_ids:
            record = self._completion_records.get(str(message.entity))
            if record is not None and record["host_confirmation_time"] is None:
                record["host_confirmation_time"] = float(now)
                record["host_confirmation_before_deadline"] = bool(float(now) <= float(record["deadline"]))
                ledger = self._contract_ledger.get(str(message.entity))
                if ledger is not None:
                    ledger.mark("host_confirmation", float(now))
        self._delivered_messages.append({"kind": kind, "entity": message.entity, "field": message.field,
                                         "message_id": message.message_id,
                                         "delivery_ordinal": delivery_ordinal,
                                         "time": now, "measured_at": message.measured_at,
                                         "received_at": message.received_at})
        if kind == "task" and message.entity not in self._public_task_entities:
            self._public_task_entities.add(message.entity)
            self._trigger_flags["task_arrival"] = True
            ledger = self._contract_ledger.get(message.entity)
            if ledger is not None and ledger.first_legal_awareness is None:
                ledger.mark("first_legal_awareness", float(now))
            spec = self._task_by_id.get(message.entity)
            if spec is not None and spec.emergency:
                self._contract_delivered_emergencies.add(message.entity)
                self._send_contract_event("emergency_task_awareness", message.entity, float(now),
                                          {"source": spec.source})
        if kind == "uav" and message.field in ("alive", "connected"):
            key = (message.entity, message.field)
            previous = self._last_public_event_values.get(key)
            current = float(message.value)
            self._last_public_event_values[key] = current
            if previous is not None and previous != current:
                if current < previous:
                    self._trigger_flags["confirmed_fault"] = True
                elif current > previous and message.field == "connected":
                    self._trigger_flags["link_recovery"] = True
                self._public_event_records.append({
                    "entity": message.entity,
                    "field": message.field,
                    "value": current,
                    "measured_at": message.measured_at,
                    "received_at": now,
                })
        if kind == "task" and message.field == "pending":
            key = (message.entity, message.field)
            previous = self._last_public_event_values.get(key)
            current = float(message.value)
            self._last_public_event_values[key] = current
            task = self.clock.tasks.get(message.entity)
            if previous is not None and previous != current and task is not None and task.state in (TaskState.COMPLETED, TaskState.EXPIRED):
                self._trigger_flags["completion_or_invalidation"] = True
        return True

    def _flush_messages(self) -> None:
        now = self.clock.time
        ready = [item for item in self._pending_messages if item[1].received_at <= now]
        self._pending_messages = [item for item in self._pending_messages if item[1].received_at > now]
        for kind, message, delivery_ordinal in sorted(ready, key=lambda item: (item[1].received_at, item[1].entity, item[1].field, item[2])):
            if now - message.measured_at > self.config.telemetry_max_age:
                self._communication_log.append({
                    "link": "telemetry", "status": "expired",
                    "entity": message.entity, "field": message.field,
                    "sequence": message.sequence, "message_id": message.message_id,
                    "delivery_ordinal": delivery_ordinal, "time": now,
                    "measured_at": message.measured_at,
                    "received_at": message.received_at,
                })
                continue
            self._accept_message(kind, message, now, delivery_ordinal=delivery_ordinal)

    def _deliver_observations(self) -> None:
        now = self.clock.time
        contract = self.config.contract_v1
        uav_heartbeat_due = bool(
            contract and contract.reporting_mode == "event" and
            now - self._contract_last_uav_heartbeat >=
            contract.effective_uav_heartbeat_interval
        )
        task_telemetry_heartbeat_due = bool(
            contract and contract.reporting_mode == "event" and
            now - self._contract_last_task_telemetry_heartbeat >=
            contract.effective_task_telemetry_heartbeat_interval
        )
        for uid, resource in self.clock.resources.items():
            # A destroyed UAV cannot manufacture a post-damage notice.  The
            # controller therefore observes stale/unknown state until another
            # legal observer reports it.
            if contract is not None and not resource.alive:
                continue
            occupied = any(task.assigned_uav == uid for task in self.clock.tasks.values())
            values = {
                "x": resource.position[0], "y": resource.position[1], "energy": resource.energy,
                "alive": float(resource.alive), "connected": float(resource.connected), "idle": float(not occupied),
            }
            for field, value in values.items():
                if self._contract_should_report(
                        "uav", uid, field, value, now, force=uav_heartbeat_due):
                    message_kind = "event" if contract and contract.reporting_mode == "event" else "telemetry"
                    if (contract and contract.reporting_mode == "event" and field == "energy" and
                            value <= contract.low_energy_threshold):
                        message_kind = "event"
                        self._send_contract_event("low_energy_threshold", uid, now,
                                                  {"source": "uav_local_observer", "threshold": contract.low_energy_threshold})
                    self._send("uav", uid, field, value, message_kind=message_kind)
        for task_id, task in self.clock.tasks.items():
            if task.state == TaskState.UNRELEASED or now < task.arrival:
                continue
            spec = self._task_by_id[task_id]
            values = {
                "x": spec.x, "y": spec.y, "deadline": task.deadline,
                "remaining_service": max(0.0, task.required_service - task.service),
                "priority": task.priority, "pending": float(task.state == TaskState.PENDING),
                "region_id": float(spec.region_id), "target_id": float(spec.target_id),
            }
            emergency_ready = bool(spec.emergency and task_id not in self._contract_published_emergencies)
            if emergency_ready:
                self._send_contract_event("emergency_task_announcement", task_id, now,
                                          {"source": spec.source, "actual_arrival": spec.arrival,
                                           "deadline": spec.deadline, "emergency": True})
            for field, value in values.items():
                force = task_telemetry_heartbeat_due or emergency_ready
                if self._contract_should_report("task", task_id, field, value, now, force=force):
                    message_kind = "telemetry"
                    if emergency_ready:
                        message_kind = "emergency_announcement"
                    self._send("task", task_id, field, value, message_kind=message_kind)
            if emergency_ready:
                self._contract_published_emergencies.add(task_id)
        if uav_heartbeat_due:
            self._contract_last_uav_heartbeat = now
        if task_telemetry_heartbeat_due:
            self._contract_last_task_telemetry_heartbeat = now
        self._flush_contract_pending_sends()

    def _observation(self, *, clear_trigger: bool = False) -> dict[str, Any]:
        snapshot: TaskPolicySnapshot = self.bridge.observe()
        uavs = np.asarray(snapshot.uavs, dtype=np.float32)
        tasks = np.asarray(snapshot.tasks, dtype=np.float32)
        node_width = 32
        uav_nodes = np.pad(uavs, ((0, 0), (0, node_width - uavs.shape[1])))
        task_nodes = np.asarray(tasks, dtype=np.float32)
        regions = np.zeros((self.config.region_count, node_width), dtype=np.float32)
        targets = np.zeros((self.config.target_count, node_width), dtype=np.float32)
        for index in range(self.config.region_count):
            regions[index, 0] = float(index)
        for index in range(self.config.target_count):
            targets[index, 0] = float(index)
        for row in task_nodes:
            values = row[::4]
            valid = row[2::4]
            if len(values) < 8 or not np.all(valid[:8] > 0.5):
                continue
            region_id = int(round(float(values[6])))
            target_id = int(round(float(values[7])))
            if 0 <= region_id < self.config.region_count:
                regions[region_id, 1] += 1.0
                regions[region_id, 2] += float(values[3])
                regions[region_id, 3] += float(values[4])
                regions[region_id, 4] = float(values[2]) if regions[region_id, 4] == 0 else min(regions[region_id, 4], float(values[2]))
                regions[region_id, 5] = 1.0
            if 0 <= target_id < self.config.target_count:
                targets[target_id, 1] += 1.0
                targets[target_id, 2] += float(values[3])
                targets[target_id, 3] += float(values[4])
                targets[target_id, 4] = float(values[2]) if targets[target_id, 4] == 0 else min(targets[target_id, 4], float(values[2]))
                targets[target_id, 5] = 1.0
        events = np.zeros((self.config.event_capacity, node_width), dtype=np.float32)
        for index, record in enumerate(self._public_event_records[-self.config.event_capacity:]):
            events[index, 0] = 1.0
            events[index, 1] = float(record["field"] == "connected")
            events[index, 2] = float(record["value"])
            events[index, 3] = float(self.uav_ids.index(record["entity"])) / max(1, self.config.uav_count - 1)
            events[index, 4] = max(0.0, self.clock.time - float(record["received_at"]))
            events[index, 5] = float(record["measured_at"]) / self.config.horizon
        relations = np.zeros((self.config.uav_count, self.config.task_capacity, self.config.relation_width), dtype=np.float32)
        for uav_index, uav in enumerate(uavs):
            ux, uy = float(uav[0]), float(uav[4])
            for task_index, task in enumerate(task_nodes):
                tx, ty = float(task[0]), float(task[4])
                visible = float(np.all(task[2::4] > 0.5))
                distance = math.dist((ux, uy), (tx, ty)) / 10.0 if visible else 0.0
                relations[uav_index, task_index] = (distance, visible, float(snapshot.mask[uav_index * self.config.task_capacity + task_index]), float(task[24]) if visible else 0.0)
        # Ordinary telemetry is deliberately not an event trigger.  These
        # flags are raised only by a newly visible semantic change or by a
        # safety gate; the policy never reads the private event schedule.
        trigger_flags = dict(self._trigger_flags)
        event_signal = float(any(trigger_flags.values()))
        flat = np.concatenate((uav_nodes.reshape(-1), regions.reshape(-1), targets.reshape(-1), task_nodes.reshape(-1), events.reshape(-1), relations.reshape(-1), np.asarray([self.clock.time / self.config.horizon, event_signal], dtype=np.float32)))
        result = {
            "flat": flat,
            "uavs": uavs,
            "tasks": tasks,
            "graph": {"node_features": np.concatenate((uav_nodes, regions, targets, task_nodes, events), axis=0), "relations": relations},
            "mask": np.asarray(snapshot.mask, dtype=np.bool_),
            "time": float(self.clock.time),
            "version": int(snapshot.version),
            "types": ("UAV", "Region", "Target", "Task", "Event"),
            "trigger_flags": trigger_flags,
            "event_signal": event_signal,
            # This is an ACKed continuation handle, not a candidate mask.  It
            # lets the controller distinguish a running lease from an unsafe
            # stale allocation candidate without exposing execution truth in
            # the learned feature vector.
            "continuation_action": self._active_action,
            "continuation_actions": tuple(sorted(self._active_actions.values())),
        }
        if clear_trigger:
            self._trigger_flags = {key: False for key in self._trigger_flags}
        return result

    def _all_terminal_or_future_empty(self) -> bool:
        return all(task.state in (TaskState.COMPLETED, TaskState.EXPIRED) for task in self.clock.tasks.values()) and self.clock.cursor >= len(self.clock.events)

    def _reward_and_counts(self, feedback: str | TaskCommand) -> tuple[float, dict[str, int]]:
        if self.config.task_completion_mode == "arrival_to_region":
            completed = 0
            expired = 0
            for task_id, task in self.clock.tasks.items():
                record = self._completion_records.get(task_id)
                if self.config.deadline_basis == "host_confirmation":
                    success = bool(record and record.get("host_confirmation_before_deadline"))
                else:
                    success = bool(record and record.get("physical_arrival_before_deadline"))
                if success:
                    completed += 1
                elif task.state == TaskState.EXPIRED or (task.state == TaskState.COMPLETED and self.clock.time >= task.deadline):
                    expired += 1
        else:
            completed = sum(task.state == TaskState.COMPLETED for task in self.clock.tasks.values())
            expired = sum(task.state == TaskState.EXPIRED for task in self.clock.tasks.values())
        rejected = sum(item.get("result") not in ("accepted", "awaiting_ack", "noop", "reuse_existing") for item in self._feedback_log)
        energy_now = sum(resource.energy for resource in self.clock.resources.values())
        energy_used = max(0.0, self._last_energy - energy_now)
        self._last_energy = energy_now
        reward = -self.config.energy_cost_weight * energy_used
        if feedback not in ("noop", "reuse_existing") and not isinstance(feedback, TaskCommand):
            reward -= self.config.penalty_rejected
        reward += self.config.reward_completion * (completed - self._last_completed)
        reward -= self.config.penalty_expired * (expired - self._last_expired)
        self._last_completed, self._last_expired = completed, expired
        return reward, {"completed": int(completed), "expired": int(expired), "rejected": int(rejected)}

    def step(self, action: int, *, submit_command: bool = True) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if type(action) is not int or not 0 <= action < self.config.action_count:
            raise ValueError("action outside fixed M10 action space")
        self._lease_renewal_results = {}
        # Deliver requests whose transport delay has elapsed before creating
        # the next public snapshot.  No executor lease table is inspected.
        self._deliver_pending_renewals(self.clock.time)
        renewal_delivery_results = dict(self._lease_renewal_results)
        obs = self._observation()
        communication_start = len(self._communication_log)
        event_log_before = len(self.clock.log)
        command_id: str | None = None
        command_task_id: str | None = None
        lease_renewal: str | dict[str, str] = "not_applicable"
        lease_renewals: dict[str, str] = {}
        newly_accepted: str | None = None
        if submit_command:
            self._command_index += 1
            command_id = f"{self._episode_id}-cmd-{self._command_index:05d}"
            command_identity = f"{command_id}|{obs['version']}|{action}"
            command_budget_ok = self._contract_control_send_allowed(
                self.clock.time, "downlink", "command", command_identity)
            command_delivered = bool(command_budget_ok and self.communication.command_delivered(
                seed=self.scenario.seed, identity=self._random_identity(command_identity)))
            self._communication_log.append({"link": "command", "status": "sent" if command_delivered else
                                            ("budget_rejected" if not command_budget_ok else "dropped"),
                                            "command_id": command_id, "time": self.clock.time})
            feedback = (self.bridge.submit(action, version=obs["version"], command_id=command_id)
                        if command_delivered else ("command_budget_rejected" if not command_budget_ok else "command_lost"))
            if isinstance(feedback, TaskCommand):
                command_task_id = feedback.task_id
                ack_result = self.execution.acknowledge(feedback.command_id, feedback.uav_id, feedback.token)
                if ack_result == "accepted":
                    ack_budget_ok = self._contract_control_send_allowed(
                        self.clock.time, "uplink", "ack", f"{command_identity}|ack")
                    ack_delivered = bool(ack_budget_ok and self.communication.ack_delivered(
                        seed=self.scenario.seed, identity=self._random_identity(command_identity)))
                    self._communication_log.append({"link": "ack", "status": "received" if ack_delivered else "dropped",
                                                    "command_id": command_id, "time": self.clock.time})
                    if ack_delivered:
                        # The control side may maintain only an ACK-confirmed
                        # handle. An execution-side acceptance with a lost ACK
                        # is intentionally not converted into hidden control
                        # knowledge.
                        self._remember_active(feedback, action)
                        newly_accepted = command_id
                    else:
                        feedback = "ack_lost_after_accept" if ack_budget_ok else "ack_budget_rejected_after_accept"
                else:
                    feedback = ack_result
            elif feedback == "noop":
                ack_result = "noop"
            else:
                ack_result = str(feedback)
        else:
            # A non-replanning interval creates no allocation command. Lease
            # maintenance below renews every already ACKed continuation.
            feedback = "reuse_existing"
            ack_result = "reuse_existing"
        # Existing tasks continue even when this step submits a new task or a
        # NOOP. The new lease receives its first renewal on the next cycle.
        lease_renewals = self._renew_active_leases(
            skip={newly_accepted} if newly_accepted is not None else set(),
        )
        if lease_renewals:
            lease_renewal = (next(iter(lease_renewals.values()))
                             if len(lease_renewals) == 1 else dict(lease_renewals))
        self._feedback_log.append({"command_id": command_id, "result": str(ack_result), "time": self.clock.time})
        if command_id is not None and command_task_id is not None:
            ledger = self._contract_ledger.get(command_task_id)
            if ledger is not None:
                ledger.mark("command_sent", float(obs["time"]))
                if ack_result == "accepted":
                    ledger.mark("command_accepted", float(obs["time"]))
        target_time = min(self.config.horizon, self.clock.time + self.config.decision_interval)
        assignments_before = {task_id: task.assigned_uav for task_id, task in self.clock.tasks.items()}
        self._advance_execution(target_time)
        if self._contract_enabled():
            new_clock_events = self.clock.log[event_log_before:]
            for event in new_clock_events:
                if event.get("kind") in ("damage", "disconnect"):
                    for task_id, uid in assignments_before.items():
                        if uid == event.get("resource"):
                            ledger = self._contract_ledger.get(task_id)
                            if ledger is not None:
                                interruption_id = f"{task_id}|interruption|{len(ledger.interruption) + 1}"
                                ledger.add_interruption(interruption_id, float(event["time"]), str(event["kind"]))
                                self._contract_open_interruptions[task_id] = interruption_id
                task_id = str(event.get("task", ""))
                ledger = self._contract_ledger.get(task_id)
                if ledger is None:
                    continue
                if event.get("kind") in ("travel", "service") and ledger.effective_execution_start is None:
                    ledger.mark("effective_execution_start", float(event["start"]))
                elif event.get("kind") in ("travel", "service") and task_id in self._contract_open_interruptions:
                    ledger.add_recovery(self._contract_open_interruptions.pop(task_id),
                                        float(event["start"]), str(event["resource"]))
                if event.get("kind") == "arrival" and ledger.physical_arrival is None:
                    ledger.mark("physical_arrival", float(event["time"]))
        self._deliver_observations()
        self._flush_messages()
        self._step_index += 1
        next_obs = self._observation(clear_trigger=True)
        reward, counts = self._reward_and_counts(feedback)
        task_terminal = self._all_terminal_or_future_empty()
        time_limit = self.clock.time >= self.config.horizon
        terminated = bool(task_terminal)
        truncated = bool(time_limit and not terminated)
        info = {
            "feedback": str(ack_result),
            "command_submitted": bool(submit_command),
            "command_id": command_id,
            "lease_renewal": lease_renewal,
            "lease_renewals": dict(lease_renewals),
            "lease_renewal_delivery_results": renewal_delivery_results,
            "active_continuations": [
                {"command_id": command.command_id, "task_id": command.task_id,
                 "uav_id": command.uav_id, "token": command.token,
                 "action": self._active_actions[command.command_id]}
                for command in self._active_commands.values()
            ],
            "step": self._step_index,
            "time": self.clock.time,
            "counts": counts,
            "energy": {uid: resource.energy for uid, resource in self.clock.resources.items()},
            "tasks": {task_id: task.state.value for task_id, task in self.clock.tasks.items()},
            "task_service": {task_id: task.service for task_id, task in self.clock.tasks.items()},
            "event_log": list(self.clock.log),
            "new_events": list(self.clock.log[event_log_before:]),
            "feedback_log": list(self._feedback_log),
            "communication_log": list(self._communication_log),
            "communication_delta": list(self._communication_log[communication_start:]),
            "delivered_message_count": len(self._delivered_messages),
            "policy_version": next_obs["version"],
            "trigger_flags": dict(next_obs["trigger_flags"]),
            "terminated": terminated,
            "truncated": truncated,
            "episode_end_reason": "terminated" if terminated else "time_limit" if truncated else None,
            "deadline_basis": self.config.deadline_basis,
            "task_completion_mode": self.config.task_completion_mode,
            "completion_records": {key: dict(value) for key, value in self._completion_records.items()},
            "contract_version": (self.config.contract_v1.version if self.config.contract_v1 else "legacy"),
            "contract_event_log": list(self._contract_event_log),
            "contract_time_ledger": {key: value.to_dict() for key, value in self._contract_ledger.items()},
            "contract_budget": (self._contract_budget.snapshot() if self._contract_budget else None),
        }
        return next_obs, float(reward), bool(terminated or truncated), info

    def action_mask(self) -> np.ndarray:
        return self._observation()["mask"].copy()

    def public_snapshot_digest(self) -> tuple[int, tuple[float, ...], tuple[bool, ...]]:
        obs = self._observation()
        return int(obs["version"]), tuple(float(x) for x in obs["flat"]), tuple(bool(x) for x in obs["mask"])


__all__ = ["M10Config", "M10TaskSpec", "M10Scenario", "M10Environment", "default_scenario", "scenario_tape", "weak_communication_tape", "scenario_to_dict", "scenario_from_dict", "m10_config_to_dict", "m10_config_from_dict"]
