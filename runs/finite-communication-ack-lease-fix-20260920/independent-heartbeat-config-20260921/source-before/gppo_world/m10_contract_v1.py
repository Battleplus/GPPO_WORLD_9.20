"""Versioned finite-communication contract for the M-10 simulator.

This module is deliberately independent from the legacy transport path.  It
provides a token bucket for *send attempts* (not real network bandwidth), a
small emergency-announcement record, and a single timestamp ledger.  The
environment opts in explicitly through ``M10Config.contract_v1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Literal


Direction = Literal["uplink", "downlink"]


@dataclass(frozen=True)
class ContractBudgetConfig:
    """Fixture-level token bucket parameters in simulator-time units."""

    uplink_rate: float = 8.0
    uplink_capacity: float = 8.0
    downlink_rate: float = 8.0
    downlink_capacity: float = 8.0
    uplink_queue_capacity: int = 32
    downlink_queue_capacity: int = 32
    message_ttl: float = 2.5
    # Same-bucket floor reserved for ACK/lease control sends. Zero preserves v1.
    control_reserve_uplink: float = 0.0
    control_reserve_downlink: float = 0.0
    # Lower number means earlier service.  The values are protocol data, not
    # a learned or result-selected policy.
    priority: tuple[tuple[str, int], ...] = (
        ("emergency_announcement", 0),
        ("event", 1),
        ("completion", 1),
        ("ack", 0),
        ("command", 0),
        ("lease_renewal", 1),
        ("telemetry", 2),
    )

    def __post_init__(self) -> None:
        values = (self.uplink_rate, self.uplink_capacity, self.downlink_rate,
                  self.downlink_capacity, self.message_ttl,
                  self.control_reserve_uplink, self.control_reserve_downlink)
        if any(not math.isfinite(float(v)) or float(v) < 0 for v in values):
            raise ValueError("contract budget rates, capacities and TTL must be finite")
        if self.uplink_capacity <= 0 or self.downlink_capacity <= 0:
            raise ValueError("contract bucket capacities must be positive")
        if self.uplink_queue_capacity <= 0 or self.downlink_queue_capacity <= 0:
            raise ValueError("contract queue capacities must be positive")
        if self.message_ttl <= 0:
            raise ValueError("contract message TTL must be positive")
        if (self.control_reserve_uplink > self.uplink_capacity or
                self.control_reserve_downlink > self.downlink_capacity):
            raise ValueError("control reserve cannot exceed bucket capacity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "uplink_rate": self.uplink_rate,
            "uplink_capacity": self.uplink_capacity,
            "downlink_rate": self.downlink_rate,
            "downlink_capacity": self.downlink_capacity,
            "uplink_queue_capacity": self.uplink_queue_capacity,
            "downlink_queue_capacity": self.downlink_queue_capacity,
            "message_ttl": self.message_ttl,
            "control_reserve_uplink": self.control_reserve_uplink,
            "control_reserve_downlink": self.control_reserve_downlink,
            "priority": [list(item) for item in self.priority],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ContractBudgetConfig":
        if not payload:
            return cls()
        return cls(
            uplink_rate=float(payload.get("uplink_rate", 8.0)),
            uplink_capacity=float(payload.get("uplink_capacity", 8.0)),
            downlink_rate=float(payload.get("downlink_rate", 8.0)),
            downlink_capacity=float(payload.get("downlink_capacity", 8.0)),
            uplink_queue_capacity=int(payload.get("uplink_queue_capacity", 32)),
            downlink_queue_capacity=int(payload.get("downlink_queue_capacity", 32)),
            message_ttl=float(payload.get("message_ttl", 2.5)),
            control_reserve_uplink=float(payload.get("control_reserve_uplink", 0.0)),
            control_reserve_downlink=float(payload.get("control_reserve_downlink", 0.0)),
            priority=(tuple((str(item[0]), int(item[1])) for item in payload["priority"])
                      if "priority" in payload else cls().priority),
        )


@dataclass(frozen=True)
class ContractV1Config:
    """Opt-in versioned contract; legacy defaults remain unchanged."""

    version: str = "finite-communication-emergency-v1"
    reporting_mode: str = "periodic"
    heartbeat_interval: float = 2.0
    low_energy_threshold: float = 1.0
    budget: ContractBudgetConfig = field(default_factory=ContractBudgetConfig)
    enable_emergency_announcements: bool = True
    enable_event_reports: bool = True
    enable_task_preemption: bool = False
    queue_admission_policy: str = "fifo"
    # Development-only candidate.  The default preserves the frozen contract.
    enable_uav_state_coalescing: bool = False
    # Development-only candidate.  The default preserves ordinary completion
    # admission and service behavior.
    enable_completion_control_eligibility: bool = False

    def __post_init__(self) -> None:
        if self.version not in ("finite-communication-emergency-v1",
                                "finite-communication-control-reserve-v1"):
            raise ValueError("unsupported contract version")
        if self.reporting_mode not in ("periodic", "event"):
            raise ValueError("reporting_mode must be periodic or event")
        if self.queue_admission_policy not in ("fifo", "priority_replace_merge"):
            raise ValueError("unsupported queue admission policy")
        if not math.isfinite(self.heartbeat_interval) or self.heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")
        if not math.isfinite(self.low_energy_threshold) or self.low_energy_threshold < 0:
            raise ValueError("low_energy_threshold must be finite and nonnegative")
        if self.enable_task_preemption:
            raise ValueError("preemption is intentionally disabled in v1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "reporting_mode": self.reporting_mode,
            "heartbeat_interval": self.heartbeat_interval,
            "low_energy_threshold": self.low_energy_threshold,
            "budget": self.budget.to_dict(),
            "enable_emergency_announcements": self.enable_emergency_announcements,
            "enable_event_reports": self.enable_event_reports,
            "enable_task_preemption": self.enable_task_preemption,
            "queue_admission_policy": self.queue_admission_policy,
            "enable_uav_state_coalescing": self.enable_uav_state_coalescing,
            "enable_completion_control_eligibility": self.enable_completion_control_eligibility,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ContractV1Config | None":
        if not payload:
            return None
        return cls(
            version=str(payload.get("version", "finite-communication-emergency-v1")),
            reporting_mode=str(payload.get("reporting_mode", "periodic")),
            heartbeat_interval=float(payload.get("heartbeat_interval", 2.0)),
            low_energy_threshold=float(payload.get("low_energy_threshold", 1.0)),
            budget=ContractBudgetConfig.from_dict(payload.get("budget")),
            enable_emergency_announcements=bool(payload.get("enable_emergency_announcements", True)),
            enable_event_reports=bool(payload.get("enable_event_reports", True)),
            enable_task_preemption=bool(payload.get("enable_task_preemption", False)),
            queue_admission_policy=str(payload.get("queue_admission_policy", "fifo")),
            enable_uav_state_coalescing=bool(payload.get("enable_uav_state_coalescing", False)),
            enable_completion_control_eligibility=bool(
                payload.get("enable_completion_control_eligibility", False)
            ),
        )


@dataclass(frozen=True)
class BudgetDecision:
    status: str
    direction: Direction
    message_kind: str
    time: float
    token_before: float
    token_after: float
    queue_depth: int
    reason: str | None = None


class SendBudgetExecutor:
    """Token bucket where every actual send attempt consumes one token.

    ``reserve`` only records a send when a token is available.  A caller may
    enqueue a no-token message, but that message has not yet been sent and
    therefore has consumed no token.  Drops after reservation never refund it.
    """

    def __init__(self, config: ContractBudgetConfig):
        self.config = config
        self.tokens = {
            "uplink": float(config.uplink_capacity),
            "downlink": float(config.downlink_capacity),
        }
        self.last_time = 0.0
        self.queue_depth = {"uplink": 0, "downlink": 0}
        self.attempts = {"uplink": 0, "downlink": 0}
        self.records: list[dict[str, Any]] = []
        self._priority = dict(config.priority)

    def _advance(self, now: float) -> None:
        if not math.isfinite(now) or now < self.last_time:
            raise ValueError("budget time must be finite and monotonic")
        delta = now - self.last_time
        rates = {"uplink": self.config.uplink_rate, "downlink": self.config.downlink_rate}
        caps = {"uplink": self.config.uplink_capacity, "downlink": self.config.downlink_capacity}
        for direction in ("uplink", "downlink"):
            self.tokens[direction] = min(caps[direction], self.tokens[direction] + delta * rates[direction])
        self.last_time = now

    def advance(self, now: float) -> None:
        """Advance token clocks without reserving a send."""
        self._advance(now)

    def reserve(self, now: float, direction: Direction, message_kind: str,
                *, control: bool = False) -> BudgetDecision:
        if direction not in ("uplink", "downlink"):
            raise ValueError("unknown budget direction")
        self._advance(now)
        before = self.tokens[direction]
        reserve_floor = 0.0
        if not control:
            reserve_floor = (self.config.control_reserve_uplink
                             if direction == "uplink"
                             else self.config.control_reserve_downlink)
        if before >= 1.0 - 1e-12 and before - 1.0 >= reserve_floor - 1e-12:
            self.tokens[direction] = before - 1.0
            self.attempts[direction] += 1
            decision = BudgetDecision("sent", direction, message_kind, now, before,
                                      self.tokens[direction], self.queue_depth[direction])
        else:
            reason = "control_reserve" if before >= 1.0 - 1e-12 else "no_token"
            decision = BudgetDecision("queued", direction, message_kind, now, before,
                                      before, self.queue_depth[direction], reason)
        if decision.status == "sent":
            self.records.append({**decision.__dict__, "token_before": before,
                                 "token_after": self.tokens[direction]})
        return decision

    def reserve_control(self, now: float, direction: Direction,
                        message_kind: str) -> BudgetDecision:
        """Reserve a control send from the same bucket, including its floor."""
        return self.reserve(now, direction, message_kind, control=True)

    def request(self, now: float, direction: Direction, message_kind: str) -> BudgetDecision:
        """Reserve now or enqueue the not-yet-sent message."""
        decision = self.reserve(now, direction, message_kind)
        if decision.status == "queued":
            return self.enqueue(now, direction, message_kind)
        return decision

    def enqueue(self, now: float, direction: Direction, message_kind: str) -> BudgetDecision:
        self._advance(now)
        limit = (self.config.uplink_queue_capacity if direction == "uplink"
                 else self.config.downlink_queue_capacity)
        before = self.tokens[direction]
        if self.queue_depth[direction] >= limit:
            decision = BudgetDecision("budget_rejected", direction, message_kind, now, before,
                                      before, self.queue_depth[direction], "queue_full")
        else:
            self.queue_depth[direction] += 1
            decision = BudgetDecision("queued", direction, message_kind, now, before,
                                      before, self.queue_depth[direction], "no_token")
        self.records.append({**decision.__dict__})
        return decision

    def dequeue(self, direction: Direction) -> None:
        if self.queue_depth[direction] <= 0:
            raise ValueError("budget queue underflow")
        self.queue_depth[direction] -= 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "tokens": dict(self.tokens),
            "last_time": self.last_time,
            "queue_depth": dict(self.queue_depth),
            "attempts": dict(self.attempts),
            "records": list(self.records),
        }


@dataclass(frozen=True)
class EmergencyTaskAnnouncement:
    task_id: str
    actual_arrival: float
    deadline: float
    priority: float
    x: float
    y: float
    service: float
    source: str = "external_task_publisher"
    emergency: bool = True

    def __post_init__(self) -> None:
        if not self.emergency or self.source != "external_task_publisher":
            raise ValueError("v1 emergency announcements require the declared publisher")
        if not all(math.isfinite(float(v)) for v in
                   (self.actual_arrival, self.deadline, self.priority, self.x, self.y, self.service)):
            raise ValueError("announcement values must be finite")
        if self.actual_arrival < 0 or self.deadline <= self.actual_arrival or self.priority <= 0 or self.service <= 0:
            raise ValueError("invalid emergency announcement timeline or task values")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ContractTimeLedger:
    """One nullable timeline shared by ordinary and emergency tasks/events."""

    task_id: str
    actual_arrival: float | None = None
    first_legal_awareness: float | None = None
    command_sent: float | None = None
    command_accepted: float | None = None
    effective_execution_start: float | None = None
    physical_arrival: float | None = None
    host_confirmation: float | None = None
    interruption: list[dict[str, Any]] = field(default_factory=list)
    replacement_execution_start: list[dict[str, Any]] = field(default_factory=list)
    null_reasons: dict[str, str] = field(default_factory=dict)

    def mark(self, field_name: str, value: float | None, *, reason: str | None = None) -> None:
        if field_name not in {
            "actual_arrival", "first_legal_awareness", "command_sent", "command_accepted",
            "effective_execution_start", "physical_arrival", "host_confirmation",
        }:
            raise ValueError("unknown contract time field")
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("contract time must be finite and nonnegative")
        setattr(self, field_name, value)
        if value is None and reason is not None:
            self.null_reasons[field_name] = reason

    def add_interruption(self, interruption_id: str, time: float, reason: str) -> None:
        if not interruption_id or not math.isfinite(time) or time < 0:
            raise ValueError("invalid interruption record")
        self.interruption.append({"interruption_id": interruption_id, "time": time, "reason": reason})

    def add_recovery(self, interruption_id: str, time: float, resource: str) -> None:
        if not interruption_id or not resource or not math.isfinite(time) or time < 0:
            raise ValueError("invalid recovery record")
        self.replacement_execution_start.append({
            "interruption_id": interruption_id, "time": time, "resource": resource,
        })

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def canonical_audit_bytes(payload: dict[str, Any]) -> int:
    """Return serialization-proxy bytes; never claim this is wire traffic."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return len(encoded)


def contract_digest(config: ContractV1Config) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "ContractBudgetConfig", "ContractV1Config", "BudgetDecision", "SendBudgetExecutor",
    "EmergencyTaskAnnouncement", "ContractTimeLedger", "canonical_audit_bytes", "contract_digest",
]
