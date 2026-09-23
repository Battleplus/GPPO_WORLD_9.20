"""Deterministic public-priority dispatch with the existing public guard.

The rule is an action proposal policy for the frozen ``4 UAV x 6 task +
NOOP`` interface.  It consumes only the public observation fields emitted by
``TaskPolicyView`` and never inspects an environment, model, replay, lease, or
execution result.  The public action mask is preserved as supplied; it is a
proposal filter and is not rewritten by this module.

The ranking policy is ``public-priority-edf-nearest-v1``:

``priority (descending), deadline (ascending), 2-D distance (ascending),
public UAV energy (descending), action id (ascending)``.

The returned rank carrier is deliberately *not* a calibrated probability.
It only gives the existing ACK-known-task guard a deterministic ordering.  A
legal NOOP remains in the carrier with the next rank weight, so that it wins
when every allocation is masked or excluded by the guard.  The carrier is
normalized over the original legal actions to satisfy the public probability
ledger contract while retaining this ordering.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

from .ack_known_task_guard import (
    GuardDecision,
    NoLegalActionError,
    PublicIdentityError,
    select_guarded_action,
)


UAV_COUNT = 4
TASK_CAPACITY = 6
NOOP_ACTION = UAV_COUNT * TASK_CAPACITY
ACTION_COUNT = NOOP_ACTION + 1
RULE_NAME = "public-priority-edf-nearest-v1"

UAV_FIELDS = ("x", "y", "energy", "alive", "connected", "idle")
TASK_FIELDS = (
    "x",
    "y",
    "deadline",
    "remaining_service",
    "priority",
    "pending",
    "region_id",
    "target_id",
)
UAV_REQUIRED_FIELDS = UAV_FIELDS
TASK_REQUIRED_FIELDS = ("x", "y", "deadline", "remaining_service", "priority", "pending")
BOOLEAN_FIELDS = frozenset(("alive", "connected", "idle", "pending"))
NONNEGATIVE_FIELDS = frozenset(("energy", "deadline", "remaining_service", "priority"))


class PublicDispatchContractError(ValueError):
    """The public dispatch observation cannot satisfy the frozen contract."""


@dataclass(frozen=True)
class _Channel:
    value: float
    known: bool
    valid: bool
    age: float


@dataclass(frozen=True)
class CandidateRank:
    """One legal action's auditable public rank information."""

    action: int
    uav_id: str | None
    task_id: str | None
    rank_key: tuple[float, ...] | None
    eligible: bool
    reason: str


@dataclass(frozen=True)
class PublicDispatchDecision:
    """Auditable rule and guard result for one public observation."""

    rule: str
    selected_action: int
    original_argmax: int
    order: tuple[int, ...]
    candidate_ranks: tuple[CandidateRank, ...]
    rank_based_probabilities: tuple[float, ...]
    legal_actions: tuple[int, ...]
    guard_decision: GuardDecision

    @property
    def final_action(self) -> int:
        """Compatibility spelling used by the existing guard decision."""

        return self.selected_action

    @property
    def probabilities(self) -> tuple[float, ...]:
        """The rank carrier passed to ``select_guarded_action``."""

        return self.rank_based_probabilities

    @property
    def candidate_actions(self) -> tuple[int, ...]:
        return self.guard_decision.candidate_actions

    @property
    def excluded_actions(self) -> tuple[int, ...]:
        return self.guard_decision.excluded_actions

    @property
    def rank_keys(self) -> tuple[tuple[int, tuple[float, ...] | None], ...]:
        """Action/key pairs in the returned deterministic candidate order."""

        return tuple((item.action, item.rank_key) for item in self.candidate_ranks)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly audit data without touching the input."""

        return {
            "rule": self.rule,
            "selected_action": self.selected_action,
            "final_action": self.final_action,
            "original_argmax": self.original_argmax,
            "order": list(self.order),
            "rank_keys": [
                {
                    "action": item.action,
                    "uav_id": item.uav_id,
                    "task_id": item.task_id,
                    "rank_key": list(item.rank_key) if item.rank_key is not None else None,
                    "eligible": item.eligible,
                    "reason": item.reason,
                }
                for item in self.candidate_ranks
            ],
            "rank_based_probabilities": list(self.rank_based_probabilities),
            "legal_actions": list(self.legal_actions),
            "guard": {
                "original_action": self.guard_decision.original_action,
                "final_action": self.guard_decision.final_action,
                "triggered": self.guard_decision.triggered,
                "reason": self.guard_decision.reason,
                "continuation_task_ids": list(self.guard_decision.continuation_task_ids),
                "excluded_actions": list(self.guard_decision.excluded_actions),
                "candidate_actions": list(self.guard_decision.candidate_actions),
            },
        }


def _sequence(value: Any, field: str) -> tuple[Any, ...]:
    """Materialize a sequence without truth-testing numpy-like arrays."""

    if isinstance(value, (str, bytes, bytearray, Mapping)) or value is None:
        raise PublicDispatchContractError(f"{field} must be a sequence")
    try:
        size = len(value)
    except (TypeError, AttributeError) as exc:
        raise PublicDispatchContractError(f"{field} must be a sized sequence") from exc
    result = []
    try:
        for index in range(size):
            result.append(value[index])
    except (IndexError, KeyError, TypeError, AttributeError) as exc:
        raise PublicDispatchContractError(f"{field} is not indexable") from exc
    return tuple(result)


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, (str, bytes, bytearray, complex)):
        raise PublicDispatchContractError(f"{field} must be a finite real number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PublicDispatchContractError(f"{field} must be a finite real number") from exc
    if not math.isfinite(number):
        raise PublicDispatchContractError(f"{field} must be finite")
    return number


def _float_allow_nonfinite(value: Any, field: str) -> float:
    """Decode a numeric placeholder without using it as a score.

    Native missing telemetry is ``0.0/False/False/0.0``.  A caller may also
    carry a masked placeholder in an array-like row.  It is decoded for shape
    and flag validation, then rejected if the row is a legal scoring target.
    """

    if isinstance(value, (str, bytes, bytearray, complex)):
        raise PublicDispatchContractError(f"{field} must be a real number")
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PublicDispatchContractError(f"{field} must be a real number") from exc


def _binary(value: Any, field: str) -> bool:
    number = _finite_float(value, field)
    if number not in (0.0, 1.0):
        raise PublicDispatchContractError(f"{field} must be 0 or 1")
    return bool(number)


def _integer(value: Any, field: str) -> int:
    if isinstance(value, (str, bytes, bytearray, bool, complex)):
        raise PublicDispatchContractError(f"{field} must be an integer")
    number = _finite_float(value, field)
    if not number.is_integer():
        raise PublicDispatchContractError(f"{field} must be an integer")
    return int(number)


def _public_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise PublicDispatchContractError(f"{field} must be a nonempty string")
    return value


def _decode_rows(value: Any, *, rows: int, fields: tuple[str, ...], name: str) -> tuple[dict[str, _Channel], ...]:
    outer = _sequence(value, name)
    if len(outer) != rows:
        raise PublicDispatchContractError(f"{name} must contain exactly {rows} rows")
    decoded = []
    for row_index, raw_row in enumerate(outer):
        row = _sequence(raw_row, f"{name}[{row_index}]")
        expected = 4 * len(fields)
        if len(row) != expected:
            raise PublicDispatchContractError(
                f"{name}[{row_index}] must contain {expected} four-channel values"
            )
        values: dict[str, _Channel] = {}
        for field_index, field in enumerate(fields):
            offset = field_index * 4
            # Value-domain and finite checks are deferred until this row is
            # proven to back an original legal allocation.  Unknown/padded
            # rows are public placeholders and are never scored.
            value_channel = _float_allow_nonfinite(
                row[offset], f"{name}[{row_index}].{field}.value"
            )
            known = _binary(row[offset + 1], f"{name}[{row_index}].{field}.known")
            valid = _binary(row[offset + 2], f"{name}[{row_index}].{field}.valid")
            # Missing native telemetry uses age=0.0.  Other placeholders are
            # carried without scoring and are checked when required below.
            age = _float_allow_nonfinite(row[offset + 3], f"{name}[{row_index}].{field}.age")
            values[field] = _Channel(value_channel, known, valid, age)
        decoded.append(values)
    return tuple(decoded)


def _available(channel: _Channel) -> bool:
    return channel.known and channel.valid


def _validate_scoring_channel(channel: _Channel, field: str) -> float:
    if not _available(channel):
        raise PublicDispatchContractError(
            f"technical stop: legal allocation required field {field} is not known and valid"
        )
    value = _finite_float(channel.value, f"technical stop: legal allocation {field}.value")
    age = _finite_float(channel.age, f"technical stop: legal allocation {field}.age")
    if age < 0.0:
        raise PublicDispatchContractError(
            f"technical stop: legal allocation {field}.age must be nonnegative"
        )
    if field in NONNEGATIVE_FIELDS and value < 0.0:
        raise PublicDispatchContractError(
            f"technical stop: legal allocation {field}.value must be nonnegative"
        )
    if field in BOOLEAN_FIELDS and value not in (0.0, 1.0):
        raise PublicDispatchContractError(
            f"technical stop: legal allocation {field}.value must be 0 or 1"
        )
    return value


def _require_uav_ready(row: Mapping[str, _Channel], action: int) -> None:
    values = {
        field: _validate_scoring_channel(row[field], field)
        for field in UAV_REQUIRED_FIELDS
    }
    if values["energy"] <= 0.0:
        raise PublicDispatchContractError(
            f"technical stop: legal allocation action {action} has nonpositive public UAV energy"
        )
    if any(values[field] != 1.0 for field in ("alive", "connected", "idle")):
        raise PublicDispatchContractError(
            f"technical stop: legal allocation action {action} has a public UAV readiness flag != 1"
        )


def _require_task_ready(row: Mapping[str, _Channel], now: float, action: int) -> None:
    values = {
        field: _validate_scoring_channel(row[field], field)
        for field in TASK_REQUIRED_FIELDS
    }
    if values["pending"] != 1.0:
        raise PublicDispatchContractError(
            f"technical stop: legal allocation action {action} has public pending != 1"
        )
    if values["deadline"] <= now:
        raise PublicDispatchContractError(
            f"technical stop: legal allocation action {action} has an expired public deadline"
        )
    if values["remaining_service"] <= 0.0:
        raise PublicDispatchContractError(
            f"technical stop: legal allocation action {action} has nonpositive public remaining service"
        )


def _public_sidecar(obs: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    sidecar = obs.get("public_entity_ids")
    if not isinstance(sidecar, Mapping):
        raise PublicDispatchContractError("public_entity_ids is required")
    raw_uavs = _sequence(sidecar.get("uavs"), "public_entity_ids.uavs")
    raw_tasks = _sequence(sidecar.get("tasks"), "public_entity_ids.tasks")
    if len(raw_uavs) != UAV_COUNT:
        raise PublicDispatchContractError(f"public_entity_ids.uavs must contain {UAV_COUNT} IDs")
    if len(raw_tasks) > TASK_CAPACITY:
        raise PublicDispatchContractError(
            f"public_entity_ids.tasks cannot exceed {TASK_CAPACITY} IDs"
        )
    uavs = tuple(_public_id(value, f"public_entity_ids.uavs[{index}]") for index, value in enumerate(raw_uavs))
    tasks = tuple(_public_id(value, f"public_entity_ids.tasks[{index}]") for index, value in enumerate(raw_tasks))
    if len(set(uavs)) != len(uavs):
        raise PublicDispatchContractError("public UAV identities must be unique")
    if len(set(tasks)) != len(tasks):
        raise PublicDispatchContractError("public task identities must be unique")
    return uavs, tasks


def _validate_observation(obs: Mapping[str, Any]) -> tuple[
    float,
    tuple[dict[str, _Channel], ...],
    tuple[dict[str, _Channel], ...],
    tuple[bool, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[int, ...],
]:
    if not isinstance(obs, Mapping):
        raise PublicDispatchContractError("observation must be a mapping")
    if "time" not in obs:
        raise PublicDispatchContractError("observation.time is required")
    now = _finite_float(obs["time"], "observation.time")
    if now < 0.0:
        raise PublicDispatchContractError("observation.time must be nonnegative")
    uavs, tasks = _public_sidecar(obs)
    decoded_uavs = _decode_rows(obs.get("uavs"), rows=UAV_COUNT, fields=UAV_FIELDS, name="uavs")
    decoded_tasks = _decode_rows(obs.get("tasks"), rows=TASK_CAPACITY, fields=TASK_FIELDS, name="tasks")

    raw_mask = _sequence(obs.get("mask"), "mask")
    if len(raw_mask) != ACTION_COUNT:
        raise PublicDispatchContractError(f"mask must contain exactly {ACTION_COUNT} entries")
    mask = tuple(_binary(value, f"mask[{index}]") for index, value in enumerate(raw_mask))
    legal = tuple(index for index, allowed in enumerate(mask) if allowed)
    if not legal:
        raise PublicDispatchContractError("mask must expose at least one legal action")
    for action in legal:
        if action == NOOP_ACTION:
            continue
        _, task_slot = divmod(action, TASK_CAPACITY)
        if task_slot >= len(tasks):
            raise PublicDispatchContractError(
                f"legal allocation action {action} has no public task identity"
            )

    raw_continuations = _sequence(obs.get("continuation_actions"), "continuation_actions")
    continuations = tuple(
        _integer(value, f"continuation_actions[{index}]")
        for index, value in enumerate(raw_continuations)
    )
    if any(action < 0 or action >= ACTION_COUNT for action in continuations):
        raise PublicDispatchContractError("continuation action outside the native action space")
    for action in continuations:
        if action == NOOP_ACTION:
            continue
        _, task_slot = divmod(action, TASK_CAPACITY)
        if task_slot >= len(tasks):
            raise PublicDispatchContractError(
                f"continuation action {action} has no public task identity"
            )
    return now, decoded_uavs, decoded_tasks, mask, uavs, tasks, continuations


def select_public_dispatch_action(
    obs: Mapping[str, Any],
    *,
    uav_count: int = UAV_COUNT,
    task_capacity: int = TASK_CAPACITY,
) -> PublicDispatchDecision:
    """Rank the public legal actions and apply the existing ACK-known guard.

    The function accepts mappings whose array-like fields support ``len`` and
    integer indexing, including numpy-like arrays, without importing numpy.
    Every source field is read-only.  Masked unknown rows are decoded as public
    placeholders and never scored.  If an original legal allocation does not
    have all required known/valid/ready public fields, the function raises a
    technical-stop contract error instead of silently falling back to NOOP.  A
    legal NOOP receives the next rank weight and is selected only when no
    allocation remains after the mask or the same-task guard.
    """

    if type(uav_count) is not int or uav_count != UAV_COUNT:
        raise PublicDispatchContractError(
            f"public dispatch is frozen to {UAV_COUNT} UAVs"
        )
    if type(task_capacity) is not int or task_capacity != TASK_CAPACITY:
        raise PublicDispatchContractError(
            f"public dispatch is frozen to task capacity {TASK_CAPACITY}"
        )
    now, uav_rows, task_rows, mask, uav_ids, task_ids, continuations = _validate_observation(obs)
    legal = tuple(index for index, allowed in enumerate(mask) if allowed)
    active_allocations: list[tuple[int, tuple[float, ...], str, str]] = []
    for action in legal:
        if action == NOOP_ACTION:
            continue
        uav_index, task_slot = divmod(action, TASK_CAPACITY)
        uav_id, task_id = uav_ids[uav_index], task_ids[task_slot]
        uav = uav_rows[uav_index]
        task = task_rows[task_slot]
        # The frozen view creates a legal allocation only after both rows are
        # complete and ready.  A contradictory legal mask is a technical stop;
        # treating it as a zero-weight candidate would hide a broken contract.
        _require_uav_ready(uav, action)
        _require_task_ready(task, now, action)
        distance = math.hypot(task["x"].value - uav["x"].value, task["y"].value - uav["y"].value)
        if not math.isfinite(distance):
            raise PublicDispatchContractError(f"distance for legal action {action} is non-finite")
        key = (
            -task["priority"].value,
            task["deadline"].value,
            distance,
            -uav["energy"].value,
            float(action),
        )
        active_allocations.append((action, key, uav_id, task_id))

    active_allocations.sort(key=lambda item: item[1])
    noop_legal = NOOP_ACTION in legal
    active_order = tuple(item[0] for item in active_allocations)
    order = active_order + ((NOOP_ACTION,) if noop_legal else ())

    if not active_allocations and not noop_legal:
        raise PublicDispatchContractError(
            "no eligible public allocation remains and legal NOOP is unavailable"
        )

    rank_weights = [0.0] * ACTION_COUNT
    for rank, (action, _key, _uav_id, _task_id) in enumerate(active_allocations):
        # These are only ordering weights, not calibrated probabilities.
        rank_weights[action] = 1.0 / float(rank + 1)
    if noop_legal:
        # NOOP is the next rank after all allocation candidates.  Normalizing
        # this carrier preserves that order and gives guard a finite fallback.
        rank_weights[NOOP_ACTION] = 1.0 / float(len(active_allocations) + 1)
    weight_total = sum(rank_weights[action] for action in legal)
    if not math.isfinite(weight_total) or weight_total <= 0.0:
        raise PublicDispatchContractError("public rank carrier has no finite positive legal weight")
    probabilities = [0.0] * ACTION_COUNT
    for action in legal:
        probabilities[action] = rank_weights[action] / weight_total

    rank_records: list[CandidateRank] = []
    active_by_action = {item[0]: item for item in active_allocations}
    for action in order:
        if action == NOOP_ACTION:
            rank_records.append(CandidateRank(action, None, None, None, True, "legal_noop_fallback"))
        elif action in active_by_action:
            _, key, uav_id, task_id = active_by_action[action]
            rank_records.append(CandidateRank(action, uav_id, task_id, key, True, "ranked_public_pending"))

    # Normalize only the fields consumed by the existing pure guard.  This
    # avoids numpy truth-testing inside the guard while preserving caller input.
    guard_obs = {
        "mask": mask,
        "public_entity_ids": {"uavs": uav_ids, "tasks": task_ids},
        "continuation_actions": continuations,
    }
    try:
        guarded = select_guarded_action(
            guard_obs,
            tuple(probabilities),
            task_capacity=TASK_CAPACITY,
            enabled=True,
        )
    except (PublicIdentityError, NoLegalActionError) as exc:
        raise PublicDispatchContractError(f"public guard contract failure: {exc}") from exc
    return PublicDispatchDecision(
        rule=RULE_NAME,
        selected_action=guarded.final_action,
        original_argmax=guarded.original_action,
        order=order,
        candidate_ranks=tuple(rank_records),
        rank_based_probabilities=tuple(probabilities),
        legal_actions=legal,
        guard_decision=guarded,
    )


# Explicit aliases keep the implementation easy to discover for runners while
# preserving one implementation and one contract.
rank_public_dispatch = select_public_dispatch_action
select_rule_guarded_action = select_public_dispatch_action
select_public_dispatch = select_public_dispatch_action


__all__ = [
    "ACTION_COUNT",
    "CandidateRank",
    "NOOP_ACTION",
    "PublicDispatchContractError",
    "PublicDispatchDecision",
    "RULE_NAME",
    "TASK_CAPACITY",
    "UAV_COUNT",
    "rank_public_dispatch",
    "select_public_dispatch",
    "select_public_dispatch_action",
    "select_rule_guarded_action",
]
