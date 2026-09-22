"""Pure public-state guard for avoiding duplicate task submissions.

The guard is deliberately independent of the simulator and model.  It maps
public action slots to public task identities, removes allocation actions for
tasks represented by ACK-known continuation handles, and selects the highest
original probability among the remaining legal actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class PublicIdentityError(ValueError):
    """The public identity sidecar cannot support a safe action mapping."""


class NoLegalActionError(ValueError):
    """The public contract exposes no legal action after guarding."""


@dataclass(frozen=True)
class GuardDecision:
    original_action: int
    final_action: int
    triggered: bool
    reason: str
    continuation_task_ids: tuple[str, ...]
    excluded_actions: tuple[int, ...]
    candidate_actions: tuple[int, ...]
    action_task_ids: tuple[tuple[int, str], ...]


def _public_ids(obs: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    sidecar = obs.get("public_entity_ids")
    if not isinstance(sidecar, Mapping):
        raise PublicIdentityError("public_entity_ids is missing")
    uavs = tuple(str(value) for value in (sidecar.get("uavs") or ()))
    tasks = tuple(str(value) for value in (sidecar.get("tasks") or ()))
    if not uavs or len(set(uavs)) != len(uavs):
        raise PublicIdentityError("public UAV identities are missing or duplicated")
    if len(set(tasks)) != len(tasks):
        raise PublicIdentityError("public task identities are duplicated")
    return uavs, tasks


def action_target(obs: Mapping[str, Any], action: int, *, task_capacity: int) -> tuple[str, str] | None:
    """Resolve one action using only the public identity sidecar.

    The final action slot is NOOP.  Allocation slots beyond the currently
    delivered public task list are treated as an identity-contract error only
    when they are legal in the public mask; an illegal slot is not revived by
    this helper.
    """
    mask = tuple(bool(value) for value in obs.get("mask", ()))
    uavs, tasks = _public_ids(obs)
    if task_capacity <= 0 or len(mask) != len(uavs) * task_capacity + 1:
        raise PublicIdentityError("action dimensions do not match public UAV/task contract")
    action = int(action)
    noop = len(mask) - 1
    if action == noop:
        return None
    if action < 0 or action >= noop:
        raise PublicIdentityError(f"action outside public action space: {action}")
    uav_index, task_slot = divmod(action, task_capacity)
    if uav_index >= len(uavs):
        raise PublicIdentityError(f"UAV slot missing for action {action}")
    if task_slot >= len(tasks):
        if mask[action]:
            raise PublicIdentityError(f"legal action {action} has no public task identity")
        return None
    return uavs[uav_index], tasks[task_slot]


def select_guarded_action(
    obs: Mapping[str, Any],
    probabilities: Sequence[float],
    *,
    task_capacity: int,
    enabled: bool = True,
) -> GuardDecision:
    """Select an action after applying the ACK-known task guard.

    ``probabilities`` are read-only model output.  This function never edits
    the observation, mask, probability vector, RNG, or model state.
    """
    mask = tuple(bool(value) for value in obs.get("mask", ()))
    if not mask or len(probabilities) != len(mask):
        raise PublicIdentityError("probability and public mask dimensions differ")
    legal = tuple(index for index, allowed in enumerate(mask) if allowed)
    if not legal:
        raise NoLegalActionError("public mask has no legal action")
    finite = [float(value) for value in probabilities]
    if any(value != value or value in (float("inf"), float("-inf")) for value in finite):
        raise PublicIdentityError("non-finite action probability")
    original_action = max(legal, key=lambda index: (finite[index], -index))
    if not enabled:
        return GuardDecision(original_action, original_action, False, "disabled", (), (), legal, ())

    uavs, tasks = _public_ids(obs)
    noop = len(mask) - 1
    continuation_actions = tuple(int(value) for value in (obs.get("continuation_actions") or ()))
    continuation_tasks: set[str] = set()
    for action in continuation_actions:
        if action < 0 or action >= len(mask):
            raise PublicIdentityError(f"continuation action outside public action space: {action}")
        if action != noop:
            _, task_slot = divmod(action, task_capacity)
            if task_slot >= len(tasks):
                raise PublicIdentityError(f"continuation action {action} has no public task identity")
        target = action_target(obs, action, task_capacity=task_capacity)
        if target is not None:
            continuation_tasks.add(target[1])

    action_task_ids: list[tuple[int, str]] = []
    excluded: list[int] = []
    allocations: list[int] = []
    for action in legal:
        if action == noop:
            continue
        target = action_target(obs, action, task_capacity=task_capacity)
        if target is None:
            raise PublicIdentityError(f"legal allocation action {action} has no public target")
        allocations.append(action)
        action_task_ids.append((action, target[1]))
        if target[1] in continuation_tasks:
            excluded.append(action)

    allocation_candidates = [action for action in allocations if action not in set(excluded)]
    candidates = list(allocation_candidates)
    if noop in legal:
        # NOOP has no task identity and remains a legal original-mask choice.
        candidates.append(noop)
    if candidates:
        final_action = max(candidates, key=lambda index: (finite[index], -index))
        if not allocation_candidates and noop in legal:
            reason = "no_remaining_allocation_choose_noop"
        else:
            reason = "excluded_continuation_task" if excluded else "no_guard_match"
    else:
        raise NoLegalActionError("all legal allocation actions were excluded and NOOP is illegal")
    return GuardDecision(
        original_action,
        final_action,
        final_action != original_action,
        reason,
        tuple(sorted(continuation_tasks)),
        tuple(excluded),
        tuple(candidates if candidates else (noop,)),
        tuple(action_task_ids),
    )


__all__ = [
    "GuardDecision",
    "NoLegalActionError",
    "PublicIdentityError",
    "action_target",
    "select_guarded_action",
]
