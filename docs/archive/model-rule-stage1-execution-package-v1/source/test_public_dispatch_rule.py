from __future__ import annotations

import copy
import math

import pytest

from gppo_world.public_dispatch_rule_v1 import (
    ACTION_COUNT,
    NOOP_ACTION,
    PublicDispatchContractError,
    RULE_NAME,
    select_public_dispatch_action,
)


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


class ArrayLike:
    """Small numpy-like fixture without importing numpy in this test module."""

    def __init__(self, values):
        self._values = values

    def __len__(self):
        return len(self._values)

    def __getitem__(self, index):
        return self._values[index]


def channel(value, *, known=True, valid=True, age=0.0):
    return [value, int(known), int(valid), age]


def uav_row(*, x=0.0, y=0.0, energy=10.0, known=True, valid=True):
    values = {
        "x": x,
        "y": y,
        "energy": energy,
        "alive": 1.0,
        "connected": 1.0,
        "idle": 1.0,
    }
    return [item for field in UAV_FIELDS for item in channel(values[field], known=known, valid=valid)]


def task_row(*, x=0.0, y=0.0, deadline=30.0, priority=1.0, pending=1.0,
             known=True, valid=True, remaining_service=1.0):
    values = {
        "x": x,
        "y": y,
        "deadline": deadline,
        "remaining_service": remaining_service,
        "priority": priority,
        "pending": pending,
        "region_id": 0.0,
        "target_id": 0.0,
    }
    return [item for field in TASK_FIELDS for item in channel(values[field], known=known, valid=valid)]


def unknown_row(fields):
    """Masked placeholder row; native missing age is exactly zero."""

    return [item for _field in fields for item in channel(-1.0, known=False, valid=False, age=0.0)]


def make_observation(*, mask=None, uavs=None, tasks=None, task_ids=None, continuation=(), time=10.0):
    if uavs is None:
        uavs = [uav_row(x=float(index), y=0.0, energy=10.0) for index in range(4)]
    if tasks is None:
        tasks = [task_row(x=float(index), y=0.0, deadline=30.0, priority=1.0) for index in range(6)]
    if task_ids is None:
        task_ids = tuple(f"task-{index}" for index in range(6))
    if mask is None:
        mask = [True] * ACTION_COUNT
    return {
        "time": time,
        "uavs": uavs,
        "tasks": tasks,
        "mask": mask,
        "public_entity_ids": {
            "uavs": ("uav-0", "uav-1", "uav-2", "uav-3"),
            "tasks": task_ids,
        },
        "continuation_actions": continuation,
    }


def legal_mask(*actions):
    mask = [False] * ACTION_COUNT
    for action in actions:
        mask[action] = True
    return mask


def test_two_public_scenes_change_the_rule_choice_and_leave_input_unchanged():
    scene_a = make_observation(
        mask=legal_mask(0, 1, NOOP_ACTION),
        tasks=[task_row(priority=8.0), task_row(priority=2.0)]
        + [task_row(priority=0.0) for _ in range(4)],
    )
    scene_b = copy.deepcopy(scene_a)
    scene_b["tasks"][0] = task_row(priority=2.0)
    scene_b["tasks"][1] = task_row(priority=8.0)
    before_a = copy.deepcopy(scene_a)
    before_b = copy.deepcopy(scene_b)

    decision_a = select_public_dispatch_action(scene_a)
    decision_b = select_public_dispatch_action(scene_b)

    assert decision_a.selected_action == 0
    assert decision_b.selected_action == 1
    assert decision_a.original_argmax == 0
    assert decision_b.original_argmax == 1
    assert decision_a.rule == decision_b.rule == RULE_NAME
    assert scene_a == before_a
    assert scene_b == before_b


@pytest.mark.parametrize(
    ("tasks", "uavs", "expected"),
    [
        # Priority descending.
        (
            [task_row(priority=9.0), task_row(priority=8.0)],
            None,
            0,
        ),
        # Earlier deadline after priority ties.
        (
            [task_row(priority=4.0, deadline=20.0), task_row(priority=4.0, deadline=15.0)],
            None,
            1,
        ),
        # Shorter two-dimensional distance after the first two ties.
        (
            [task_row(priority=4.0, deadline=20.0, x=9.0, y=0.0), task_row(priority=4.0, deadline=20.0, x=1.0, y=0.0)],
            [uav_row(x=0.0, y=0.0) for _ in range(4)],
            1,
        ),
        # Public energy descending after distance ties.
        (
            [task_row(priority=4.0, deadline=20.0, x=0.0, y=0.0)] * 2,
            [uav_row(x=0.0, y=0.0, energy=2.0), uav_row(x=0.0, y=0.0, energy=9.0)]
            + [uav_row(x=50.0, y=50.0) for _ in range(2)],
            6,
        ),
        # Complete tie goes to the lower native action id.
        (
            [task_row(priority=4.0, deadline=20.0, x=0.0, y=0.0)] * 2,
            [uav_row(x=0.0, y=0.0, energy=9.0) for _ in range(4)],
            0,
        ),
    ],
)
def test_priority_edf_nearest_energy_and_action_tie_order(tasks, uavs, expected):
    observation = make_observation(
        mask=legal_mask(0, 1, 6, NOOP_ACTION),
        tasks=tasks + [task_row(priority=0.0) for _ in range(6 - len(tasks))],
        uavs=uavs,
    )
    decision = select_public_dispatch_action(observation)
    assert decision.selected_action == expected
    assert decision.order[0] == expected
    assert decision.rank_based_probabilities[expected] > decision.rank_based_probabilities[NOOP_ACTION]


def test_legal_mask_with_insufficient_known_valid_fields_is_a_technical_stop():
    observation = make_observation(
        mask=legal_mask(0, NOOP_ACTION),
        tasks=[task_row(priority=99.0, known=False, valid=False)]
        + [task_row(priority=0.0) for _ in range(5)],
    )
    with pytest.raises(PublicDispatchContractError, match="technical stop.*required field"):
        select_public_dispatch_action(observation)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda obs: obs["tasks"][0].__setitem__(20, 0.0),  # pending value
        lambda obs: obs["tasks"][0].__setitem__(8, 10.0),  # expired deadline
        lambda obs: obs["tasks"][0].__setitem__(12, 0.0),  # no remaining service
        lambda obs: obs["uavs"][0].__setitem__(8, 0.0),  # nonpositive energy
        lambda obs: obs["uavs"][0].__setitem__(12, 0.0),  # alive readiness
        lambda obs: obs["uavs"][0].__setitem__(10, 0.0),  # energy valid flag
    ],
)
def test_legal_mask_ready_mismatch_is_never_silently_replaced_by_noop(mutator):
    observation = make_observation(mask=legal_mask(0, NOOP_ACTION), task_ids=("task-0",))
    mutator(observation)
    with pytest.raises(PublicDispatchContractError):
        select_public_dispatch_action(observation)


def test_all_noop_keeps_the_original_legal_set():
    observation = make_observation(mask=legal_mask(NOOP_ACTION), task_ids=())
    before = copy.deepcopy(observation)
    decision = select_public_dispatch_action(observation)
    assert decision.selected_action == NOOP_ACTION
    assert decision.original_argmax == NOOP_ACTION
    assert decision.order == (NOOP_ACTION,)
    assert decision.legal_actions == (NOOP_ACTION,)
    assert decision.candidate_actions == (NOOP_ACTION,)
    assert observation == before


def test_guard_excludes_same_task_across_uavs_and_noop_remains_legal():
    observation = make_observation(
        mask=legal_mask(0, 6, NOOP_ACTION),
        task_ids=("task-0",),
        continuation=(6,),
    )
    decision = select_public_dispatch_action(observation)
    assert decision.original_argmax == 0
    assert decision.selected_action == NOOP_ACTION
    assert set(decision.excluded_actions) == {0, 6}
    assert decision.candidate_actions == (NOOP_ACTION,)
    assert NOOP_ACTION in decision.legal_actions
    assert NOOP_ACTION in decision.order


def test_rule_does_not_force_noop_when_a_public_allocation_is_eligible():
    observation = make_observation(mask=legal_mask(0, NOOP_ACTION), task_ids=("task-0",))
    decision = select_public_dispatch_action(observation)
    assert decision.selected_action == 0
    assert decision.original_argmax == 0
    assert decision.legal_actions == (0, NOOP_ACTION)
    assert decision.order == (0, NOOP_ACTION)
    assert decision.rank_based_probabilities[0] > decision.rank_based_probabilities[NOOP_ACTION]
    assert decision.candidate_actions == (0, NOOP_ACTION)


def test_native_task_policy_view_shape_accepts_masked_unknown_padding():
    observation = make_observation(
        mask=legal_mask(0, NOOP_ACTION),
        uavs=[uav_row(x=0.0, y=0.0)] + [unknown_row(UAV_FIELDS) for _ in range(3)],
        tasks=[task_row(x=1.0, y=2.0)] + [unknown_row(TASK_FIELDS) for _ in range(5)],
        task_ids=("task-0",),
    )
    # region_id/target_id are not scoring fields and may remain masked in a
    # row whose required fields form a legal candidate.
    observation["tasks"][0][24] = -1.0
    observation["tasks"][0][28] = -1.0
    before = copy.deepcopy(observation)
    decision = select_public_dispatch_action(observation)
    assert decision.selected_action == 0
    assert len(observation["uavs"]) == 4
    assert len(observation["tasks"]) == 6
    assert observation == before


def test_masked_unknown_value_placeholders_are_not_scored_or_rejected():
    observation = make_observation(
        mask=legal_mask(0, NOOP_ACTION),
        uavs=[uav_row(x=0.0, y=0.0)] + [unknown_row(UAV_FIELDS) for _ in range(3)],
        tasks=[task_row(x=1.0, y=2.0)] + [unknown_row(TASK_FIELDS) for _ in range(5)],
        task_ids=("task-0",),
    )
    # The native missing representation is zero, but a masked transport may
    # carry nonfinite placeholders.  These rows are not scoring inputs.
    observation["uavs"][1][0] = math.nan
    observation["tasks"][1][0] = math.nan
    observation["tasks"][1][3] = math.nan
    decision = select_public_dispatch_action(observation)
    assert decision.selected_action == 0


def test_rank_carrier_is_normalized_over_original_legal_actions():
    observation = make_observation(mask=legal_mask(0, 1, NOOP_ACTION), task_ids=("task-0", "task-1"))
    decision = select_public_dispatch_action(observation)
    assert sum(decision.rank_based_probabilities) == pytest.approx(1.0)
    assert decision.rank_based_probabilities[0] > decision.rank_based_probabilities[1]
    assert decision.rank_based_probabilities[1] > decision.rank_based_probabilities[NOOP_ACTION]
    assert decision.rank_based_probabilities[NOOP_ACTION] > 0.0


def test_array_like_public_fields_are_read_without_mutation():
    observation = make_observation(mask=legal_mask(0, NOOP_ACTION), task_ids=("task-0",))
    observation["uavs"] = ArrayLike([ArrayLike(row) for row in observation["uavs"]])
    observation["tasks"] = ArrayLike([ArrayLike(row) for row in observation["tasks"]])
    observation["mask"] = ArrayLike(observation["mask"])
    observation["continuation_actions"] = ArrayLike([])
    before = copy.deepcopy(observation)
    decision = select_public_dispatch_action(observation)
    assert decision.selected_action == 0
    assert observation["mask"][:2] == before["mask"][:2]
    assert len(observation["uavs"]) == len(before["uavs"]) == 4


@pytest.mark.parametrize(
    "mutator",
    [
        lambda obs: obs.update(mask=[False] * (ACTION_COUNT - 1)),
        lambda obs: obs["mask"].__setitem__(0, 2),
        lambda obs: obs["tasks"][0].__setitem__(0, math.nan),
        lambda obs: obs["uavs"][0].__setitem__(2, math.inf),
        lambda obs: obs.update(time=math.inf),
        lambda obs: obs.update(continuation_actions=(ACTION_COUNT,)),
    ],
)
def test_malformed_mask_fields_and_nonfinite_values_raise_contract_error(mutator):
    observation = make_observation(mask=legal_mask(0, NOOP_ACTION), task_ids=("task-0",))
    mutator(observation)
    with pytest.raises(PublicDispatchContractError):
        select_public_dispatch_action(observation)


def test_legal_allocation_without_public_task_identity_is_rejected():
    observation = make_observation(mask=legal_mask(0, NOOP_ACTION), task_ids=())
    with pytest.raises(PublicDispatchContractError):
        select_public_dispatch_action(observation)


def test_rank_carrier_is_audit_only_and_guard_result_is_exposed():
    observation = make_observation(mask=legal_mask(0, 1, NOOP_ACTION), task_ids=("task-0", "task-1"))
    decision = select_public_dispatch_action(observation)
    assert len(decision.rank_based_probabilities) == ACTION_COUNT
    assert decision.probabilities == decision.rank_based_probabilities
    assert decision.final_action == decision.selected_action == decision.guard_decision.final_action
    assert [action for action, _key in decision.rank_keys] == list(decision.order)
    assert decision.to_dict()["rank_based_probabilities"] == list(decision.rank_based_probabilities)


def test_only_the_frozen_native_dimensions_are_accepted():
    observation = make_observation(mask=legal_mask(0, NOOP_ACTION), task_ids=("task-0",))
    assert select_public_dispatch_action(observation, uav_count=4, task_capacity=6).selected_action == 0
    with pytest.raises(PublicDispatchContractError):
        select_public_dispatch_action(observation, task_capacity=5)
