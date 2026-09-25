import pytest
import torch

from gppo_world.graph5 import Graph5Snapshot
from gppo_world.joint_consequence import build_group_targets, build_joint_target_record, validate_candidate_group
from gppo_world.joint_consequence_baseline import public_joint_action_score, select_public_joint_action


def graph(mask=(True, True) + (False,) * 22 + (True,)):
    nodes = {name: torch.zeros((count, 32)) for name, count in (("uav", 4), ("region", 3), ("target", 4), ("task", 6), ("event", 4))}
    candidate = torch.zeros((24, 4))
    candidate[0, 0] = 0.1
    candidate[0, 1] = 1.0
    candidate[1, 0] = 0.5
    candidate[1, 1] = 1.0
    return Graph5Snapshot(nodes, candidate, torch.tensor(mask), torch.zeros(27), 1).as_dict()


def branch(action, outcomes, *, parent="p0", prefix="p0:x"):
    return {"parent_episode_id": parent, "prefix_id": prefix, "graph5_t": graph(), "target": {"action": action}, "joint_task_set": ["task-0", "task-1"], "joint_task_outcomes": outcomes, "exogenous_key": "shared-key"}


def outcomes(*, censored=False, first=True):
    return [
        {"task_id": "task-0", "deadline_observed": not censored, "arrival_before_deadline_physical": None if censored else first},
        {"task_id": "task-1", "deadline_observed": True, "arrival_before_deadline_physical": not first},
    ]


def test_joint_target_preserves_action_opportunity_cost_delta():
    ref = branch(0, [
        {"task_id": "task-0", "deadline_observed": True, "arrival_before_deadline_physical": False},
        {"task_id": "task-1", "deadline_observed": True, "arrival_before_deadline_physical": False},
    ])
    candidate = branch(1, outcomes(first=True))
    result = build_joint_target_record(candidate, ref, observation_window_steps=6)
    assert result["target"]["new_on_time_count"] == 1
    assert result["target"]["deadline_failure_count"] == 1
    assert result["target"]["delta_new_on_time_count_vs_reference"] == 1
    assert result["target"]["delta_deadline_failure_count_vs_reference"] == -1
    assert result["label_masks"]["delta_vs_reference"] is True
    assert result["target"]["label_source"] == "simulator-counterfactual"
    assert result["joint_task_set"] == ["task-0", "task-1"]
    assert [item["task_id"] for item in result["joint_task_outcomes"]] == ["task-0", "task-1"]
    assert result["exogenous_key"] == "shared-key"


def test_censoring_masks_aggregate_and_delta_instead_of_defaulting_to_failure():
    ref = branch(0, outcomes(first=False))
    candidate = branch(1, outcomes(censored=True))
    result = build_joint_target_record(candidate, ref, observation_window_steps=6)
    assert result["target"]["new_on_time_count"] is None
    assert result["target"]["deadline_failure_count"] is None
    assert result["target"]["delta_new_on_time_count_vs_reference"] is None
    assert result["censoring"]["candidate_task_ids"] == ["task-0"]


def test_group_requires_fixed_task_set_and_unique_legal_actions():
    ref = branch(0, outcomes(first=False))
    bad = branch(1, outcomes(first=True))
    bad["joint_task_set"] = ["task-0"]
    with pytest.raises(ValueError, match="joint_task_set"):
        validate_candidate_group([ref, bad])
    duplicate = branch(0, outcomes(first=True))
    with pytest.raises(ValueError, match="duplicate candidate"):
        validate_candidate_group([ref, duplicate])


def test_group_adapter_requires_reference_branch():
    with pytest.raises(ValueError, match="reference_action"):
        build_group_targets([branch(0, outcomes(first=False)), branch(1, outcomes(first=True))], reference_action=2, observation_window_steps=6)


def test_group_adapter_keeps_reference_and_repeat_identity():
    reference = branch(0, outcomes(first=False))
    candidate = branch(1, outcomes(first=True))
    reference["repeat_id"] = candidate["repeat_id"] = "2"
    rows = build_group_targets([reference, candidate], reference_action=0, observation_window_steps=6)
    assert [row["target"]["action"] for row in rows] == [0, 1]
    assert all(row["repeat_id"] == "2" for row in rows)
    assert rows[0]["target"]["delta_new_on_time_count_vs_reference"] == 0


def test_public_joint_baseline_uses_candidate_and_opportunity_terms():
    nodes = {name: torch.zeros((count, 32)) for name, count in (("uav", 4), ("region", 3), ("target", 4), ("task", 6), ("event", 4))}
    for row in nodes["uav"]:
        for index, value in enumerate((0.0, 0.0, 9.0, 1.0, 1.0, 1.0)):
            row[index * 4:index * 4 + 3] = torch.tensor((value, 1.0, 1.0))
    for row in nodes["task"][:2]:
        for index, value in enumerate((1.0, 1.0, 10.0, 0.0, 1.0, 1.0)):
            row[index * 4:index * 4 + 3] = torch.tensor((value, 1.0, 1.0))
    snapshot = Graph5Snapshot(
        nodes=nodes,
        candidate_features=torch.tensor([[0.1, 1.0, 1.0, 0.0], [0.5, 1.0, 1.0, 0.0]] + [[0.0] * 4] * 22),
        action_mask=torch.tensor((True, True) + (False,) * 22 + (True,)),
        global_features=torch.zeros(27),
        graph_version=1,
    )
    score = public_joint_action_score(snapshot, 0)
    assert score["estimated_arrival"] < public_joint_action_score(snapshot, 1)["estimated_arrival"]
    assert select_public_joint_action(snapshot) == 0
