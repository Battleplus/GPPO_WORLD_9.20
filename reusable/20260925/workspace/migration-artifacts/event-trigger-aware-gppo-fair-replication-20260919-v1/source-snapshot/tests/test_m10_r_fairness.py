from __future__ import annotations

import numpy as np
import torch

from gppo_world.m10_environment import M10Config, M10Environment, scenario_from_dict, scenario_tape, scenario_to_dict
from gppo_world.m10_training import M10ActorCritic, M10WorldModel, calibrate_trigger_threshold, collect_world_dataset, train_world_model


def test_seeded_tapes_are_distinct_and_round_trip():
    train = scenario_tape("train", count=3, base_seed=7001)
    validation = scenario_tape("validation", count=3, base_seed=7001)
    assert len({scenario.tape_id for scenario in train + validation}) == 6
    assert [scenario_to_dict(scenario) for scenario in train] == [scenario_to_dict(scenario_from_dict(scenario_to_dict(scenario))) for scenario in train]
    assert any(left.tasks != right.tasks or left.events != right.events for left, right in zip(train, validation))


def test_graph_candidate_scores_follow_task_entity_permutation():
    config = M10Config(uav_count=4, task_capacity=6)
    env = M10Environment(config, scenario_tape("train", count=1, base_seed=9001)[0])
    obs = env.reset()
    model = M10ActorCritic(uav_count=4, task_capacity=6, action_count=config.action_count, encoder="graph", type_count=5, history=False)
    model.eval()
    base = torch.tensor(obs["flat"], dtype=torch.float32)
    swapped = base.clone()
    node_count = config.uav_count + config.region_count + config.target_count + config.task_capacity + config.event_capacity
    task_start = (config.uav_count + config.region_count + config.target_count) * 32
    left = task_start
    right = task_start + 32
    swapped[left:left + 32], swapped[right:right + 32] = base[right:right + 32].clone(), base[left:left + 32].clone()
    relation_start = node_count * 32
    relation = swapped[relation_start:relation_start + config.uav_count * config.task_capacity * config.relation_width].reshape(config.uav_count, config.task_capacity, config.relation_width)
    relation[:, :2, :] = relation[:, [1, 0], :].clone()
    logits_a, _, _ = model(base[None, :])
    logits_b, _, _ = model(swapped[None, :])
    assert torch.allclose(logits_a[0, [0, 1]], logits_b[0, [1, 0]], atol=1e-5)
    assert torch.allclose(logits_a[0, [2, 3]], logits_b[0, [3, 2]], atol=1e-5)


def test_world_model_reports_test_metrics_and_trains_context_head():
    config = M10Config(uav_count=4, task_capacity=6, horizon=5.0)
    train = collect_world_dataset(episodes=2, seed=100, config=config, scenarios=scenario_tape("train", count=2, base_seed=100), split="train")
    validation = collect_world_dataset(episodes=1, seed=100, config=config, scenarios=scenario_tape("validation", count=1, base_seed=100), split="validation")
    test = collect_world_dataset(episodes=1, seed=100, config=config, scenarios=scenario_tape("test", count=1, base_seed=100), split="test")
    model, metadata = train_world_model(train + validation + test, seed=100, action_count=config.action_count, epochs=2, device="cpu")
    assert metadata["test_rows"] == len(test)
    assert set(("reward_rmse", "event_bce", "done_bce", "context_rmse")) <= metadata["test_metrics"].keys()
    assert any(parameter.grad is not None for name, parameter in model.named_parameters() if "context_head" in name)
    calibration = calibrate_trigger_threshold(model, validation, device="cpu")
    assert calibration["selection_unit"] == "validation_rows_only"


def test_world_context_requires_explicit_legal_action_mask_and_ood_fallback():
    model = M10WorldModel(obs_dim=4, action_count=3)
    obs = torch.zeros((1, 4))
    try:
        model.context_for(obs)
    except ValueError:
        pass
    else:
        raise AssertionError("action-agnostic context must not silently use action 0")
    context, active, risk = model.context_for(torch.ones_like(obs) * 100.0, action_mask=torch.tensor([True, False, False]))
    assert context.shape == (1, 8)
    assert not active and risk == 1.0
