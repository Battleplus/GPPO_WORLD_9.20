from __future__ import annotations

import copy

import numpy as np
import torch

from gppo_world.joint_gppo import (
    ActionConditionedTemporalWorldModel,
    JointGraphPreferencePolicy,
    JointTrainConfig,
    build_observed_event_labels,
)
from gppo_world.joint_training import (
    GROUPS,
    _make_optimizers,
    _obs_tensor,
    _relation_for_action,
    _zero_hidden,
    _zero_wm_hidden,
    world_model_updates,
)
from gppo_world.m10_environment import M10Config, M10Environment, default_scenario


def _state_delta(before, after):
    return max(
        float((after[name] - before[name]).abs().max())
        for name in before
        if torch.is_tensor(before[name]) and before[name].numel()
    )


def _one_transition(policy, env_config):
    env = M10Environment(env_config, default_scenario("mixed", seed=9101, split="gradient-test"))
    obs = env.reset()
    action = int(np.flatnonzero(obs["mask"])[0])
    obs_tensor = _obs_tensor(obs, torch.device("cpu"))
    hidden = _zero_hidden(torch.device("cpu"))
    with torch.no_grad():
        _, _, hidden_after = policy.encode(obs_tensor, hidden)
    next_obs, reward, _, info = env.step(action)
    event = build_observed_event_labels(obs, next_obs, initial_energy=env_config.initial_energy)
    return {
        "obs": np.asarray(obs["flat"], dtype=np.float32),
        "next_obs": np.asarray(next_obs["flat"], dtype=np.float32),
        "observation_dict": obs,
        "next_observation_dict": next_obs,
        "mask": np.asarray(obs["mask"], dtype=np.bool_),
        "action": action,
        "old_values": np.zeros(2, dtype=np.float32),
        "policy_hidden_before": hidden[0, 0].numpy(),
        "policy_hidden_after": hidden_after[0, 0].numpy(),
        "world_hidden_before": _zero_wm_hidden(torch.device("cpu"))[0].numpy(),
        "candidate_features": np.zeros((25, 17), dtype=np.float32),
        "preference": np.asarray((0.5, 0.5), dtype=np.float32),
        "vector_reward": np.asarray((float(reward), 0.0), dtype=np.float32),
        "task_consequence": np.zeros(2, dtype=np.float32),
        "event_label": event["labels"],
        "event_mask": event["mask"],
        "state_target_valid": True,
        "vector_reward_valid": True,
        "task_consequence_valid": True,
    }


def test_isolated_world_optimizer_excludes_shared_encoder():
    config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    train = JointTrainConfig(seed=1101)
    for group, expected in (("CS", True), ("CI", False), ("DI", False)):
        policy = JointGraphPreferencePolicy(config, history=True)
        world = ActionConditionedTemporalWorldModel()
        _, optimizer = _make_optimizers(policy, world, GROUPS[group], train)
        names = {id(parameter): name for name, parameter in policy.named_parameters()}
        has_shared = any(names.get(id(parameter), "").startswith("base.")
                         for block in optimizer.param_groups for parameter in block["params"])
        assert has_shared is expected


def test_isolated_world_update_keeps_policy_encoder_bitwise_fixed_and_updates_world():
    torch.manual_seed(1101)
    config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    train = JointTrainConfig(seed=1101, wm_minibatch_size=1)
    policy = JointGraphPreferencePolicy(config, history=True)
    world = ActionConditionedTemporalWorldModel()
    _, optimizer = _make_optimizers(policy, world, GROUPS["CI"], train)
    transition = _one_transition(policy, config)
    policy_before = copy.deepcopy(policy.state_dict())
    world_before = copy.deepcopy(world.state_dict())
    rows = world_model_updates(
        policy, world, [transition], optimizer, train, torch.device("cpu"),
        updates=1, event_enabled=False, isolate_policy_gradient=True,
    )
    assert rows[0]["world_gradient_isolated"] is True
    assert rows[0]["shared_encoder_parameter_delta"] == 0.0
    assert _state_delta(world_before, world.state_dict()) > 0.0
    assert _state_delta(policy_before, policy.state_dict()) == 0.0


def test_isolated_world_update_replays_identically_from_same_cpu_boundary():
    torch.manual_seed(1101)
    config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    train = JointTrainConfig(seed=1101, wm_minibatch_size=1)
    policy_u = JointGraphPreferencePolicy(config, history=True)
    world_u = ActionConditionedTemporalWorldModel()
    transition = _one_transition(policy_u, config)
    policy_r = JointGraphPreferencePolicy(config, history=True)
    world_r = ActionConditionedTemporalWorldModel()
    policy_r.load_state_dict(copy.deepcopy(policy_u.state_dict()), strict=True)
    world_r.load_state_dict(copy.deepcopy(world_u.state_dict()), strict=True)
    _, optimizer_u = _make_optimizers(policy_u, world_u, GROUPS["CI"], train)
    _, optimizer_r = _make_optimizers(policy_r, world_r, GROUPS["CI"], train)
    world_model_updates(
        policy_u, world_u, [transition], optimizer_u, train, torch.device("cpu"),
        updates=1, event_enabled=False, batch_rng=np.random.default_rng(401),
        isolate_policy_gradient=True,
    )
    world_model_updates(
        policy_r, world_r, [transition], optimizer_r, train, torch.device("cpu"),
        updates=1, event_enabled=False, batch_rng=np.random.default_rng(401),
        isolate_policy_gradient=True,
    )
    assert _state_delta(world_u.state_dict(), world_r.state_dict()) == 0.0
    assert _state_delta(policy_u.state_dict(), policy_r.state_dict()) == 0.0
    for key, value in optimizer_u.state_dict()["state"].items():
        other = optimizer_r.state_dict()["state"][key]
        for state_key in value:
            left, right = value[state_key], other[state_key]
            assert torch.equal(left, right) if torch.is_tensor(left) else left == right
