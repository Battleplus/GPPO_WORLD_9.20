from __future__ import annotations

import torch

from gppo_world.m10_environment import M10Config, M10Environment, M10Scenario, M10TaskSpec
from gppo_world.m10_training import M10ActorCritic, M10WorldModel, PPOConfig, collect_rollout, evaluate_policy


def _fixture():
    config = M10Config(uav_count=1, task_capacity=1, horizon=20.0)
    scenario = M10Scenario(
        "triggered-context-continuation",
        (M10TaskSpec("task-0", 0.0, 0.0, 0.0, 20.0, 8.0, 1.0),),
        (),
    )
    env = M10Environment(config, scenario)
    world = M10WorldModel(len(env.reset()["flat"]), config.action_count)
    policy = M10ActorCritic(
        uav_count=config.uav_count,
        task_capacity=config.task_capacity,
        action_count=config.action_count,
        encoder="graph",
        type_count=5,
        history=True,
        context_dim=world.context_dim,
        region_count=config.region_count,
        target_count=config.target_count,
        event_capacity=config.event_capacity,
        relation_width=config.relation_width,
    )
    return config, scenario, world, policy


def test_triggered_rollout_uses_context_sized_critic_on_continuation():
    config, scenario, world, policy = _fixture()
    transitions = collect_rollout(
        policy,
        config=PPOConfig(rollout_steps=4, update_epochs=1),
        env_config=config,
        seed=1101,
        device=torch.device("cpu"),
        model=world,
        fusion="triggered",
        trigger_threshold=1.1,
        scenarios=[scenario],
        max_replan_interval=99,
    )
    assert any(not transition.actor_decision for transition in transitions)
    assert all(len(transition.obs) == policy.input_dim for transition in transitions)
    assert all(len(transition.context) == world.context_dim for transition in transitions)


def test_triggered_evaluation_keeps_context_sized_critic_on_continuation():
    config, scenario, world, policy = _fixture()
    result = evaluate_policy(
        policy,
        model=world,
        fusion="triggered",
        env_config=config,
        seeds=[1101],
        device="cpu",
        trigger_threshold=1.1,
        scenarios={1101: scenario},
        max_replan_interval=99,
    )
    record = result["episodes"][0]
    assert record["continuation_steps"] > 0
