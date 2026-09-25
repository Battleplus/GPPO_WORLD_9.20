from __future__ import annotations

from unittest.mock import patch

import numpy as np
import torch

from gppo_world.m10_environment import M10Config, M10Environment, M10Scenario, M10TaskSpec
from gppo_world.service_clock import ServiceEvent
from gppo_world.m10_training import (
    M10ActorCritic,
    PPOConfig,
    Transition,
    _evaluate_sequence,
    _gae,
    _trigger_decision,
    collect_rollout,
)


def _transition(*, value: float, reward: float, terminated: bool = False,
                truncated: bool = False, next_value: float = 0.0,
                episode_start: bool = False, actor_decision: bool = True,
                obs_dim: int = 10) -> Transition:
    return Transition(
        obs=np.zeros(obs_dim, dtype=np.float32), context=np.zeros(0, dtype=np.float32),
        mask=np.ones(2, dtype=np.bool_), action=0, log_prob=-0.1, value=value,
        reward=reward, done=terminated or truncated, episode_start=episode_start,
        info={}, terminated=terminated, truncated=truncated, next_value=next_value,
        actor_decision=actor_decision,
    )


def test_gae_bootstraps_rollout_cutoff_but_resets_at_true_terminal():
    transitions = [
        _transition(value=1.0, reward=1.0, next_value=2.0),
        _transition(value=2.0, reward=0.0, next_value=3.0),
        _transition(value=3.0, reward=4.0, truncated=True, next_value=5.0),
    ]
    advantages, returns = _gae(transitions, gamma=0.9, gae_lambda=0.8)
    np.testing.assert_allclose(advantages.numpy(), [5.1552, 4.66, 5.5], rtol=1e-6)
    np.testing.assert_allclose(returns.numpy(), [6.1552, 6.66, 8.5], rtol=1e-6)

    terminal_middle = [
        _transition(value=1.0, reward=1.0, next_value=2.0),
        _transition(value=2.0, reward=10.0, terminated=True, next_value=99.0),
        _transition(value=3.0, reward=4.0, truncated=True, next_value=5.0),
    ]
    advantages, returns = _gae(terminal_middle, gamma=0.9, gae_lambda=0.8)
    np.testing.assert_allclose(advantages.numpy(), [7.56, 8.0, 5.5], rtol=1e-6)
    np.testing.assert_allclose(returns.numpy(), [8.56, 10.0, 8.5], rtol=1e-6)


def test_history_hidden_state_resets_at_episode_boundary():
    torch.manual_seed(2)
    policy = M10ActorCritic(
        uav_count=1, task_capacity=1, action_count=2, encoder="mlp",
        type_count=2, history=True,
    ).eval()
    first = _transition(value=0.0, reward=0.0, episode_start=True, obs_dim=policy.base_obs_dim)
    second = _transition(value=0.0, reward=0.0, episode_start=True, obs_dim=policy.base_obs_dim)
    sequence_logp, sequence_values, _ = _evaluate_sequence(policy, [first, second], torch.device("cpu"))
    isolated_logp, isolated_values, _ = _evaluate_sequence(policy, [second], torch.device("cpu"))
    torch.testing.assert_close(sequence_logp[1], isolated_logp[0])
    torch.testing.assert_close(sequence_values[1], isolated_values[0])


def test_trigger_priority_distinguishes_risk_wait_safety_and_noop():
    base = {"mask": np.asarray([True, True]), "trigger_flags": {}}
    decision, reason, conditions = _trigger_decision(
        base, fusion="triggered", risk_active=False, last_action=None,
        steps_since_replan=0, max_replan_interval=3,
    )
    assert decision and reason == "initial" and conditions["initial"]

    for flags, expected in (
        ({}, "none"),
        ({"confirmed_fault": True}, "confirmed_fault"),
        ({"task_arrival": True}, "task_arrival"),
    ):
        obs = {"mask": np.asarray([True, True]), "trigger_flags": flags}
        decision, reason, _ = _trigger_decision(
            obs, fusion="triggered", risk_active=False, last_action=0,
            steps_since_replan=0, max_replan_interval=3,
        )
        assert decision is (expected != "none") and reason == expected

    obs = {"mask": np.asarray([True, True]), "trigger_flags": {}}
    assert _trigger_decision(obs, fusion="triggered", risk_active=True, last_action=0, steps_since_replan=0, max_replan_interval=3)[1] == "risk"
    assert _trigger_decision(obs, fusion="triggered", risk_active=False, last_action=0, steps_since_replan=3, max_replan_interval=3)[1] == "max_wait"
    assert _trigger_decision({"mask": np.asarray([False, True]), "trigger_flags": {}}, fusion="triggered", risk_active=False, last_action=0, steps_since_replan=0, max_replan_interval=3)[1] == "safety"


def test_regular_telemetry_does_not_raise_public_event_signal():
    scenario = M10Scenario("telemetry", (M10TaskSpec("task-0", 0.0, 0.0, 0.0, 10.0, 8.0, 1.0),), ())
    env = M10Environment(M10Config(uav_count=1, task_capacity=1, horizon=5.0), scenario)
    obs = env.reset()
    assert obs["trigger_flags"]["task_arrival"]
    obs, _, _, info = env.step(1)
    assert obs["event_signal"] == 0.0
    assert not any(obs["trigger_flags"].values())
    assert info["command_submitted"] is True


def test_confirmed_fault_is_visible_as_a_distinct_trigger():
    scenario = M10Scenario(
        "fault", (M10TaskSpec("task-0", 0.0, 0.0, 0.0, 10.0, 8.0, 1.0),),
        (ServiceEvent(1.0, "uav-0", "disconnect"),),
    )
    env = M10Environment(M10Config(uav_count=1, task_capacity=1, horizon=4.0), scenario)
    env.reset()
    obs, _, _, _ = env.step(1)
    assert obs["trigger_flags"]["confirmed_fault"]
    decision, reason, _ = _trigger_decision(
        obs, fusion="triggered", risk_active=False, last_action=1,
        steps_since_replan=0, max_replan_interval=3,
    )
    assert decision and reason == "confirmed_fault"


def test_reuse_renews_existing_lease_without_resubmitting_assignment():
    scenario = M10Scenario("reuse", (M10TaskSpec("task-0", 0.0, 0.0, 0.0, 10.0, 8.0, 1.0),), ())
    env = M10Environment(M10Config(uav_count=1, task_capacity=1, horizon=5.0), scenario)
    env.reset()
    _, _, _, first_info = env.step(0)
    command_count = len(env.execution.commands)
    _, _, _, reuse_info = env.step(0, submit_command=False)
    assert first_info["feedback"] == "accepted"
    assert reuse_info["feedback"] == "reuse_existing"
    assert reuse_info["command_submitted"] is False
    assert reuse_info["lease_renewal"] == "renewed"
    assert len(env.execution.commands) == command_count


def test_collect_rollout_skips_actor_on_continuation_steps():
    scenario = M10Scenario("static", (M10TaskSpec("task-0", 0.0, 0.0, 0.0, 20.0, 8.0, 1.0),), ())
    config = M10Config(uav_count=1, task_capacity=1, horizon=20.0)
    ppo = PPOConfig(rollout_steps=5, update_epochs=1)
    policy = M10ActorCritic(uav_count=1, task_capacity=1, action_count=2, encoder="mlp", type_count=2, history=False)
    with patch("gppo_world.m10_training._act", wraps=__import__("gppo_world.m10_training", fromlist=["_act"])._act) as actor:
        transitions = collect_rollout(
            policy, config=ppo, env_config=config, seed=4, device=torch.device("cpu"),
            model=None, fusion="triggered", trigger_threshold=0.5,
            scenarios=[scenario], max_replan_interval=3,
        )
    assert actor.call_count == sum(t.actor_decision for t in transitions)
    assert actor.call_count < len(transitions)
    assert any(not t.actor_decision and not t.info["command_submitted"] for t in transitions)


def test_environment_distinguishes_termination_from_time_limit_truncation():
    unfinished = M10Scenario("unfinished", (M10TaskSpec("task-0", 0.0, 0.0, 0.0, 20.0, 8.0, 1.0),), ())
    env = M10Environment(M10Config(uav_count=1, task_capacity=1, horizon=0.5), unfinished)
    _, _, done, info = env.step(1)
    assert done and info["truncated"] and not info["terminated"]

    finished = M10Scenario("finished", (M10TaskSpec("task-0", 0.0, 0.0, 0.0, 20.0, 0.1, 1.0),), ())
    env = M10Environment(M10Config(uav_count=1, task_capacity=1, horizon=5.0), finished)
    _, _, done, info = env.step(0)
    assert done and info["terminated"] and not info["truncated"]


def test_training_metadata_reports_rollouts_epochs_and_real_optimizer_steps():
    from gppo_world.m10_training import train_policy

    policy, metadata = train_policy(
        variant="r2-accounting", encoder="mlp", type_count=2, history=False,
        fusion="triggered", model=None, seed=3, steps=8,
        env_config=M10Config(horizon=20.0),
        ppo_config=PPOConfig(rollout_steps=4, update_epochs=2),
        device="cpu", max_replan_interval=3,
    )
    assert metadata["environment_steps"] == 8
    assert metadata["rollout_updates"] == 2
    assert metadata["update_epochs"] == 2
    assert metadata["optimizer_updates"] == 4
    assert metadata["actor_decisions"] + metadata["continuation_steps"] == 8
