import torch

from gppo_world.joint_gppo import JointTrainConfig
from gppo_world.joint_training import GROUPS, scalarized_preference_advantage


def test_weighted_preference_uses_each_sample_preference_and_fixed_scales():
    config = JointTrainConfig(task_reward_scale=0.5, energy_reward_scale=1.0)
    advantages = torch.tensor([[2.0, -1.0], [2.0, -1.0]])
    preference = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    actual = scalarized_preference_advantage(preference, advantages, config)

    # The same batch must not be replaced by its mean preference.  The fixed
    # task/energy scales give +1.0 for task-only and -1.0 for energy-only.
    assert torch.equal(actual, torch.tensor([1.0, -1.0]))


def test_weighted_preference_direction_changes_with_objective_preference():
    config = JointTrainConfig(task_reward_scale=1.0, energy_reward_scale=1.0)
    advantages = torch.tensor([[1.0, -2.0]])
    task_first = scalarized_preference_advantage(torch.tensor([[1.0, 0.0]]), advantages, config)
    energy_first = scalarized_preference_advantage(torch.tensor([[0.0, 1.0]]), advantages, config)

    assert task_first.item() > 0
    assert energy_first.item() < 0


def test_pair_groups_differ_only_in_actor_update_mode_and_have_no_world_updates():
    assert GROUPS["P"].preference and GROUPS["P"].preco
    assert GROUPS["W"].preference and not GROUPS["W"].preco
    assert not GROUPS["P"].world_model and not GROUPS["W"].world_model
