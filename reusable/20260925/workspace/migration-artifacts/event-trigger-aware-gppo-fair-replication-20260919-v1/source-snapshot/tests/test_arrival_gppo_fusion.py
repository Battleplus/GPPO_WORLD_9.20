import pytest
import torch

from gppo_world.arrival_consequence_model import ArrivalModelConfig, Graph5ArrivalConsequenceModel
from gppo_world.arrival_gppo_fusion import (
    CandidateAwarePolicy,
    CandidateConsequenceFeatures,
    CandidateFeatureContract,
    FrozenArrivalConsequenceScorer,
    choose_candidate_action,
    select_candidate_action,
    validate_arrival_policy_metadata,
)
from gppo_world.graph5 import Graph5Snapshot
from gppo_world.m10_environment import M10Config
from gppo_world.m10_environment import M10Environment, default_scenario
from gppo_world.m10_training import M10ActorCritic


def _graph(mask=None):
    nodes = {name: torch.zeros((count, 32)) for name, count in (('uav', 4), ('region', 3), ('target', 4), ('task', 6), ('event', 4))}
    return Graph5Snapshot(
        nodes=nodes,
        candidate_features=torch.zeros((24, 4)),
        action_mask=torch.tensor(mask or ([True] * 25)),
        global_features=torch.zeros(27),
        graph_version=1,
    )


def test_scorer_keeps_candidate_rows_and_is_frozen():
    model = Graph5ArrivalConsequenceModel(ArrivalModelConfig(hidden_dim=16, horizon_steps=6))
    scorer = FrozenArrivalConsequenceScorer(model, device=torch.device('cpu'), contract=CandidateFeatureContract())
    result = scorer.score_snapshot(_graph([True, True] + [False] * 22 + [True]))
    assert result.actions.tolist() == [0, 1, 24]
    assert result.dense().shape == (25, 4)
    assert scorer.model.training is False
    assert all(parameter.requires_grad is False for parameter in scorer.model.parameters())


def test_candidate_prior_changes_only_the_corresponding_action_logit():
    config = M10Config(task_completion_mode='arrival_to_region', deadline_basis='physical_arrival')
    base = M10ActorCritic(
        uav_count=4, task_capacity=6, action_count=25, encoder='graph', type_count=5,
        history=False, region_count=config.region_count, target_count=config.target_count,
        event_capacity=config.event_capacity, relation_width=config.relation_width,
    )
    policy = CandidateAwarePolicy(base)
    with torch.no_grad():
        policy.prior.projection.weight.fill_(1.0)
    obs = torch.zeros((1, base.base_obs_dim))
    features = torch.zeros((1, 25, 4))
    features[0, 1, 0] = 2.0
    logits, _, _ = policy(obs, features)
    assert not torch.equal(logits[0, 0], logits[0, 1])
    assert torch.equal(logits[0, 0], logits[0, 2])


def test_selection_masks_illegal_actions_and_uses_lowest_tie():
    logits = torch.zeros((25,))
    mask = torch.tensor([True, False] + [False] * 22 + [True])
    assert select_candidate_action(logits, mask).item() == 0
    with pytest.raises(ValueError, match='at least one legal'):
        select_candidate_action(logits, torch.zeros(25, dtype=torch.bool))


def test_old_policy_metadata_is_rejected_for_arrival_fusion():
    with pytest.raises(ValueError, match='arrival_to_region'):
        validate_arrival_policy_metadata({'env_config': {'task_completion_mode': 'continuous_service_until_deadline', 'action_count': 25}}, history=True)


def test_public_prediction_policy_and_environment_step_are_connected():
    config = M10Config(task_completion_mode='arrival_to_region', deadline_basis='physical_arrival')
    env = M10Environment(config, default_scenario('mixed', seed=7711, split='test'))
    observation = env.reset()
    base = M10ActorCritic(
        uav_count=config.uav_count, task_capacity=config.task_capacity, action_count=config.action_count,
        encoder='graph', type_count=5, history=True, region_count=config.region_count,
        target_count=config.target_count, event_capacity=config.event_capacity,
        relation_width=config.relation_width,
    )
    policy = CandidateAwarePolicy(base)
    scorer = FrozenArrivalConsequenceScorer(
        Graph5ArrivalConsequenceModel(ArrivalModelConfig(hidden_dim=16, horizon_steps=6)),
        device=torch.device('cpu'), contract=CandidateFeatureContract(),
    )
    action, logits, features, _ = choose_candidate_action(policy, scorer, observation)
    assert 0 <= action < config.action_count and bool(observation['mask'][action])
    assert logits.shape == (config.action_count,)
    assert features.dense().shape == (config.action_count, 4)
    _, _, _, info = env.step(action, submit_command=True)
    assert 'completion_records' in info


def test_candidate_features_do_not_accept_malformed_width():
    with pytest.raises(ValueError, match='25 action slots'):
        from gppo_world.arrival_gppo_fusion import CandidateConsequencePrior
        CandidateConsequencePrior()(torch.zeros((24, 4)))
