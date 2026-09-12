import torch

from gppo_world.consequence_model import ConsequenceModelConfig, Graph5ActionConsequenceWorldModel
from gppo_world.graph5 import graph5_from_m10_observation
from gppo_world.m10_environment import M10Environment
from tools.generate_m10_consequence_dataset import make_split


def test_m10_public_observation_is_explicit_graph5_contract():
    observation = M10Environment().reset()
    graph = graph5_from_m10_observation(observation)
    assert tuple(graph.nodes) == ("uav", "region", "target", "task", "event")
    assert graph.num_actions == 25
    model = Graph5ActionConsequenceWorldModel(ConsequenceModelConfig())
    legal = [index for index, allowed in enumerate(graph.action_mask.tolist()) if allowed]
    prediction = model.predict_candidates(graph, legal)
    assert prediction.means.shape[0] == len(legal)
    assert prediction.means.shape[1] == 3
    assert torch.isfinite(prediction.means).all()


def test_graph5_cannot_be_treated_as_legacy_graph_snapshot():
    observation = M10Environment().reset()
    graph = graph5_from_m10_observation(observation)
    assert not hasattr(graph, "candidate_edges")
    assert graph.action_mask.shape == (25,)


def test_counterfactual_generator_reuses_prefix_random_stream():
    rows, ledgers = make_split("train", 1, 91011, 2, 1)
    assert rows and len(rows) == len(ledgers)
    assert len({row["parent_episode_id"] for row in rows}) == 1
    assert len({row["prefix_id"] for row in rows}) == 1
    assert len({row["target"]["exogenous_key"] for row in rows}) == 1
    assert all(item["trace"] for item in ledgers)
