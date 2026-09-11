import torch

from gppo_world.consequence_model import (
    ActionConsequenceWorldModel,
    ConsequenceModelConfig,
    ConsequencePrediction,
    ConsequenceTarget,
    consequence_loss,
)
from gppo_world.contracts import GraphSnapshot
from gppo_world.model import GraphWorldModel
from gppo_world.registry import FEATURE_REGISTRY


def make_graph() -> GraphSnapshot:
    edge_index = {}
    edge_attr = {}
    counts = {
        ("uav", "can_serve", "region"): 16,
        ("region", "served_by", "uav"): 16,
        ("region", "adjacent", "region"): 8,
        ("target", "located_in", "region"): 3,
        ("region", "contains", "target"): 3,
        ("uav", "tracks", "target"): 12,
        ("target", "tracked_by", "uav"): 12,
        ("uav", "communicates", "uav"): 12,
    }
    for relation, dimension in FEATURE_REGISTRY.edge_dimensions.items():
        edge_index[relation] = torch.zeros((2, counts[relation]), dtype=torch.long)
        edge_attr[relation] = torch.zeros((counts[relation], dimension))
    return GraphSnapshot(
        nodes={"uav": torch.zeros((4, 12)), "region": torch.zeros((4, 12)), "target": torch.zeros((3, 16))},
        edge_index=edge_index,
        edge_attr=edge_attr,
        candidate_edges=torch.tensor([(uav, region) for uav in range(4) for region in range(4)]),
        action_mask=torch.tensor([True, False, True] + [False] * 14),
        graph_version=1,
    )


def test_candidate_rows_are_action_conditioned_and_visible_only():
    torch.manual_seed(4)
    model = ActionConsequenceWorldModel(GraphWorldModel(), ConsequenceModelConfig())
    model.eval()
    prediction = model.predict_candidates(make_graph(), [0, 2])
    assert prediction.actions.tolist() == [0, 2]
    assert prediction.means.shape == (2, 3)
    assert not torch.equal(prediction.means[0], prediction.means[1])
    assert torch.isfinite(prediction.stddev).all()


def test_loss_keeps_bce_and_continuous_uncertainty_separate():
    prediction = ConsequencePrediction(
        actions=torch.tensor([0, 2]),
        means=torch.zeros((2, 3)),
        logvars=torch.zeros((2, 3)),
        deadline_logits=torch.zeros(2),
    )
    losses = consequence_loss(
        prediction,
        {
            "travel_time": torch.tensor([1.0, 2.0]),
            "service_progress": torch.tensor([0.5, 0.0]),
            "energy_delta": torch.tensor([0.1, 0.2]),
            "deadline_risk": torch.tensor([1.0, 0.0]),
        },
    )
    assert set(losses) == {"travel_time_nll", "service_progress_nll", "energy_delta_nll", "deadline_risk_bce", "total"}
    assert losses["deadline_risk_bce"].item() > 0
    assert torch.isfinite(losses["total"])


def test_counterfactual_label_rejects_wrong_source_or_horizon():
    label = ConsequenceTarget("ep", 0, 2, 2, "ep:0:shared", 1.0, 0.2, 0.4, 0.1)
    label.validate(expected_horizon_steps=2)
    bad = ConsequenceTarget("ep", 0, 2, 1, "ep:0:shared", 1.0, 0.2, 0.4, 0.1)
    try:
        bad.validate(expected_horizon_steps=2)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched horizon must be rejected")


def test_policy_scores_are_auxiliary_not_reward_replacement():
    model = ActionConsequenceWorldModel(GraphWorldModel(), ConsequenceModelConfig())
    model.eval()
    prediction = model.predict_candidates(make_graph(), [0, 2])
    policy_scores = torch.tensor([2.0, 1.0])
    combined = model.combine_policy_scores(policy_scores, prediction, 0.25)
    assert combined.shape == policy_scores.shape
    assert torch.allclose(combined - policy_scores, 0.25 * model.auxiliary_score(prediction))
