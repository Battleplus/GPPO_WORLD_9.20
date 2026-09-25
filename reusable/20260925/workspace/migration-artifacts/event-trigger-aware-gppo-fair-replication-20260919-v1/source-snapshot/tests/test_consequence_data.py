import json

import pytest
import torch

from gppo_world.consequence_data import example_from_dict
from gppo_world.registry import FEATURE_REGISTRY


def graph_dict():
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
    return {
        "nodes": {"uav": torch.zeros((4, 12)).tolist(), "region": torch.zeros((4, 12)).tolist(), "target": torch.zeros((3, 16)).tolist()},
        "edge_index": {"/".join(relation): torch.zeros((2, counts[relation]), dtype=torch.long).tolist() for relation in FEATURE_REGISTRY.edge_dimensions},
        "edge_attr": {"/".join(relation): torch.zeros((counts[relation], dimension)).tolist() for relation, dimension in FEATURE_REGISTRY.edge_dimensions.items()},
        "candidate_edges": [[uav, region] for uav in range(4) for region in range(4)],
        "action_mask": [True] + [False] * 16,
        "graph_version": 1,
    }


def record():
    return {
        "episode_id": "ep-1",
        "graph_t": graph_dict(),
        "target": {
            "episode_id": "ep-1", "decision_index": 0, "action": 0,
            "horizon_steps": 1, "exogenous_key": "shared-1",
            "travel_time": 1.0, "service_progress": 0.5,
            "energy_delta": 0.2, "deadline_risk": 0.1,
        },
        "history": [0.0, 1.0],
    }


def test_loader_accepts_visible_counterfactual_record():
    example = example_from_dict(record(), expected_horizon_steps=1)
    assert example.target.action == 0
    assert example.history.shape == (2,)


@pytest.mark.parametrize("field", ["graph_tp1", "hidden_state", "future_events"])
def test_loader_rejects_leakage_fields(field):
    value = record()
    value[field] = {}
    with pytest.raises(ValueError, match="forbidden"):
        example_from_dict(value, expected_horizon_steps=1)


def test_loader_rejects_illegal_counterfactual_action():
    value = record()
    value["target"]["action"] = 1
    with pytest.raises(ValueError, match="legal"):
        example_from_dict(value, expected_horizon_steps=1)
