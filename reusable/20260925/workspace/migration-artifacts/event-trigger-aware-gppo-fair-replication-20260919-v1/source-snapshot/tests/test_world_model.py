from __future__ import annotations

import tempfile

import torch

from gppo_world.data import apply_predicted_delta, state_vector
from gppo_world.model import STATE_DIM, FlatGRUWorldModel, GraphWorldModel

from test_contracts import make_graph


def test_graph_world_model_output_contract():
    graph = make_graph()
    model = GraphWorldModel()
    model.eval()
    outputs, hidden = model.step(graph, action=3, sample=False)
    assert outputs["state_delta"].shape == (STATE_DIM,)
    assert outputs["state_change_probability"].shape == (STATE_DIM,)
    assert outputs["state_logvar"].shape == (STATE_DIM,)
    assert outputs["costs"].shape == (7,)
    assert outputs["h"].shape == (64,)
    assert outputs["z"].shape == (24,)
    assert hidden.shape == (64,)


def test_flat_gru_has_same_prediction_shapes():
    outputs, _ = FlatGRUWorldModel().step(make_graph(), action=3)
    assert outputs["state_delta"].shape == (STATE_DIM,)
    assert outputs["costs"].shape == (7,)


def test_action_condition_changes_prediction():
    torch.manual_seed(1)
    model = GraphWorldModel()
    model.eval()
    left, _ = model.step(make_graph(), action=1, sample=False)
    right, _ = model.step(make_graph(), action=2, sample=False)
    assert not torch.equal(left["state_delta_raw"], right["state_delta_raw"])


def test_no_action_uses_neutral_embedding():
    model = GraphWorldModel()
    model.eval()
    outputs, _ = model.step(make_graph(), action=None, sample=False)
    assert outputs["state_delta"].shape == (STATE_DIM,)


def test_checkpoint_roundtrip():
    torch.manual_seed(2)
    graph = make_graph()
    model = GraphWorldModel()
    model.eval()
    expected, _ = model.step(graph, 3, sample=False)
    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/wm.pt"
        model.save(path, extra={"test": True})
        restored, metadata = GraphWorldModel.load(path)
        restored.eval()
        actual, _ = restored.step(graph, 3, sample=False)
    assert metadata["test"] is True
    assert torch.equal(expected["state_delta"], actual["state_delta"])


def test_predicted_delta_rebuilds_valid_graph():
    graph = make_graph()
    predicted = apply_predicted_delta(graph, torch.zeros(STATE_DIM))
    assert torch.equal(state_vector(graph), state_vector(predicted))
    assert predicted.graph_version == graph.graph_version + 1
