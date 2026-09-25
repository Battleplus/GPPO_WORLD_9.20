from copy import deepcopy
from types import SimpleNamespace
import torch
import pytest

from tools.run_r02_readouts import macro, views
from gppo_world.jepa import GraphJEPA, batch_graphs, encode_batch
from gppo_world.adapter_probe import AdapterProbe
from test_contracts import make_graph
from test_adapter_probe import setup


def test_macro_respects_scenario_and_tape_clusters():
    rows = [{'scenario_id': 'a', 'tape_id': 'x'}] * 3 + [
        {'scenario_id': 'a', 'tape_id': 'y'}, {'scenario_id': 'b', 'tape_id': 'z'}]
    result = macro(torch.tensor([0., 0., 0., 4., 10.]), rows)
    assert result['scenario_macro'] == 6.0
    assert result['transition_mean'] == pytest.approx(2.8)


def test_future_oracle_uses_same_encoder_without_leaking_into_other_views():
    torch.manual_seed(19)
    model = GraphJEPA().eval()
    graph = make_graph()
    data = {'graph': batch_graphs([graph, graph]), 'next': batch_graphs([graph, graph]),
            'actions': torch.tensor([1, 2]), 'evidence': torch.zeros(2, 12)}
    rows = [SimpleNamespace(graph=graph)] * 2
    before = {k: v.clone() for k, v in model.state_dict().items()}
    original = views(model, data, rows)
    changed = deepcopy(data)
    changed['next']['nodes']['uav'].add_(2)
    after = views(model, changed, rows)
    for key in original:
        if key != 'observed_future_oracle':
            assert torch.equal(original[key], after[key])
    assert not torch.equal(original['observed_future_oracle'], after['observed_future_oracle'])
    assert torch.allclose(after['observed_future_oracle'][:, :32].float(), encode_batch(model.online, changed['next']))
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_adapter_intervention_changes_actual_adapter_and_preserves_weights():
    model, store, graph = setup()
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    observer = AdapterProbe(model)
    model.enabled = True
    observer.act(graph, deterministic=True)
    model.enabled = False
    observer.act(graph, deterministic=True)
    assert observer.records[0]['reason'] == 'used'
    assert observer.records[1]['reason'] == 'disabled'
    assert observer.records[1]['legal_actor_residual_l2'] == 0
    assert observer.records[1]['critic_residual'] == 0
    assert all(torch.equal(weights[k], v) for k, v in model.state_dict().items())
