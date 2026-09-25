from copy import deepcopy
import inspect

import pytest
import torch

from gppo_world.jepa import GraphJEPA, batch_graphs, encode_batch, representation_loss
from test_contracts import make_graph


def inputs():
    return batch_graphs([make_graph(), make_graph()]), torch.tensor([1, 2]), torch.zeros(2, 12)


def test_batched_encoder_matches_independent_and_does_not_mutate():
    model = GraphJEPA()
    graphs = [make_graph(), make_graph()]
    batch = batch_graphs(graphs)
    before = deepcopy(batch)
    assert torch.allclose(encode_batch(model.online, batch), torch.stack([model.online(g) for g in graphs]), atol=1e-6)
    for part in batch:
        for key in batch[part]:
            assert torch.equal(batch[part][key], before[part][key])


def test_future_target_is_stop_gradient_and_ema_only():
    model = GraphJEPA().train()
    graph, actions, evidence = inputs()
    old = [p.clone() for p in model.target_encoder.parameters()]
    current, predicted = model(graph, actions, evidence)
    target = model.target(graph)
    assert not target.requires_grad and not model.target_encoder.training
    loss, _ = representation_loss(current, predicted, target)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    loss.backward()
    assert all(p.grad is None and not p.requires_grad for p in model.target_encoder.parameters())
    optimizer.step()
    assert all(torch.equal(p, q) for p, q in zip(old, model.target_encoder.parameters()))
    expected = [p * 0.99 + q.detach() * 0.01 for p, q in zip(old, model.online.parameters())]
    model.update_target()
    assert all(torch.allclose(p, q, atol=1e-7) for p, q in zip(expected, model.target_encoder.parameters()))


def test_action_ablation_and_future_not_in_inference_api():
    graph, actions, evidence = inputs()
    torch.manual_seed(3)
    action = GraphJEPA()
    neutral = GraphJEPA(group="no_action_jepa")
    neutral.load_state_dict(action.state_dict())
    assert not torch.equal(action(graph, actions, evidence)[1][0], action(graph, actions, evidence)[1][1])
    assert torch.equal(neutral(graph, actions, evidence)[1], neutral(graph, actions.flip(0), evidence)[1])
    assert tuple(inspect.signature(GraphJEPA.forward).parameters) == ("self", "graph", "actions", "evidence")


def test_collapse_penalty_finite_and_nonzero():
    collapsed = torch.zeros(8, 32, requires_grad=True)
    loss, parts = representation_loss(collapsed, collapsed, collapsed)
    assert parts["variance"] > 0.9 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(collapsed.grad).all()
    with pytest.raises(ValueError):
        representation_loss(collapsed[:1], collapsed[:1], collapsed[:1])


@pytest.mark.parametrize("group", ["action_jepa", "no_action_jepa", "supervised_graph"])
def test_checkpoint_roundtrip(tmp_path, group):
    model = GraphJEPA(group=group).eval()
    path = tmp_path / "model.pt"
    model.save(path, {"epoch": 60})
    restored, metadata = GraphJEPA.load(path)
    assert metadata == {"epoch": 60}
    assert torch.equal(model(*inputs())[1], restored(*inputs())[1])
    assert all(not p.requires_grad for p in restored.target_encoder.parameters())


def test_reject_foreign_checkpoint(tmp_path):
    model = GraphJEPA()
    path = tmp_path / "model.pt"
    model.save(path, {})
    data = torch.load(path, weights_only=True)
    data["format"] = "eawm"
    torch.save(data, path)
    with pytest.raises(ValueError):
        GraphJEPA.load(path)
