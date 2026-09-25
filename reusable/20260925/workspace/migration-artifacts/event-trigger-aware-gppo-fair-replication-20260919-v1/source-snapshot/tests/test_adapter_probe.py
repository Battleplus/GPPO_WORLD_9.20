from dataclasses import dataclass

import pytest
import torch
from torch import nn

from gppo_world.adapter_probe import AdapterProbe
from gppo_world.gppo_adapter import LatentAdapterConfig, LatentAugmentedActorCritic, LatentContext, LatentContextStore


@dataclass
class Graph:
    graph_version: int
    action_mask: torch.Tensor


class Base(nn.Module):
    def forward(self, graph):
        raw = torch.tensor([0.2, 0.5, -0.1])
        return (raw.masked_fill(~graph.action_mask, torch.finfo(raw.dtype).min),
                torch.tensor(0.4), {"raw_logits": raw,
                "pre_mask_invalid_probability": torch.softmax(raw, 0)[~graph.action_mask].sum()})


def setup(reason=None):
    config = LatentAdapterConfig(latent_dim=2, adapter_dim=4, num_actions=3)
    store = LatentContextStore(config, model_variant="wm", model_version="v1")
    context = (LatentContext.zero(2, model_variant="wm", model_version="v1",
               post_graph_version=7, post_action_version=8, reason=reason) if reason else
               LatentContext((1., 2.), True, "wm", "v1", 7, 8, "episode", 1))
    store.publish(context)
    model = LatentAugmentedActorCritic(Base(), config, context_store=store, model_variant="wm").eval()
    with torch.no_grad():
        model.adapter.actor.weight.fill_(0.3)
        model.adapter.critic.weight.fill_(0.2)
    return model, store, Graph(7, torch.tensor([True, False, True]))


@pytest.mark.parametrize("reason", [None, "episode_reset", "ood", "high_uncertainty", "stale_before", "timeout"])
def test_probe_is_observational_and_reads_context_once(reason):
    model, store, graph = setup(reason)
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    direct = model.act(graph, deterministic=True)
    before = store.counters
    rng_before = torch.get_rng_state()
    probe = AdapterProbe(model)
    observed = probe.act(graph, deterministic=True)
    assert observed == direct
    assert torch.equal(rng_before, torch.get_rng_state())
    after = store.counters
    assert (after["served_valid"] + after["served_fallback"]
            - before["served_valid"] - before["served_fallback"]) == 1
    assert all(torch.equal(weights[k], v) for k, v in model.state_dict().items())
    assert probe.records[0]["invalid_logits_unchanged"]
    assert probe.records[0]["reason"] == (reason or "used")
    assert not model._forward_hooks and not model.base_model._forward_hooks


def test_probe_refuses_training_mode():
    model, _, graph = setup()
    model.train()
    with pytest.raises(ValueError):
        AdapterProbe(model).act(graph)


def test_hooks_removed_on_forward_failure():
    model, _, graph = setup()
    def fail(*args):
        raise RuntimeError("injected")
    model.base_model.forward = fail
    with pytest.raises(RuntimeError):
        AdapterProbe(model).act(graph)
    assert not model._forward_hooks and not model.base_model._forward_hooks
