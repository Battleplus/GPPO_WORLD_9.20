from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from gppo_world.gppo_adapter import (
    LatentAdapterConfig,
    LatentAugmentedActorCritic,
    LatentContext,
    LatentContextStore,
    freeze_world_model,
)


@dataclass(frozen=True)
class FakeGraph:
    action_mask: torch.Tensor
    graph_version: int


class FakeGPPO(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, graph):
        raw = self.scale * torch.tensor([0.2, -0.1, 0.4])
        logits = raw.masked_fill(~graph.action_mask, torch.finfo(raw.dtype).min)
        value = self.scale * torch.tensor(0.25)
        diagnostics = {
            "raw_logits": raw,
            "pre_mask_invalid_probability": torch.softmax(raw, dim=-1)[~graph.action_mask].sum(),
            "gates": {},
        }
        return logits, value, diagnostics


def context(*, graph_version=7, action_version=4, valid=True, values=(1.0, 2.0)):
    return LatentContext(
        latent=tuple(values),
        valid=valid,
        model_variant="wm",
        model_version="wm-v1",
        post_graph_version=graph_version,
        post_action_version=action_version,
        source_episode_id="episode-1",
        source_step=3,
        fallback_reason=None if valid else "ood",
    )


def make_model(*, enabled=True, store=None):
    return LatentAugmentedActorCritic(
        FakeGPPO(),
        LatentAdapterConfig(latent_dim=2, adapter_dim=4, num_actions=3),
        enabled=enabled,
        context_store=store,
        model_variant="wm",
        model_version=None if store is None else store.model_version,
    )


def test_disabled_zero_invalid_and_stale_paths_are_bit_exact():
    graph = FakeGraph(torch.tensor([True, False, True]), graph_version=7)
    base = FakeGPPO()
    base_logits, base_value, base_diagnostics = base(graph)

    disabled = LatentAugmentedActorCritic(
        base,
        LatentAdapterConfig(latent_dim=2, adapter_dim=4, num_actions=3),
        enabled=False,
        model_variant="wm",
    )
    cases = (
        (disabled, context()),
        (make_model(), context(valid=False, values=(0.0, 0.0))),
        (make_model(), context(graph_version=8)),
        (make_model(), context(values=(0.0, 0.0))),
    )
    for model, item in cases:
        logits, value, diagnostics = model(graph, context=item)
        assert torch.equal(logits, base_logits)
        assert torch.equal(value, base_value)
        assert diagnostics.keys() == base_diagnostics.keys()


def test_valid_latent_changes_only_legal_logits_and_value():
    graph = FakeGraph(torch.tensor([True, False, True]), graph_version=7)
    model = make_model()
    with torch.no_grad():
        model.adapter.actor.weight.fill_(0.25)
        model.adapter.critic.weight.fill_(0.25)
    base_logits, base_value, _ = model.base_model(graph)
    logits, value, diagnostics = model(graph, context=context())
    assert diagnostics["latent_adapter_used"] is True
    assert not torch.equal(logits[graph.action_mask], base_logits[graph.action_mask])
    assert torch.equal(logits[~graph.action_mask], base_logits[~graph.action_mask])
    assert not torch.equal(value, base_value)


def test_store_requires_both_graph_and_action_version():
    config = LatentAdapterConfig(latent_dim=2, adapter_dim=4, num_actions=3)
    store = LatentContextStore(config, model_variant="wm", model_version="wm-v1")
    store.publish(context())
    store.prepare_decision(7, 4)
    assert store.read(7).valid
    store.prepare_decision(7, 5)
    rejected = store.read(7)
    assert not rejected.valid
    assert rejected.fallback_reason == "invalid_or_stale_context"
    assert store.counters["stale_context"] == 1


def test_explicit_context_checks_action_and_model_version():
    graph = FakeGraph(torch.tensor([True, False, True]), graph_version=7)
    model = LatentAugmentedActorCritic(
        FakeGPPO(),
        LatentAdapterConfig(latent_dim=2, adapter_dim=4, num_actions=3),
        model_variant="wm",
        model_version="wm-v1",
    )
    with torch.no_grad():
        model.adapter.actor.weight.fill_(0.25)
    base_logits = model.base_model(graph)[0]
    wrong_action = model(graph, context=context(), action_version=5)[0]
    wrong_model = model(
        graph,
        context=LatentContext(
            **{**context().__dict__, "model_version": "other-checkpoint"}
        ),
        action_version=4,
    )[0]
    valid = model(graph, context=context(), action_version=4)[0]
    assert torch.equal(wrong_action, base_logits)
    assert torch.equal(wrong_model, base_logits)
    assert not torch.equal(valid[graph.action_mask], base_logits[graph.action_mask])


def test_save_roundtrip_and_legacy_checkpoint_fallback(tmp_path):
    model = make_model()
    with torch.no_grad():
        model.adapter.actor.weight.fill_(0.125)
    path = tmp_path / "adapter.pt"
    model.save(path, extra={"seed": 9})
    loaded, metadata = LatentAugmentedActorCritic.load(
        path,
        base_factory=lambda spec: FakeGPPO(),
    )
    assert metadata["seed"] == 9
    assert loaded.enabled
    assert torch.equal(loaded.adapter.actor.weight, model.adapter.actor.weight)

    legacy_path = tmp_path / "legacy.pt"
    torch.save({"format": "random-event-gppo-v1"}, legacy_path)
    legacy, legacy_metadata = LatentAugmentedActorCritic.load(
        legacy_path,
        base_factory=lambda spec: FakeGPPO(),
        legacy_loader=lambda *args, **kwargs: (FakeGPPO(), {"original": True}),
    )
    assert not legacy.enabled
    assert legacy_metadata == {
        "original": True,
        "loaded_from_legacy_gppo": True,
        "lossless_fallback": True,
    }


def test_world_model_freeze_is_complete():
    world_model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    freeze_world_model(world_model)
    assert not world_model.training
    assert all(not parameter.requires_grad for parameter in world_model.parameters())


def test_context_rejects_non_finite_latent():
    with pytest.raises(ValueError, match="finite"):
        context(values=(float("nan"), 0.0))
