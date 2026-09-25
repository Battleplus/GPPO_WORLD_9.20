from __future__ import annotations

from dataclasses import replace
import inspect
import tempfile

import torch

from gppo_world.data import TensorTransition
from gppo_world.events import (
    EVIDENCE_EVENTS,
    NOMINAL_SLOTS,
    ORDINAL_SLOTS,
    STRUCTURAL_SLOT_NAMES,
    freeze_event_schema,
    generate_event_labels,
    ges_weight,
    label_digest,
    label_episodes,
)
from gppo_world.model import EventAwareGraphWorldModel

from test_contracts import make_graph


def _evidence(event_id: str, *, status: str = "CONFIRMED", severity: float = 0.5) -> dict:
    return {
        "source": "sensor",
        "signal_type": "TARGET_DISCOVERED",
        "received_at": 1.0,
        "payload": {
            "event_id": event_id,
            "status": status,
            "severity": severity,
            "evidence_count": 1,
            "affected_uavs": ["0"],
            "affected_regions": ["0"],
            "affected_targets": ["0"],
        },
    }


def _transition(step: int = 0, *, evidence=(), next_graph=None) -> TensorTransition:
    graph = make_graph(7 + step)
    if next_graph is None:
        next_graph = make_graph(8 + step)
        next_graph.nodes["uav"][0, 0] = 0.2
        next_graph.action_mask[3] = False
        next_graph.action_mask[4] = True
    return TensorTransition(
        episode_id="episode-1",
        step=step,
        graph=graph,
        next_graph=next_graph,
        evidence=tuple(evidence),
        action=3,
        reward=1.0,
        costs=torch.zeros(7),
        continuation=1.0,
    )


def _schema_and_episode():
    episode = [
        _transition(0, evidence=()),
        _transition(1, evidence=(_evidence("E0"),)),
    ]
    return freeze_event_schema([episode]), episode


def test_event_generation_is_byte_identical():
    schema, episode = _schema_and_episode()
    left = label_episodes([episode], schema)
    right = label_episodes([episode], schema)
    assert label_digest(left) == label_digest(right)
    assert left[0][0][1].canonical_bytes() == right[0][0][1].canonical_bytes()


def test_ordinal_nominal_structural_and_evidence_targets():
    schema, episode = _schema_and_episode()
    labels = generate_event_labels(episode[0], episode[1], schema)
    assert labels.ordinal.shape == (len(ORDINAL_SLOTS),)
    assert labels.nominal.shape == (len(NOMINAL_SLOTS),)
    assert labels.structural.shape == (len(STRUCTURAL_SLOT_NAMES),)
    assert bool(labels.structural[-17 + 3])
    assert bool(labels.structural[-17 + 4])
    assert bool(labels.evidence[EVIDENCE_EVENTS.index("new")])
    assert not bool(labels.evidence[EVIDENCE_EVENTS.index("confirm")])
    assert not bool(labels.evidence_valid[EVIDENCE_EVENTS.index("expire")])


def test_existing_status_transition_generates_confirm_not_conflict():
    schema, _ = _schema_and_episode()
    current = _transition(0, evidence=(_evidence("E0", status="PENDING"),))
    future = _transition(1, evidence=(_evidence("E0", status="CONFIRMED"),))
    labels = generate_event_labels(current, future, schema)
    assert bool(labels.evidence[EVIDENCE_EVENTS.index("confirm")])
    assert not bool(labels.evidence[EVIDENCE_EVENTS.index("conflict")])


def test_terminal_evidence_target_is_ineligible():
    schema, episode = _schema_and_episode()
    labels = generate_event_labels(episode[-1], None, schema)
    assert not bool(labels.evidence_valid.any())


def test_ges_boundary_and_smooth_formula_are_deterministic():
    assert ges_weight(0.19, 0.20, "hard") == 1.0
    assert ges_weight(0.20, 0.20, "hard") == 0.0
    assert ges_weight(0.21, 0.20, "hard") == 0.0
    assert ges_weight(0.10, 0.20, "smooth") > 1.0
    assert ges_weight(0.20, 0.20, "smooth") == 0.0


def test_event_model_has_no_future_input_and_uses_only_current_evidence():
    schema, episode = _schema_and_episode()
    parameters = inspect.signature(EventAwareGraphWorldModel.step).parameters
    assert "next_graph" not in parameters
    assert "reward" not in parameters
    assert "costs" not in parameters
    model = EventAwareGraphWorldModel(event_schema=schema).eval()
    left, _ = model.step(
        episode[0].graph, episode[0].action, sample=False, evidence=episode[0].evidence
    )
    changed_future = replace(
        episode[0], next_graph=make_graph(999), reward=-100.0, costs=torch.ones(7) * 99
    )
    right, _ = model.step(
        changed_future.graph,
        changed_future.action,
        sample=False,
        evidence=changed_future.evidence,
    )
    assert torch.equal(left["ordinal_event_logits"], right["ordinal_event_logits"])


def test_event_checkpoint_roundtrip_is_exact():
    schema, episode = _schema_and_episode()
    torch.manual_seed(11)
    model = EventAwareGraphWorldModel(event_schema=schema).eval()
    expected, _ = model.step(
        episode[0].graph, episode[0].action, sample=False, evidence=episode[0].evidence
    )
    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/event-wm.pt"
        model.save(path, extra={"test": True})
        restored, metadata = EventAwareGraphWorldModel.load(path)
        restored.eval()
        actual, _ = restored.step(
            episode[0].graph, episode[0].action, sample=False, evidence=episode[0].evidence
        )
    assert metadata["test"] is True
    assert torch.equal(expected["evidence_event_logits"], actual["evidence_event_logits"])
