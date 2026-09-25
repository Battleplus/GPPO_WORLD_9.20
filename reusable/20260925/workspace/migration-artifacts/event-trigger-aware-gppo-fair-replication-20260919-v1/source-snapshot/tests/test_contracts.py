from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from gppo_world.contracts import (
    EvidenceItem,
    ExecutionRecord,
    GraphSnapshot,
    Transition,
    snapshot_from_gppo,
)
from gppo_world.registry import FEATURE_REGISTRY, SCHEMA_VERSION


REL_DIMS = FEATURE_REGISTRY.edge_dimensions


def make_graph(version: int = 7) -> GraphSnapshot:
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
    for relation, dim in REL_DIMS.items():
        count = counts[relation]
        edge_index[relation] = torch.zeros((2, count), dtype=torch.long)
        edge_attr[relation] = torch.zeros((count, dim), dtype=torch.float32)
    mask = torch.zeros(17, dtype=torch.bool)
    mask[3] = True
    return GraphSnapshot(
        nodes={
            "uav": torch.zeros((4, 12)),
            "region": torch.zeros((4, 12)),
            "target": torch.zeros((3, 16)),
        },
        edge_index=edge_index,
        edge_attr=edge_attr,
        candidate_edges=torch.tensor([(u, r) for u in range(4) for r in range(4)]),
        action_mask=mask,
        graph_version=version,
    )


def make_transition(**overrides) -> Transition:
    values = dict(
        episode_id="episode-1",
        scenario_id="single",
        tape_id="single-0001",
        behavior_policy="random_legal",
        seed=42,
        step=0,
        decision_time=10.0,
        next_decision_time=11.0,
        graph_t=make_graph(7),
        evidence_t=(EvidenceItem("detector", "damage", 9.5, {"confidence": 0.8}),),
        execution=ExecutionRecord(3, 3, True, 7, 2, "acked", "cmd-1", "ack-1"),
        reward=1.0,
        costs={"uncovered": 0.0},
        graph_tp1=make_graph(8),
        continuation=True,
    )
    values.update(overrides)
    return Transition(**values)


def test_registry_matches_frozen_gppo_dimensions():
    assert dict(FEATURE_REGISTRY.node_dimensions) == {"uav": 12, "region": 12, "target": 16}
    assert FEATURE_REGISTRY.edge_dimensions[("uav", "can_serve", "region")] == 5
    assert len(FEATURE_REGISTRY.sha256()) == 64


def test_graph_contract_is_16_edges_plus_noop():
    graph = make_graph()
    assert graph.candidate_edges.shape == (16, 2)
    assert graph.num_actions == 17
    assert graph.noop_action == 16


def test_snapshot_copies_source_tensors():
    graph = make_graph()

    @dataclass
    class Source:
        nodes: object
        edge_index: object
        edge_attr: object
        candidate_edges: object
        action_mask: object
        graph_version: int

    source = Source(graph.nodes, graph.edge_index, graph.edge_attr, graph.candidate_edges, graph.action_mask, 7)
    snapshot = snapshot_from_gppo(source)
    source.action_mask[3] = False
    assert bool(snapshot.action_mask[3])


def test_future_evidence_is_rejected():
    evidence = (EvidenceItem("detector", "damage", 10.1, {}),)
    with pytest.raises(ValueError, match="after decision_time"):
        make_transition(evidence_t=evidence)


@pytest.mark.parametrize(
    "payload",
    [
        {"future_graph": [1, 2]},
        {"nested": {"truth_event_occurred_at": 12.0}},
        {"items": [{"oracle_action": 3}]},
    ],
)
def test_truth_only_payload_is_rejected(payload):
    with pytest.raises(ValueError, match="truth-only"):
        EvidenceItem("sim", "event", 9.0, payload)


def test_raw_proposal_cannot_masquerade_as_execution():
    with pytest.raises(ValueError, match="rejected proposal"):
        ExecutionRecord(3, 3, False, 7, 2, "rejected")


def test_illegal_executed_action_is_rejected():
    execution = ExecutionRecord(4, 4, True, 7, 2, "acked")
    with pytest.raises(ValueError, match="must be legal"):
        make_transition(execution=execution)


def test_stale_graph_version_is_rejected():
    execution = ExecutionRecord(3, 3, True, 6, 2, "acked")
    with pytest.raises(ValueError, match="graph_version"):
        make_transition(execution=execution)


def test_model_input_contains_no_future_targets():
    model_input = make_transition().model_input()
    assert model_input.schema_version == SCHEMA_VERSION
    assert not hasattr(model_input, "graph_tp1")
    assert not hasattr(model_input, "reward")
    assert not hasattr(model_input, "continuation")
    assert not hasattr(model_input, "delta_time")
    assert model_input.executed_action == 3


def test_rejected_proposal_is_represented_as_no_execution():
    execution = ExecutionRecord(4, None, False, 7, 2, "stale_rejected")
    transition = make_transition(execution=execution)
    assert transition.model_input().executed_action is None
    assert transition.model_input().execution_accepted is False
