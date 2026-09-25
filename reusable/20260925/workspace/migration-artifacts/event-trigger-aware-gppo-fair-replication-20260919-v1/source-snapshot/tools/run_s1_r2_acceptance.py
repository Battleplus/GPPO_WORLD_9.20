"""Run corrected S1-R2 acceptance with causal timing and confirmation-chain audits.

The wrapper deliberately keeps the policy graph and 17-action contract unchanged.
Energy and emergency metadata are execution/evaluation fields; communication
staleness is exercised through the baseline's versioned submission API.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from gppo_world.s1_r_contracts import EnergyLedger, classify_episode, count_reassignments, low_confidence_confirmed
except ImportError:  # server runner receives the dependency as a sibling module
    from s1_r_contracts import EnergyLedger, classify_episode, count_reassignments, low_confidence_confirmed

POLICY_GRAPH_ALLOWLIST = {"nodes", "edge_index", "edge_attr", "candidate_edges", "action_mask", "graph_version", "action_version", "num_actions", "noop_action"}


SCENARIOS = (
    "normal",
    "emergency",
    "uav_damage",
    "energy_insufficient",
    "communication_interrupt",
    "composite_three_factor",
)
TAPES_PER_SCENARIO = 10
MAX_DECISIONS = 100
ENERGY_INITIAL = 1.0
ENERGY_COST = 1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def json_safe(value: Any) -> Any:
    """Convert checkpoint metadata and tensor-like values to audit JSON."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return json_safe(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return str(value)


def import_baseline(baseline_root: Path):
    ppo_root = baseline_root / "ppo_allocation"
    if str(baseline_root) not in sys.path:
        sys.path.insert(0, str(baseline_root))
    sys.path.insert(0, str(ppo_root))
    from random_event.environment import ActionSubmission, RandomEventAllocationEnv  # type: ignore
    from random_event.events import EventTape, RandomEvent, RandomEventType  # type: ignore
    from random_event.graph import decode_edge_action  # type: ignore
    from random_event.models import GraphActorCritic  # type: ignore

    return ActionSubmission, RandomEventAllocationEnv, EventTape, RandomEvent, RandomEventType, decode_edge_action, GraphActorCritic


def event(
    RandomEvent,
    RandomEventType,
    event_id: str,
    event_type,
    occurred_at: float,
    observed_at: float,
    *,
    uavs=(),
    regions=(),
    targets=(),
    severity=1.0,
    payload=None,
):
    return RandomEvent(
        event_id=event_id,
        event_type=event_type,
        occurred_at=occurred_at,
        observed_at=observed_at,
        source_event="s1-fixed-tape",
        affected_uavs=tuple(uavs),
        affected_regions=tuple(regions),
        affected_targets=tuple(targets),
        severity=severity,
        payload={} if payload is None else payload,
        event_seed=int(occurred_at * 1000),
        state_version=0,
    )


def make_tape(EventTape, RandomEvent, RandomEventType, scenario: str, index: int):
    initial_seed = 910000 + index
    event_seed = 920000 + index
    damage = event(RandomEvent, RandomEventType, f"{scenario}-{index}-damage", RandomEventType.UAV_DAMAGE, 2.0, 2.0, uavs=(0,))
    vacancy = event(RandomEvent, RandomEventType, f"{scenario}-{index}-vacancy", RandomEventType.REGION_VACANCY, 4.0, 4.0, regions=(0,))
    discovery = event(RandomEvent, RandomEventType, f"{scenario}-{index}-discovery", RandomEventType.TARGET_DISCOVERED, 6.0, 6.0, uavs=(1,), targets=(0,))
    destroyed = event(RandomEvent, RandomEventType, f"{scenario}-{index}-destroyed", RandomEventType.TARGET_DESTROYED, 8.0, 8.0, targets=(1,))
    if scenario == "normal":
        events = ()
    elif scenario == "emergency":
        events = (event(RandomEvent, RandomEventType, f"{scenario}-{index}-urgent", RandomEventType.TARGET_DISCOVERED, 2.0, 2.0, uavs=(1,), targets=(0,), payload={"task_kind": "emergency", "deadline": 10.0}),)
    elif scenario == "uav_damage":
        events = (damage, vacancy)
    elif scenario == "energy_insufficient":
        events = (vacancy,)
    elif scenario == "communication_interrupt":
        events = (
            event(RandomEvent, RandomEventType, f"{scenario}-{index}-anchor", RandomEventType.TARGET_DESTROYED, 0.0, 0.0, targets=(1,)),
            event(RandomEvent, RandomEventType, f"{scenario}-{index}-delayed", RandomEventType.REGION_VACANCY, 2.0, 2.5, regions=(0,)),
        )
    elif scenario == "composite_three_factor":
        events = (
            event(RandomEvent, RandomEventType, f"{scenario}-{index}-anchor", RandomEventType.TARGET_DESTROYED, 0.0, 0.0, targets=(1,)),
            damage,
            event(RandomEvent, RandomEventType, f"{scenario}-{index}-weak", RandomEventType.REGION_VACANCY, 4.0, 4.5, regions=(1,), severity=0.55, payload={"confidence": 0.55}),
            destroyed,
        )
    else:  # pragma: no cover
        raise ValueError(scenario)
    return EventTape(initial_seed=initial_seed, event_seed=event_seed, mode="single", events=tuple(events))


@dataclass
class TapeResult:
    scenario: str
    tape_id: str
    tape_sha256: str
    decisions: int
    episode_ended: bool
    task_failed: bool
    infeasible: bool
    deadline_timeout: bool
    end_reason: str
    task_completed: bool
    timeout: bool
    reassignments: int
    constraint_violations: int
    fallback_count: int
    stale_rejections: int
    energy_rejections: int
    illegal_effective_actions: int
    future_input_violations: int
    collector_safety_violations: int
    observed_event_types: list[str]
    factors: dict[str, bool]
    emergency_metrics: dict[str, Any]
    causal_evidence: dict[str, Any]
    confirmation_pipeline: list[dict[str, Any]]
    assertion_failures: list[str]
    scenario_assertions_passed: bool
    trace_sha256: str
    trace: list[dict[str, Any]]


def _alloc(env):
    return {int(k): getattr(v, "assigned_uav", None) for k, v in getattr(env, "regions", {}).items()}


def _task_state(env, scenario):
    pending = list(getattr(env, "pending_regions", ()) or ())
    queue = list(getattr(env, "event_queue", ()) or ())
    records = getattr(env, "event_records", {})
    urgent = [r for r in records.values() if getattr(getattr(r, "event", None), "payload", {}).get("task_kind") == "emergency"]
    urgent_done = True
    for r in urgent:
        for tid in getattr(r.event, "affected_targets", ()):
            target = getattr(env, "targets", {}).get(tid)
            urgent_done = urgent_done and bool(getattr(target, "tracked", False) or getattr(target, "discovered", False))
    complete = not pending and not queue and urgent_done
    deadline = None
    completion_time = None
    for r in urgent:
        deadline = r.event.payload.get("deadline")
        if getattr(r, "resolved_at", None) is not None:
            completion_time = float(r.resolved_at)
    now = float(getattr(env, "current_time", 0.0))
    deadline_timeout = bool(deadline is not None and ((completion_time is not None and completion_time > float(deadline)) or (now > float(deadline) and not urgent_done)))
    if deadline_timeout:
        complete = False
    return complete, bool(getattr(env, "final_infeasible", False)), deadline_timeout, {"pending_regions": pending, "event_queue_length": len(queue), "urgent_event_count": len(urgent), "urgent_target_done": urgent_done, "deadline": deadline, "completion_time": completion_time, "current_time": now}


def _confirmed_ids(env):
    belief = getattr(getattr(getattr(env, "runtime_bridge", None), "adapter", None), "belief", None)
    ids = getattr(belief, "confirmed_events", ()) if belief is not None else ()
    values = ids.keys() if isinstance(ids, dict) else ids
    return {str(getattr(x, "event_id", x)) for x in values}


def _policy_action(model, graph) -> int:
    # Keep the M-09 replay/demo entry on the same inference-only path as S2.
    with __import__("torch").inference_mode():
        action, _, _, _ = model.act(graph, deterministic=True)
    return int(action)


def _graph_payload(graph):
    return {
        "nodes": {str(k): json_safe(v) for k, v in graph.nodes.items()},
        "edge_index": {str(k): json_safe(v) for k, v in graph.edge_index.items()},
        "edge_attr": {str(k): json_safe(v) for k, v in graph.edge_attr.items()},
        "candidate_edges": json_safe(graph.candidate_edges),
        "action_mask": json_safe(graph.action_mask),
        "graph_version": int(getattr(graph, "graph_version", 0)),
        "action_version": int(getattr(graph, "action_version", 0)),
    }


def _model_evidence(model, graph):
    with __import__("torch").inference_mode():
        logits, value, _ = model(graph)
    payload = _graph_payload(graph)
    payload["logits"] = json_safe(logits)
    payload["value"] = json_safe(value)
    return json_hash(payload), json_safe(logits), json_safe(value)


def _confirmation_pipeline(env):
    bridge = getattr(env, "runtime_bridge", None)
    adapter = getattr(bridge, "adapter", None)
    machine = getattr(adapter, "state_machine", None)
    records = getattr(machine, "records", {}) if machine is not None else {}
    rows = []
    for event_id, record in records.items():
        observations = getattr(record, "observations", ())
        confirmed = getattr(record, "confirmed_event", None)
        rows.append({
            "event_id": str(event_id),
            "status": str(getattr(record, "status", "")),
            "suspected_at": getattr(record, "suspected_at", None),
            "confirmed_at": getattr(record, "confirmed_at", None),
            "resolved_at": getattr(record, "resolved_at", None),
            "evidence_sources": sorted(str(x) for x in getattr(record, "positive_evidence_sources", set())),
            "observation_count": len(observations),
            "observations": [{"observation_id": str(o.observation_id), "confidence": float(o.confidence), "severity": float(o.severity), "emitted_at": float(o.emitted_at), "received_at": float(o.received_at), "occurred_at": o.occurred_at, "source_id": str(o.source_id), "signal_type": str(o.signal_type), "positive": bool(o.positive), "payload": json_safe(getattr(o, "payload", {}))} for o in observations],
            "confirmed_event_id": None if confirmed is None else str(getattr(confirmed, "event_id", None)),
            "confirmed_received_at": None if confirmed is None else getattr(confirmed, "received_at", None),
        })
    return rows


def _commit_energy(ledger, submission_id, action, graph, info, decode_edge_action, noop):
    effective = int(info.get("repaired_action") if info.get("repaired_action") is not None else action)
    decoded = decode_edge_action(graph, effective)
    uid = decoded[0] if decoded is not None else None
    legal = bool(0 <= effective < int(graph.num_actions) and bool(graph.action_mask[effective].item()))
    return ledger.submit(submission_id=submission_id, action=effective, uid=uid, legal=legal, noop_action=noop, stale=bool(info.get("stale_decision", False)))


def _run_execution_layer_probes():
    ledger = EnergyLedger({0: 0.0, 1: 1.0}, cost=ENERGY_COST)
    low = ledger.submit(submission_id="probe-low", action=1, uid=0, legal=True, noop_action=16)
    first = ledger.submit(submission_id="probe-stale", action=2, uid=1, legal=True, noop_action=16)
    stale = ledger.submit(submission_id="probe-stale", action=2, uid=1, legal=True, noop_action=16, stale=True)
    duplicate = ledger.submit(submission_id="probe-stale", action=2, uid=1, legal=True, noop_action=16)
    return [asdict(x) for x in (low, first, stale, duplicate)]


def run_tape(args, imports, scenario: str, index: int, model):
    ActionSubmission, Env, EventTape, RandomEvent, RandomEventType, decode_edge_action, _ = imports
    tape = make_tape(EventTape, RandomEvent, RandomEventType, scenario, index)
    env = Env(initial_seed=tape.initial_seed, event_seed=tape.event_seed, mode=tape.mode, event_tape=tape, max_decisions=MAX_DECISIONS)
    graph, _ = env.reset()
    ledger = EnergyLedger({uid: (0.0 if scenario == "energy_insufficient" else ENERGY_INITIAL) for uid in env.uavs}, cost=ENERGY_COST)
    trace = []
    counters = {"reassignments": 0, "constraint_violations": 0, "fallback_count": 0, "stale_rejections": 0, "energy_rejections": 0, "illegal_effective_actions": 0, "future_input_violations": 0, "collector_safety_violations": 0}
    initial_alloc = _alloc(env)
    previous_alloc = dict(initial_alloc)
    terminated = truncated = False
    for step in range(MAX_DECISIONS):
        ctx = env.begin_decision()
        decision_start_time = float(getattr(env, "current_time", step))
        pre_decision_confirmed_ids = sorted(_confirmed_ids(env))
        fields = {str(x) for x in vars(ctx.graph)}
        counters["future_input_violations"] += len(fields - POLICY_GRAPH_ALLOWLIST)
        input_hash, policy_logits, policy_value = _model_evidence(model, ctx.graph)
        proposed = _policy_action(model, ctx.graph)
        noop = int(ctx.graph.noop_action)
        if not (0 <= proposed < int(ctx.graph.num_actions) and bool(ctx.graph.action_mask[proposed].item())):
            counters["illegal_effective_actions"] += 1
            proposed = noop
        if scenario in {"communication_interrupt", "composite_three_factor"} and step == 0:
            advanced_event_ids = env.advance_time(3.0)
        else:
            advanced_event_ids = []
        submit_time = float(getattr(env, "current_time", step))
        stale_before = json_hash(env.snapshot())
        result = env.submit_action(ActionSubmission.from_decision(proposed, ctx))
        info = dict(result[-1])
        first = _commit_energy(ledger, f"{scenario}-{index}-d{step}", proposed, ctx.graph, info, decode_edge_action, noop)
        if first.status == "energy_rejected":
            counters["energy_rejections"] += 1
            counters["fallback_count"] += 1
        if bool(info.get("stale_decision", False)):
            counters["stale_rejections"] += 1
            if stale_before != json_hash(env.snapshot()):
                counters["collector_safety_violations"] += 1
            retry = env.begin_decision()
            retry_decision_time = float(getattr(env, "current_time", step))
            retry_fields = {str(x) for x in vars(retry.graph)}
            counters["future_input_violations"] += len(retry_fields - POLICY_GRAPH_ALLOWLIST)
            retry_input_hash, retry_logits, retry_value = _model_evidence(model, retry.graph)
            retry_action = _policy_action(model, retry.graph)
            retry_noop = int(retry.graph.noop_action)
            if not (0 <= retry_action < int(retry.graph.num_actions) and bool(retry.graph.action_mask[retry_action].item())):
                counters["illegal_effective_actions"] += 1
                retry_action = retry_noop
            retry_result = env.submit_action(ActionSubmission.from_decision(retry_action, retry))
            retry_info = dict(retry_result[-1])
            second = _commit_energy(ledger, f"{scenario}-{index}-d{step}-retry", retry_action, retry.graph, retry_info, decode_edge_action, retry_noop)
            if second.status == "energy_rejected":
                counters["energy_rejections"] += 1
                counters["fallback_count"] += 1
            attempts = [first, second]
            result, info, executed = retry_result, retry_info, second.executed_action
        else:
            attempts = [first]
            executed = first.executed_action
        if any(float(v) < -1e-9 for v in ledger.energy.values()):
            counters["constraint_violations"] += 1
        observed = []
        for event_id in info.get("new_events", ()):
            record = env.event_records.get(event_id)
            if record is not None:
                observed.append(record.event.event_type.value)
        after_alloc = _alloc(env)
        counters["reassignments"] += count_reassignments(previous_alloc, after_alloc, initial_alloc)
        previous_alloc = after_alloc
        event_timing = []
        for event_id, record in getattr(env, "event_records", {}).items():
            event = record.event
            # Event records are evaluation evidence.  A future record is not a
            # policy input unless its id was actually exposed in this decision.
            event_timing.append({"event_id": str(event_id), "occurred_at": float(event.occurred_at), "observed_at": float(event.observed_at), "decision_start_time": decision_start_time, "submit_time": submit_time, "step_return_time": float(getattr(env, "current_time", step)), "status": str(getattr(record, "status", ""))})
        row = {"step": step, "decision_time": decision_start_time, "submit_time": submit_time, "step_return_time": float(getattr(env, "current_time", step)), "advanced_event_ids": list(advanced_event_ids), "pre_decision_confirmed_ids": pre_decision_confirmed_ids, "graph_input_hash": input_hash, "policy_logits": policy_logits, "policy_value": policy_value, "graph_fields": sorted(fields), "policy_input_allowlist_ok": not (fields - POLICY_GRAPH_ALLOWLIST), "graph_version": int(ctx.graph_version), "action_version": int(ctx.action_version), "proposed_action": proposed, "executed_action": executed, "attempts": [asdict(x) for x in attempts], "energy_before": attempts[0].energy_before, "energy_after": attempts[-1].energy_after, "energy": dict(ledger.energy), "stale_decision": bool(info.get("stale_decision", False)), "new_events": list(info.get("new_events", ())), "post_step_new_events_are_off_policy_inputs": True, "event_timing": event_timing, "reward": float(result[1]), "terminated": bool(result[2]), "truncated": bool(result[3]), "observed_event_types": observed}
        if bool(info.get("stale_decision", False)):
            row.update({"retry_decision_time": retry_decision_time, "retry_graph_input_hash": retry_input_hash, "retry_policy_logits": retry_logits, "retry_policy_value": retry_value, "retry_graph_version": int(retry.graph_version), "retry_action_version": int(retry.action_version)})
        trace.append(row)
        terminated, truncated = bool(result[2]), bool(result[3])
        if terminated or truncated:
            break
    complete, infeasible, deadline_timeout, state_evidence = _task_state(env, scenario)
    outcome = classify_episode(terminated=terminated, truncated=truncated, task_completed=complete, infeasible=infeasible, deadline_exceeded=deadline_timeout)
    observed_types = sorted({record.event.event_type.value for record in env.event_records.values()})
    confirmed = _confirmed_ids(env)
    pipeline = _confirmation_pipeline(env)
    low_conf_observed = False
    low_conf_confirmed = False
    injection_effective = False
    for row in pipeline:
        payload_conf = None
        observed_confidences = [float(o["confidence"]) for o in row["observations"]]
        for observation in row["observations"]:
            candidate = observation.get("payload", {}).get("confidence")
            if candidate is not None:
                payload_conf = float(candidate)
                break
        if payload_conf is not None:
            injection_effective = injection_effective or any(abs(x - float(payload_conf)) < 1e-9 for x in observed_confidences)
        has_low_conf = any(x < 0.6 for x in observed_confidences)
        low_conf_observed = low_conf_observed or has_low_conf
        low_conf_confirmed = low_conf_confirmed or has_low_conf and row["status"].endswith("CONFIRMED")
    factors = {"single_uav_damage": "UAV_DAMAGE" in observed_types, "communication_anomaly": counters["stale_rejections"] > 0, "low_confidence_observed": low_conf_observed, "low_confidence_confirmed": low_conf_confirmed, "low_confidence_injection_effective": injection_effective}
    emergency = {"arrived": False, "response_time": None, "completed": False, "deadline": None, "completion_time": state_evidence.get("completion_time"), "deadline_met": False}
    for record in env.event_records.values():
        event = record.event
        if event.payload.get("task_kind") == "emergency":
            arrival = float(event.observed_at)
            response = next((row["decision_time"] for row in trace if row["decision_time"] >= arrival and row["executed_action"] != 16), None)
            emergency = {"arrived": True, "response_time": None if response is None else response - arrival, "completed": bool(complete), "deadline": event.payload.get("deadline"), "completion_time": state_evidence.get("completion_time"), "deadline_met": bool(complete and not deadline_timeout)}
    failures = []
    if scenario == "emergency":
        if not emergency["arrived"]: failures.append("emergency_not_arrived")
        if emergency["response_time"] is None: failures.append("emergency_not_responded")
        if not emergency["completed"]: failures.append("emergency_not_completed")
        if not emergency["deadline_met"]: failures.append("emergency_deadline_missed")
    if scenario in {"uav_damage", "composite_three_factor"} and not factors["single_uav_damage"]: failures.append("uav_damage_not_observed")
    if scenario == "energy_insufficient" and counters["energy_rejections"] == 0: failures.append("energy_constraint_not_exercised_by_autonomous_policy")
    if scenario in {"communication_interrupt", "composite_three_factor"} and counters["stale_rejections"] == 0: failures.append("stale_submission_not_rejected")
    if scenario == "composite_three_factor" and not injection_effective: failures.append("low_confidence_injection_not_consumed")
    if scenario == "composite_three_factor" and low_conf_confirmed: failures.append("low_confidence_confirmed_without_sufficient_evidence")
    causal = {"post_step_new_events_are_off_policy_inputs": True, "pre_decision_graph_hashes_recorded": True, "step_return_times_recorded": True, "stale_retry_reaudited": any("retry_graph_input_hash" in row for row in trace)}
    return TapeResult(scenario, f"{scenario}-{index:02d}", hashlib.sha256(tape.to_bytes()).hexdigest(), len(trace), outcome.episode_ended, outcome.task_failed, outcome.infeasible, outcome.deadline_exceeded, outcome.end_reason, outcome.task_completed, bool(truncated), counters["reassignments"], counters["constraint_violations"], counters["fallback_count"], counters["stale_rejections"], counters["energy_rejections"], counters["illegal_effective_actions"], counters["future_input_violations"], counters["collector_safety_violations"], observed_types, factors, emergency, causal, pipeline, failures, not failures, json_hash(trace), trace)


def run_causal_pair(imports, model):
    ActionSubmission, Env, EventTape, RandomEvent, RandomEventType, _, _ = imports
    anchor = event(RandomEvent, RandomEventType, "pair-anchor", RandomEventType.TARGET_DESTROYED, 0.0, 0.0, targets=(1,))
    future_a = event(RandomEvent, RandomEventType, "pair-future-a", RandomEventType.REGION_VACANCY, 4.0, 4.0, regions=(0,))
    future_b = event(RandomEvent, RandomEventType, "pair-future-b", RandomEventType.UAV_DAMAGE, 4.0, 4.0, uavs=(0,))
    tapes = [EventTape(initial_seed=990001, event_seed=990002, mode="single", events=(anchor, future_a)), EventTape(initial_seed=990001, event_seed=990002, mode="single", events=(anchor, future_b))]
    envs = [Env(initial_seed=t.initial_seed, event_seed=t.event_seed, mode=t.mode, event_tape=t, max_decisions=MAX_DECISIONS) for t in tapes]
    for e in envs:
        e.reset()
    before = [e.begin_decision() for e in envs]
    before_evidence = [_model_evidence(model, c.graph) for c in before]
    before_actions = [_policy_action(model, c.graph) for c in before]
    for e in envs:
        e.advance_time(1.0)
    during = [e.begin_decision() for e in envs]
    during_evidence = [_model_evidence(model, c.graph) for c in during]
    during_actions = [_policy_action(model, c.graph) for c in during]
    arrivals = [e.advance_time(3.0) for e in envs]
    after = [e.begin_decision() for e in envs]
    after_evidence = [_model_evidence(model, c.graph) for c in after]
    after_actions = [_policy_action(model, c.graph) for c in after]
    return {
        "future_event_ids": [[x.event_id for x in t.events[1:]] for t in tapes],
        "decision_times": {"before": 0.0, "during": 1.0, "after": 4.0},
        "before_equal": before_evidence[0][0] == before_evidence[1][0],
        "during_equal": during_evidence[0][0] == during_evidence[1][0],
        "after_graphs_differ": after_evidence[0][0] != after_evidence[1][0],
        "before_logit_equal": before_evidence[0][1] == before_evidence[1][1],
        "during_logit_equal": during_evidence[0][1] == during_evidence[1][1],
        "after_logit_differ": after_evidence[0][1] != after_evidence[1][1],
        "before_action_equal": before_actions[0] == before_actions[1],
        "during_action_equal": during_actions[0] == during_actions[1],
        "after_action_differ": after_actions[0] != after_actions[1],
        "actions": {"before": before_actions, "during": during_actions, "after": after_actions},
        "arrivals": [[str(x) for x in a] for a in arrivals],
        "graph_versions": [[int(c.graph_version) for c in before], [int(c.graph_version) for c in during], [int(c.graph_version) for c in after]],
        "negative_test_future_hidden_until_arrival": before_evidence[0][0] == before_evidence[1][0] and during_evidence[0][0] == during_evidence[1][0],
        "positive_test_after_arrival_changes_input": after_evidence[0][0] != after_evidence[1][0],
    }


def run_confirmation_fixtures(baseline_root: Path):
    if str(baseline_root) not in sys.path:
        sys.path.insert(0, str(baseline_root))
    from event_runtime.detector import EventDetector
    from event_runtime.adapter import EventRuntimeAdapter
    from event_runtime.events import EventType, TruthEvent, TruthEventTape
    from event_runtime.observation import Observation

    detector = EventDetector()
    payload_event = TruthEvent(
        event_id="payload",
        event_type=EventType.REGION_VACANCY,
        source_event="fixture",
        occurred_at=1.0,
        affected_regions=("0",),
        severity=0.9,
        payload={"confidence": 0.55},
    )
    payload_tape = TruthEventTape.build(
        tape_id="payload",
        initial_seed=1,
        event_seed=2,
        mode="single",
        initial_snapshot_hash="fixture",
        events=(payload_event,),
    )
    payload_observations = [
        o for o in detector.generate_observation_tape(payload_tape).observations
        if o.event_id == "payload"
    ]

    def make_obs(event_id, source, confidence, received, positive=True):
        return Observation(observation_id=f"obs-{event_id}-{source}-{received}", event_id=event_id, event_type=EventType.TARGET_DISCOVERED, source_event="fixture", source_id=source, source_type="sensor", signal_type="SENSOR_DETECTION", sequence=1, confidence=confidence, positive=positive, emitted_at=received - 0.1, received_at=received, occurred_at=0.0, affected_targets=("0",), severity=0.55, payload={"confidence": confidence})

    insufficient = EventRuntimeAdapter(target_confirmation_count=3)
    first = insufficient.process_observation(make_obs("insufficient", "src-0", 0.55, 1.0))
    second = insufficient.process_observation(make_obs("insufficient", "src-1", 0.55, 2.0))
    insufficient_status = str(insufficient.state_machine.records["insufficient"].status)
    sufficient = EventRuntimeAdapter(target_confirmation_count=3)
    confirmed = [sufficient.process_observation(make_obs("sufficient", f"src-{i}", 0.55, float(i + 1))) for i in range(3)]
    sufficient_record = sufficient.state_machine.records["sufficient"]
    region_sufficient = EventRuntimeAdapter()
    region_observations = [
        Observation(observation_id=f"obs-region-src-{i}", event_id="region-sufficient", event_type=EventType.REGION_VACANCY, source_event="fixture", source_id=f"src-{i}", source_type="sensor", signal_type="REGION_LEASE_VACANT", sequence=i + 1, confidence=0.55, positive=True, emitted_at=float(i + 1) - 0.1, received_at=float(i + 1), occurred_at=0.0, affected_regions=("0",), severity=0.8, payload={"confidence": 0.55})
        for i in range(2)
    ]
    region_confirmed = [region_sufficient.process_observation(o) for o in region_observations]
    expiring = EventRuntimeAdapter(target_confirmation_count=3, suspicion_timeout=5.0)
    expiring.process_observation(make_obs("expiring", "src-0", 0.55, 1.0))
    expiring.advance_time(7.0)
    contradiction = EventRuntimeAdapter(target_confirmation_count=3)
    contradiction.process_observation(make_obs("contradiction", "src-0", 0.55, 1.0, True))
    contradiction.process_observation(make_obs("contradiction", "src-1", 0.55, 2.0, False))
    return {
        "contract": {"target_discovered_requires_distinct_sources": 3, "direct_confirmation_confidence_threshold": 0.95, "severity_is_separate_from_confidence": True},
        "payload_consumption": {"payload_confidence": 0.55, "observed_confidences": [float(o.confidence) for o in payload_observations], "consumed": bool(payload_observations) and all(abs(float(o.confidence) - 0.55) < 1e-9 for o in payload_observations)},
        "insufficient_evidence": {"confirmed": first is not None or second is not None, "status": insufficient_status, "expected": "SUSPECTED"},
        "sufficient_evidence": {"confirmed": any(x is not None for x in confirmed), "status": str(sufficient_record.status), "evidence_sources": sorted(sufficient_record.positive_evidence_sources), "expected": "CONFIRMED"},
        "low_confidence_region_sufficient_evidence": {"confirmed": any(x is not None for x in region_confirmed), "status": str(region_sufficient.state_machine.records["region-sufficient"].status), "evidence_sources": sorted(region_sufficient.state_machine.records["region-sufficient"].positive_evidence_sources), "expected": "CONFIRMED"},
        "expiry": {"status": str(expiring.state_machine.records["expiring"].status), "expected": "EXPIRED"},
        "contradiction": {"status": str(contradiction.state_machine.records["contradiction"].status), "expected": "FALSE_ALARM"},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    imports = import_baseline(args.baseline_root.resolve())
    GraphActorCritic = imports[-1]
    model, metadata = GraphActorCritic.load(args.checkpoint.resolve(), map_location="cpu")
    model.eval()
    results = [asdict(run_tape(args, imports, scenario, index, model)) for scenario in SCENARIOS for index in range(TAPES_PER_SCENARIO)]
    summary = {}
    for scenario in SCENARIOS:
        rows = [r for r in results if r["scenario"] == scenario]
        summary[scenario] = {"tapes": len(rows), "episode_ended": sum(r["episode_ended"] for r in rows), "task_completed": sum(r["task_completed"] for r in rows), "task_failed": sum(r["task_failed"] for r in rows), "infeasible": sum(r["infeasible"] for r in rows), "deadline_timeouts": sum(r["deadline_timeout"] for r in rows), "timeouts": sum(r["timeout"] for r in rows), "reassignments": sum(r["reassignments"] for r in rows), "constraint_violations": sum(r["constraint_violations"] for r in rows), "fallbacks": sum(r["fallback_count"] for r in rows), "stale_rejections": sum(r["stale_rejections"] for r in rows), "energy_rejections": sum(r["energy_rejections"] for r in rows), "illegal_effective_actions": sum(r["illegal_effective_actions"] for r in rows), "future_input_violations": sum(r["future_input_violations"] for r in rows), "collector_safety_violations": sum(r["collector_safety_violations"] for r in rows), "scenario_assertion_failures": sum(len(r["assertion_failures"]) for r in rows)}
    probes = _run_execution_layer_probes()
    causal_pair = run_causal_pair(imports, model)
    confirmation_fixtures = run_confirmation_fixtures(args.baseline_root.resolve())
    causal_ok = bool(causal_pair["negative_test_future_hidden_until_arrival"] and causal_pair["positive_test_after_arrival_changes_input"] and causal_pair["before_logit_equal"] and causal_pair["during_logit_equal"] and causal_pair["before_action_equal"] and causal_pair["during_action_equal"] and causal_pair["after_action_differ"])
    confirmation_ok = bool(confirmation_fixtures["payload_consumption"]["consumed"] and confirmation_fixtures["insufficient_evidence"]["status"].endswith("SUSPECTED") and confirmation_fixtures["sufficient_evidence"]["status"].endswith("CONFIRMED") and confirmation_fixtures["low_confidence_region_sufficient_evidence"]["status"].endswith("CONFIRMED") and confirmation_fixtures["expiry"]["status"].endswith("EXPIRED") and confirmation_fixtures["contradiction"]["status"].endswith("FALSE_ALARM"))
    status = "passed" if all(r["episode_ended"] and r["constraint_violations"] == 0 and r["illegal_effective_actions"] == 0 and r["future_input_violations"] == 0 and r["collector_safety_violations"] == 0 and r["scenario_assertions_passed"] for r in results) and causal_ok and confirmation_ok else "blocked"
    report = {"format": "m09-s1r2-corrected-acceptance/1.0.0", "status": status, "policy_checkpoint_sha256": sha256(args.checkpoint.resolve()), "policy_metadata": json_safe(metadata), "scenarios": list(SCENARIOS), "tape_count": len(results), "summary": summary, "execution_layer_probes": probes, "causal_pair_fixture": causal_pair, "confirmation_fixtures": confirmation_fixtures, "results": results}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "s1r-acceptance.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (args.output / "s1r2-summary.json").write_text(json.dumps({"format": report["format"], "status": status, "summary": summary, "execution_layer_probes": probes, "causal_pair_fixture": causal_pair, "confirmation_fixtures": confirmation_fixtures}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "tape_count": len(results), "summary": summary}, indent=2, sort_keys=True))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

