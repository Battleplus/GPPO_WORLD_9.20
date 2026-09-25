"""Run corrected S1-R acceptance around frozen GPPO.

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
import sys
from typing import Any

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
    return {str(x) for x in (ids.keys() if isinstance(ids, dict) else ids)}


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
        fields = {str(x) for x in vars(ctx.graph)}
        counters["future_input_violations"] += len(fields - POLICY_GRAPH_ALLOWLIST)
        proposed, _, _, _ = model.act(ctx.graph, deterministic=True)
        proposed = int(proposed)
        noop = int(ctx.graph.noop_action)
        if not (0 <= proposed < int(ctx.graph.num_actions) and bool(ctx.graph.action_mask[proposed].item())):
            counters["illegal_effective_actions"] += 1
            proposed = noop
        if scenario in {"communication_interrupt", "composite_three_factor"} and step == 0:
            env.advance_time(3.0)
        decision_time = float(getattr(env, "current_time", step))
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
            retry_fields = {str(x) for x in vars(retry.graph)}
            counters["future_input_violations"] += len(retry_fields - POLICY_GRAPH_ALLOWLIST)
            retry_action, _, _, _ = model.act(retry.graph, deterministic=True)
            retry_action = int(retry_action)
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
                if float(record.event.observed_at) > decision_time:
                    counters["future_input_violations"] += 1
        after_alloc = _alloc(env)
        counters["reassignments"] += count_reassignments(previous_alloc, after_alloc, initial_alloc)
        previous_alloc = after_alloc
        event_timing = []
        for event_id, record in getattr(env, "event_records", {}).items():
            event = record.event
            # Event records are evaluation evidence.  A future record is not a
            # policy input unless its id was actually exposed in this decision.
            event_timing.append({"event_id": str(event_id), "occurred_at": float(event.occurred_at), "observed_at": float(event.observed_at), "decision_time": decision_time, "status": str(getattr(record, "status", ""))})
        trace.append({"step": step, "decision_time": decision_time, "graph_fields": sorted(fields), "policy_input_allowlist_ok": not (fields - POLICY_GRAPH_ALLOWLIST), "graph_version": int(ctx.graph_version), "action_version": int(ctx.action_version), "proposed_action": proposed, "executed_action": executed, "attempts": [asdict(x) for x in attempts], "energy_before": attempts[0].energy_before, "energy_after": attempts[-1].energy_after, "energy": dict(ledger.energy), "stale_decision": bool(info.get("stale_decision", False)), "new_events": list(info.get("new_events", ())), "event_timing": event_timing, "reward": float(result[1]), "terminated": bool(result[2]), "truncated": bool(result[3]), "observed_event_types": observed})
        terminated, truncated = bool(result[2]), bool(result[3])
        if terminated or truncated:
            break
    complete, infeasible, deadline_timeout, state_evidence = _task_state(env, scenario)
    outcome = classify_episode(terminated=terminated, truncated=truncated, task_completed=complete, infeasible=infeasible, deadline_exceeded=deadline_timeout)
    observed_types = sorted({record.event.event_type.value for record in env.event_records.values()})
    confirmed = _confirmed_ids(env)
    low_conf = False
    for record in env.event_records.values():
        event = record.event
        if event.severity < 0.6:
            low_conf = low_confidence_confirmed(event_id=str(event.event_id), confidence=float(event.payload.get("confidence", event.severity)), confirmed_ids=confirmed, occurred_at=float(event.occurred_at), observed_at=float(event.observed_at), decision_time=float(getattr(env, "current_time", 0.0)))
    factors = {"single_uav_damage": "UAV_DAMAGE" in observed_types, "communication_anomaly": counters["stale_rejections"] > 0, "low_confidence": low_conf}
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
    if scenario == "composite_three_factor" and not factors["low_confidence"]: failures.append("low_confidence_not_detector_confirmed")
    return TapeResult(scenario, f"{scenario}-{index:02d}", hashlib.sha256(tape.to_bytes()).hexdigest(), len(trace), outcome.episode_ended, outcome.task_failed, outcome.infeasible, outcome.deadline_exceeded, outcome.end_reason, outcome.task_completed, bool(truncated), counters["reassignments"], counters["constraint_violations"], counters["fallback_count"], counters["stale_rejections"], counters["energy_rejections"], counters["illegal_effective_actions"], counters["future_input_violations"], counters["collector_safety_violations"], observed_types, factors, emergency, failures, not failures, json_hash(trace), trace)

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
    status = "passed" if all(r["episode_ended"] and r["constraint_violations"] == 0 and r["illegal_effective_actions"] == 0 and r["future_input_violations"] == 0 and r["collector_safety_violations"] == 0 and r["scenario_assertions_passed"] for r in results) else "blocked"
    report = {"format": "m09-s1r-corrected-acceptance/1.0.0", "status": status, "policy_checkpoint_sha256": sha256(args.checkpoint.resolve()), "policy_metadata": json_safe(metadata), "scenarios": list(SCENARIOS), "tape_count": len(results), "summary": summary, "execution_layer_probes": probes, "results": results}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "s1r-acceptance.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (args.output / "s1r-summary.json").write_text(json.dumps({"format": report["format"], "status": status, "summary": summary, "execution_layer_probes": probes}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "tape_count": len(results), "summary": summary}, indent=2, sort_keys=True))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
