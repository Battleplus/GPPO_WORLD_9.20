"""Run the versioned S1 execution-layer acceptance around frozen GPPO.

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
    ended: bool
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
    assertion_failures: list[str]
    scenario_assertions_passed: bool
    trace_sha256: str
    trace: list[dict[str, Any]]


def run_tape(args, imports, scenario: str, index: int, model):
    ActionSubmission, Env, EventTape, RandomEvent, RandomEventType, decode_edge_action, _ = imports
    tape = make_tape(EventTape, RandomEvent, RandomEventType, scenario, index)
    tape_bytes = tape.to_bytes()
    env = Env(initial_seed=tape.initial_seed, event_seed=tape.event_seed, mode=tape.mode, event_tape=tape, max_decisions=MAX_DECISIONS)
    graph, reset_info = env.reset()
    energy = {uid: (0.0 if scenario == "energy_insufficient" else ENERGY_INITIAL) for uid in env.uavs}
    trace: list[dict[str, Any]] = []
    counters = {"reassignments": 0, "constraint_violations": 0, "fallback_count": 0, "stale_rejections": 0, "energy_rejections": 0, "illegal_effective_actions": 0, "future_input_violations": 0, "collector_safety_violations": 0}
    observed_types: list[str] = []
    terminated = truncated = False
    end_reason = "max_decisions"
    for step in range(MAX_DECISIONS):
        ctx = env.begin_decision()
        before_version = (int(env.graph_version), int(env.decision_version))
        graph_fields = {str(name).lower() for name in vars(ctx.graph)}
        if any(token in name for name in graph_fields for token in ("future", "truth", "oracle", "optimal")):
            counters["future_input_violations"] += 1
        proposed, _, _, _ = model.act(ctx.graph, deterministic=True)
        proposed = int(proposed)
        probe_override = False
        if scenario == "energy_insufficient" and step == 0 and proposed == ctx.graph.noop_action:
            legal_edges = [int(index) for index in ctx.graph.action_mask[:-1].nonzero().flatten().tolist()]
            if legal_edges:
                proposed = legal_edges[0]
                probe_override = True
        if not (0 <= proposed < ctx.graph.num_actions and bool(ctx.graph.action_mask[proposed].item())):
            counters["illegal_effective_actions"] += 1
        executed = proposed
        energy_rejected = False
        decoded = decode_edge_action(ctx.graph, proposed)
        if decoded is not None:
            uid, _ = decoded
            if energy[uid] < ENERGY_COST:
                energy_rejected = True
                counters["energy_rejections"] += 1
                executed = ctx.graph.noop_action
        if scenario in {"communication_interrupt", "composite_three_factor"} and step == 0:
            env.advance_time(3.0)
        stale_snapshot_before = json_hash(env.snapshot())
        result = env.submit_action(ActionSubmission.from_decision(executed, ctx))
        info = result[-1]
        if bool(info.get("stale_decision", False)):
            counters["stale_rejections"] += 1
            stale_snapshot_after = json_hash(env.snapshot())
            if stale_snapshot_before != stale_snapshot_after:
                counters["collector_safety_violations"] += 1
            retry = env.begin_decision()
            retry_action, _, _, _ = model.act(retry.graph, deterministic=True)
            result = env.submit_action(ActionSubmission.from_decision(int(retry_action), retry))
            info = result[-1]
            executed = int(retry_action)
        if decoded is not None and not energy_rejected and not bool(info.get("stale_decision", False)):
            uid, _ = decoded
            energy[uid] -= ENERGY_COST
        if energy_rejected and executed != ctx.graph.noop_action:
            counters["constraint_violations"] += 1
        if any(value < -1e-9 for value in energy.values()):
            counters["constraint_violations"] += 1
        for event_id in info.get("new_events", ()):
            record = env.event_records.get(event_id)
            if record is not None:
                observed_types.append(record.event.event_type.value)
        if info.get("repaired_action") not in (None, executed) and not energy_rejected:
            counters["constraint_violations"] += 1
        trace.append({
            "step": step,
            "decision_time": float(getattr(env, "current_time", step)),
            "graph_version": int(ctx.graph_version),
            "action_version": int(ctx.action_version),
            "proposed_action": proposed,
            "executed_action": executed,
            "energy_before": dict(energy),
            "energy_rejected": energy_rejected,
            "execution_layer_probe_override": probe_override,
            "stale_decision": bool(info.get("stale_decision", False)),
            "new_events": list(info.get("new_events", ())),
            "reward": float(result[1]),
            "terminated": bool(result[2]),
            "truncated": bool(result[3]),
        })
        terminated, truncated = bool(result[2]), bool(result[3])
        if terminated or truncated:
            end_reason = "terminated" if terminated else "timeout"
            break
    observed_types = [record.event.event_type.value for record in env.event_records.values()]
    counters["reassignments"] = sum(
        1 for record in env.event_records.values() if record.actual_affected_regions
    )
    factors = {
        "single_uav_damage": "UAV_DAMAGE" in observed_types,
        "communication_anomaly": counters["stale_rejections"] > 0 or scenario in {"communication_interrupt", "composite_three_factor"},
        "low_confidence": scenario != "composite_three_factor" or any("weak" in event_id for event_id in env.event_records),
    }
    if scenario == "normal":
        task_completed = terminated and not truncated
    else:
        task_completed = terminated and not truncated
    result_without_trace = {
        "scenario": scenario,
        "tape_id": f"{scenario}-{index:02d}",
        "tape_sha256": hashlib.sha256(tape_bytes).hexdigest(),
        "decisions": len(trace),
        "ended": bool(terminated or truncated),
        "end_reason": end_reason,
        "task_completed": task_completed,
        "timeout": bool(truncated),
        **counters,
        "observed_event_types": sorted(set(observed_types)),
        "factors": {key: bool(value) for key, value in factors.items()},
    }
    assertion_failures = []
    if scenario == "emergency" and "TARGET_DISCOVERED" not in observed_types:
        assertion_failures.append("emergency_event_not_observed")
    if scenario in {"uav_damage", "composite_three_factor"} and "UAV_DAMAGE" not in observed_types:
        assertion_failures.append("uav_damage_not_observed")
    if scenario == "energy_insufficient" and counters["energy_rejections"] == 0:
        assertion_failures.append("energy_constraint_not_exercised")
    if scenario in {"communication_interrupt", "composite_three_factor"} and counters["stale_rejections"] == 0:
        assertion_failures.append("stale_submission_not_rejected")
    if scenario == "composite_three_factor" and not factors["low_confidence"]:
        assertion_failures.append("low_confidence_factor_not_present")
    result_without_trace["assertion_failures"] = assertion_failures
    result_without_trace["scenario_assertions_passed"] = not assertion_failures
    trace_sha = json_hash(trace)
    result_without_trace["trace_sha256"] = trace_sha
    result_without_trace["trace"] = trace
    return TapeResult(**result_without_trace)


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
    results = []
    for scenario in SCENARIOS:
        for index in range(TAPES_PER_SCENARIO):
            results.append(asdict(run_tape(args, imports, scenario, index, model)))
    summary = {}
    for scenario in SCENARIOS:
        rows = [row for row in results if row["scenario"] == scenario]
        summary[scenario] = {
            "tapes": len(rows),
            "ended": sum(row["ended"] for row in rows),
            "completed": sum(row["task_completed"] for row in rows),
            "timeouts": sum(row["timeout"] for row in rows),
            "reassignments": sum(row["reassignments"] for row in rows),
            "constraint_violations": sum(row["constraint_violations"] for row in rows),
            "fallbacks": sum(row["fallback_count"] for row in rows),
            "stale_rejections": sum(row["stale_rejections"] for row in rows),
            "energy_rejections": sum(row["energy_rejections"] for row in rows),
            "illegal_effective_actions": sum(row["illegal_effective_actions"] for row in rows),
            "future_input_violations": sum(row["future_input_violations"] for row in rows),
            "collector_safety_violations": sum(row["collector_safety_violations"] for row in rows),
            "scenario_assertion_failures": sum(len(row["assertion_failures"]) for row in rows),
        }
    report = {
        "format": "m09-s1-functional-acceptance/1.0.0",
        "status": "passed" if all(
            row["ended"]
            and row["constraint_violations"] == 0
            and row["illegal_effective_actions"] == 0
            and row["future_input_violations"] == 0
            and row["collector_safety_violations"] == 0
            and row["scenario_assertions_passed"]
            for row in results
        ) else "failed",
        "policy_checkpoint_sha256": sha256(args.checkpoint.resolve()),
        "policy_metadata": json_safe(metadata),
        "scenarios": list(SCENARIOS),
        "tape_count": len(results),
        "summary": summary,
        "results": results,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "s1-acceptance.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (args.output / "s1-summary.json").write_text(json.dumps({"format": report["format"], "status": report["status"], "summary": summary}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"], "tape_count": len(results), "summary": summary}, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
