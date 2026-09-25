"""No-training weak-communication availability diagnosis.

This tool traces the complete public-to-execution chain and runs a legal,
deterministic scheduler through a frozen feasibility ladder.  The scheduler
uses only the delivered observation and the same bridge/execution gates as a
policy; it is a feasibility diagnostic, not a policy benchmark.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gppo_world.m10_communication import weak_communication_profile
from gppo_world.m10_environment import (
    M10Config, M10Environment, M10Scenario, M10TaskSpec, scenario_from_dict,
)
from gppo_world.service_clock import ServiceEvent


LEVELS = ("ideal", "telemetry-delay", "random-loss", "burst-loss",
          "reorder", "recovery", "composite")


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               sort_keys=True, default=str), encoding="utf-8")


def compact_obs(obs: dict[str, Any]) -> dict[str, Any]:
    return {
        "time": float(obs["time"]),
        "version": int(obs["version"]),
        "mask": [bool(x) for x in np.asarray(obs["mask"]).tolist()],
        "trigger_flags": dict(obs.get("trigger_flags", {})),
        "continuation_action": obs.get("continuation_action"),
    }


def execution_slice(env: M10Environment, start: int) -> list[dict[str, Any]]:
    return [dict(item) for item in env.execution.log[start:]]


def service_slice(env: M10Environment, start: int) -> list[dict[str, Any]]:
    return [dict(item) for item in env.clock.log[start:]]


def communication_ledger(log: list[dict[str, Any]]) -> dict[str, Any]:
    """Return mutually exclusive wire outcomes and nested delivery outcomes.

    A telemetry message_id is one original wire attempt.  ``received``,
    ``stale_or_duplicate`` and ``expired`` classify delivery outcomes; a
    duplicated delivery has the same message_id and is therefore nested in
    the delivery count rather than counted as a second wire send.
    """
    result: dict[str, Any] = {}
    for link in ("telemetry", "command", "ack"):
        items = [x for x in log if x.get("link") == link]
        statuses: dict[str, int] = {}
        for item in items:
            statuses[str(item.get("status"))] = statuses.get(str(item.get("status")), 0) + 1
        result[link] = {"status_counts": statuses, "records": len(items)}
    tele = result["telemetry"]["status_counts"]
    original_attempts = tele.get("sent", 0) + tele.get("dropped", 0)
    delivered = tele.get("received", 0) + tele.get("stale_or_duplicate", 0) + tele.get("expired", 0)
    message_ids = {str(x.get("message_id")) for x in log
                   if x.get("link") == "telemetry" and x.get("message_id") is not None}
    result["telemetry"]["conservation"] = {
        "original_wire_attempts": original_attempts,
        "sent_plus_link_dropped": original_attempts,
        "delivery_outcomes_including_duplicate_deliveries": delivered,
        "unique_message_ids": len(message_ids),
        "duplicate_delivery_is_nested": delivered >= tele.get("received", 0),
        "identity_note": "message_id identifies one original packet; delivery_ordinal identifies duplicate delivery",
    }
    return result


def first_valid_action(obs: dict[str, Any]) -> int | None:
    mask = np.asarray(obs["mask"], dtype=np.bool_)
    valid = np.flatnonzero(mask[:-1])
    return int(valid[0]) if len(valid) else None


def run_legal_scheduler(scenario: M10Scenario, config: M10Config, *, trace: bool = False) -> dict[str, Any]:
    """Run a deterministic scheduler with no access to simulator truth."""
    env = M10Environment(config, scenario)
    obs = env.reset()
    active_action: int | None = None
    records: list[dict[str, Any]] = []
    done = False
    max_steps = int(config.horizon / config.decision_interval) + 3
    while not done and len(records) < max_steps:
        before_obs = compact_obs(obs)
        if active_action is not None and env._active_command is not None:
            action, submit, mode = active_action, False, "continue_existing_lease"
        else:
            candidate = first_valid_action(obs)
            action = candidate if candidate is not None else config.action_count - 1
            submit, mode = True, "new_visible_candidate" if candidate is not None else "hold_no_visible_candidate"
            active_action = action if candidate is not None else None
        resolved = None if action == config.action_count - 1 else env.view.resolve(action)
        execution_start = len(env.execution.log)
        clock_start = len(env.clock.log)
        obs, reward, done, info = env.step(action, submit_command=submit)
        row = {
            "step": len(records), "before": before_obs,
            "action": int(action), "candidate": resolved, "submit_command": submit,
            "scheduler_mode": mode, "reward": float(reward),
            "after": compact_obs(obs),
            "feedback": info["feedback"], "command_id": info.get("command_id"),
            "lease_renewal": info.get("lease_renewal"),
            "tasks": dict(info["tasks"]), "task_service": dict(info["task_service"]),
            "communication_delta": list(info.get("communication_delta", [])),
            "execution_delta": execution_slice(env, execution_start),
            "clock_delta": service_slice(env, clock_start),
            "terminated": bool(info["terminated"]), "truncated": bool(info["truncated"]),
        }
        if trace:
            records.append(row)
        if env._active_command is None:
            active_action = None
    comm = list(getattr(env, "_communication_log", []))
    task_summary = {tid: {"state": task.state.value, "service": float(task.service),
                          "deadline": float(task.deadline), "assigned_uav": task.assigned_uav}
                    for tid, task in env.clock.tasks.items()}
    accepted = [x for x in env.execution.log if x.get("result") == "accepted"]
    return {
        "tape_id": scenario.tape_id, "scenario_seed": scenario.seed,
        "scenario": {"tasks": [task.__dict__ for task in scenario.tasks],
                      "events": [event.__dict__ for event in scenario.events],
                      "communication": scenario.communication.to_dict()},
        "completed": sum(task.state.value == "completed" for task in env.clock.tasks.values()),
        "expired": sum(task.state.value == "expired" for task in env.clock.tasks.values()),
        "task_summary": task_summary, "steps": len(records) if trace else None,
        "accepted_commands": len(accepted),
        "execution_log": list(env.execution.log), "clock_log": list(env.clock.log),
        "communication": communication_ledger(comm),
        "communication_log": comm if trace else [],
        "trace": records,
        "scheduler_contract": "legal delivered observation/mask only; same bridge, ACK, lease, fencing and service gates; no simulator truth",
    }


def minimal_scenarios() -> dict[str, M10Scenario]:
    one = M10TaskSpec("task-0", 0.0, 0.0, 0.0, 8.0, 1.0, 1.0)
    return {
        "one_uav_one_task": M10Scenario("diagnostic-one", (one,), (), seed=1701,
                                         split="diagnostic", tape_id="diagnostic-one"),
        "one_uav_recovery": M10Scenario(
            "diagnostic-recovery", (M10TaskSpec("task-0", 0.0, 0.0, 0.0, 10.0, 3.0, 1.0),),
            (ServiceEvent(1.0, "uav-0", "disconnect"),
             ServiceEvent(4.0, "uav-0", "reconnect")), seed=1702,
            split="diagnostic", tape_id="diagnostic-recovery"),
    }


def minimal_config(name: str) -> M10Config:
    return M10Config(uav_count=1 if name == "one_uav_one_task" else 2,
                     task_capacity=1, region_count=1, target_count=1,
                     event_capacity=2, horizon=10.0)


def policy_trace(policy_path: Path, scenario: M10Scenario, config: M10Config,
                 device: str = "cpu") -> dict[str, Any]:
    """Trace the frozen GPPO actor without training."""
    import torch
    from gppo_world.m10_training import _act, _policy_input_bundle
    from tools.run_m10_r3 import load_policy

    policy, metadata = load_policy(policy_path, config, device)
    env = M10Environment(config, scenario)
    obs = env.reset()
    hidden = None
    rows: list[dict[str, Any]] = []
    done = False
    device_obj = torch.device(device)
    max_steps = int(config.horizon / config.decision_interval) + 3
    while not done and len(rows) < max_steps:
        before = compact_obs(obs)
        vector, _, _, _ = _policy_input_bundle(
            env, obs, None, fusion="base", device=device_obj, trigger_threshold=0.1)
        execution_start, clock_start = len(env.execution.log), len(env.clock.log)
        action, log_prob, value, hidden = _act(
            policy, vector, obs["mask"], hidden, device_obj, deterministic=True)
        resolved = None if action == config.action_count - 1 else env.view.resolve(int(action))
        obs, reward, done, info = env.step(int(action), submit_command=True)
        rows.append({
            "step": len(rows), "before": before, "action": int(action),
            "candidate": resolved, "log_prob": float(log_prob), "value": float(value),
            "after": compact_obs(obs), "reward": float(reward),
            "feedback": info["feedback"], "command_id": info.get("command_id"),
            "communication_delta": list(info.get("communication_delta", [])),
            "execution_delta": execution_slice(env, execution_start),
            "clock_delta": service_slice(env, clock_start),
            "tasks": dict(info["tasks"]), "task_service": dict(info["task_service"]),
        })
    return {
        "policy": str(policy_path), "policy_metadata": metadata,
        "tape_id": scenario.tape_id, "scenario_seed": scenario.seed,
        "completed": sum(task.state.value == "completed" for task in env.clock.tasks.values()),
        "expired": sum(task.state.value == "expired" for task in env.clock.tasks.values()),
        "trace": rows, "execution_log": list(env.execution.log),
        "clock_log": list(env.clock.log),
        "communication": communication_ledger(list(env._communication_log)),
    }


def profile_scenarios(scenarios: list[M10Scenario], level: str) -> list[M10Scenario]:
    profile = weak_communication_profile(level)
    return [replace(item, communication=profile) for item in scenarios]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tapes", type=Path, help="JSON containing final_test scenarios")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-episodes", type=int, default=16)
    args = parser.parse_args()
    config = M10Config()
    if args.tapes:
        payload = json.loads(args.tapes.read_text(encoding="utf-8"))
        full = [scenario_from_dict(item) for item in payload["final_test"]][:args.max_episodes]
    else:
        full = list(minimal_scenarios().values())

    ladder: dict[str, Any] = {}
    for level in LEVELS:
        level_scenarios = profile_scenarios(full, level)
        ladder[level] = {
            "legal_scheduler": [run_legal_scheduler(item, config, trace=True)
                                 for item in level_scenarios],
            "aggregate": {
                "episodes": len(level_scenarios),
                "completed_tasks": sum(run_legal_scheduler(item, config)["completed"]
                                        for item in level_scenarios),
            },
        }
    minimal = {}
    for name, scenario in minimal_scenarios().items():
        small_config = minimal_config(name)
        minimal[name] = {level: run_legal_scheduler(
            replace(scenario, communication=weak_communication_profile(level)), small_config, trace=True)
                         for level in LEVELS}

    result: dict[str, Any] = {
        "protocol": "M-10 weak communication availability diagnosis; no training",
        "source": "current checkout",
        "levels": list(LEVELS), "full_tape_count": len(full),
        "full_tape_ids": [item.tape_id for item in full],
        "ladder": ladder, "minimal_feasibility": minimal,
        "interpretation_rules": {
            "completed": "TaskLifecycle completed after real service, never assignment alone",
            "recovery": "affected task served by a different legal resource after disconnect/damage; source UAV recovery is not claimed",
            "telemetry_conservation": "sent+dropped counts original wire attempts; delivery outcomes include nested duplicate deliveries",
            "bytes": "not measured by this tool; prior canonical JSON audit bytes are not physical wire bytes",
        },
    }
    if args.policy:
        failure = profile_scenarios(full[:1], "composite")[0]
        result["frozen_gppo_trace"] = policy_trace(args.policy, failure, config, args.device)
    dump(args.output, result)


if __name__ == "__main__":
    main()
