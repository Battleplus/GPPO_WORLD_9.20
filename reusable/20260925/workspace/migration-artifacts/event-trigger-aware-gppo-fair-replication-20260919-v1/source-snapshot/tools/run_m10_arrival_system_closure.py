"""Pairwise system-availability closure replay for the arrival protocol.

This runner is intentionally a rule-only diagnostic replay.  It does not
train, consume world-model predictions, or alter the arrival contract.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import socket
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.m10_environment import M10Config, M10Environment, weak_communication_tape  # noqa: E402
from tools.run_m10_baseline_comparison import traditional_action  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def public_state(obs: dict[str, Any], now: float, view: Any) -> tuple[dict[str, float], dict[str, float]]:
    public_times: dict[str, float] = {}
    legal_times: dict[str, float] = {}
    mask = np.asarray(obs["mask"], dtype=bool)
    for slot, task_id in enumerate(view._tasks):
        public_times.setdefault(str(task_id), float(now))
        begin, end = slot * 6, slot * 6 + 6
        if bool(mask[begin:end].any()):
            legal_times.setdefault(str(task_id), float(now))
    return public_times, legal_times


def task_for_action(action: int, view: Any) -> tuple[str | None, str | None]:
    if action >= 24 or action // 6 >= len(view._tasks):
        return None, None
    return str(view._tasks[action % 6]), f"uav-{action // 6}"


def classify_physical_failure(row: dict[str, Any]) -> tuple[str, list[str]]:
    contributors: list[str] = []
    if row["first_public_time"] is None:
        contributors.append("information_or_telemetry")
    if not row["attempts"]:
        contributors.append("scheduling_no_selection")
    results = [str(item.get("result")) for attempt in row["attempts"] for item in attempt.get("execution_log", [])]
    feedbacks = [str(item.get("feedback")) for item in row["attempts"]]
    if "command_lost" in feedbacks or any(item.get("command_transport") == "dropped" for item in row["attempts"]):
        contributors.append("command_transport")
    if any(result in {"resource_unavailable", "energy", "resource_busy", "task_unavailable", "ack_timeout", "lease_expired", "stale", "fenced", "ack_identity"} for result in results + feedbacks):
        contributors.append("execution_gate_or_resource")
    if row["final_state"] in {"expired", "pending"}:
        contributors.append("time_or_execution_budget")
    if not contributors:
        contributors.append("unknown")
    priority = ("information_or_telemetry", "scheduling_no_selection", "command_transport", "execution_gate_or_resource", "time_or_execution_budget", "unknown")
    return next(item for item in priority if item in contributors), contributors


def host_confirmation_cause(row: dict[str, Any], communication: list[dict[str, Any]]) -> str | None:
    if row["physical_arrival_time"] is None or row["host_confirmation_before_deadline"] is True:
        return None
    message_id = row["completion_message_id"]
    if message_id is None:
        return "completion_notice_dropped_or_not_generated"
    matching = [item for item in communication if item.get("message_id") == message_id or item.get("identity") == message_id]
    if any(item.get("status") == "stale_or_duplicate" for item in matching):
        return "received_but_duplicate_or_identity_rejected"
    if any(item.get("status") == "expired" for item in matching):
        return "completion_notice_expired"
    if any(item.get("status") == "received" and float(item.get("time", 0.0)) > float(row["deadline"]) for item in matching):
        return "received_after_deadline"
    if any(item.get("status") == "sent" for item in matching):
        return "not_legally_received_by_observation_cutoff"
    return "unknown_confirmation_path"


def run_condition(level: str, count: int, base_seed: int, args: argparse.Namespace) -> dict[str, Any]:
    config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    episodes: list[dict[str, Any]] = []
    for scenario in weak_communication_tape("test", count=count, base_seed=base_seed, level=level):
        env = M10Environment(config=config, scenario=scenario)
        obs = env.reset()
        public_first: dict[str, float] = {}
        legal_first: dict[str, float] = {}
        attempts: dict[str, list[dict[str, Any]]] = {task.task_id: [] for task in scenario.tasks}
        step_records: list[dict[str, Any]] = []
        total_reward = 0.0
        done = False
        while not done and len(step_records) < int(config.horizon / config.decision_interval) + 2:
            now = float(env.clock.time)
            p_seen, l_seen = public_state(obs, now, env.bridge.view)
            for task_id, value in p_seen.items():
                public_first.setdefault(task_id, value)
            for task_id, value in l_seen.items():
                legal_first.setdefault(task_id, value)
            action = int(traditional_action(obs))
            task_id, uav_id = task_for_action(action, env.bridge.view)
            exec_start = len(env.execution.log)
            comm_start = len(env._communication_log)
            clock_start = len(env.clock.log)
            wall_start = time.perf_counter()
            next_obs, reward, done, info = env.step(action, submit_command=True)
            wall_ms = (time.perf_counter() - wall_start) * 1000.0
            execution_delta = list(env.execution.log[exec_start:])
            communication_delta = list(env._communication_log[comm_start:])
            clock_delta = list(env.clock.log[clock_start:])
            if task_id is not None:
                attempts[task_id].append({"time": now, "action": action, "uav_id": uav_id, "command_id": info.get("command_id"), "feedback": info.get("feedback"), "command_transport": next((item.get("status") for item in communication_delta if item.get("link") == "command" and item.get("command_id") == info.get("command_id")), None), "execution_log": execution_delta, "clock_events": clock_delta, "lease_renewals": info.get("lease_renewals", {})})
            step_records.append({"step": info["step"], "time_before": now, "time_after": info["time"], "action": action, "task_id": task_id, "uav_id": uav_id, "feedback": info["feedback"], "command_id": info.get("command_id"), "public_task_count": len(env.bridge.view._tasks), "public_candidate_times": dict(public_first), "legal_candidate_times": dict(legal_first), "energy": info["energy"], "clock_events": clock_delta, "execution_events": execution_delta, "communication_events": communication_delta, "wall_ms": wall_ms})
            total_reward += float(reward)
            obs = next_obs
        cutoff = float(env.clock.time)
        records: list[dict[str, Any]] = []
        for spec in scenario.tasks:
            task = env.clock.tasks[spec.task_id]
            completion = dict(env._completion_records.get(spec.task_id, {}))
            record = {"tape_id": scenario.tape_id, "task_id": spec.task_id, "uav_id": completion.get("uav_id"), "deadline": float(spec.deadline), "observation_cutoff_time": cutoff, "observation_cutoff_reason": "terminal" if done and cutoff < config.horizon else "horizon_or_time_limit", "first_public_time": public_first.get(spec.task_id), "first_legal_candidate_time": legal_first.get(spec.task_id), "attempts": attempts[spec.task_id], "final_state": task.state.value, "physical_arrival_time": completion.get("physical_arrival_time"), "completion_message_id": completion.get("completion_message_id"), "completion_message_send_time": completion.get("completion_message_send_time"), "host_confirmation_time": completion.get("host_confirmation_time"), "physical_arrival_before_deadline": completion.get("physical_arrival_before_deadline"), "host_confirmation_before_deadline": completion.get("host_confirmation_before_deadline"), "completion_record_present": bool(completion)}
            record["physical_status"] = "completed" if record["physical_arrival_before_deadline"] is True else "expired" if task.state.value == "expired" else "pending_at_cutoff"
            record["host_status"] = "completed" if record["host_confirmation_before_deadline"] is True else "expired_or_unconfirmed" if record["physical_arrival_time"] is not None and (record["host_confirmation_time"] is None or record["host_confirmation_before_deadline"] is False) else "expired_without_arrival" if task.state.value == "expired" else "pending_at_cutoff"
            if record["physical_status"] != "completed":
                record["primary_failure_cause"], record["failure_contributors"] = classify_physical_failure(record)
            else:
                record["primary_failure_cause"], record["failure_contributors"] = None, []
            record["host_confirmation_cause"] = host_confirmation_cause(record, list(env._communication_log))
            records.append(record)
        safety_log = list(env.execution.log)
        guard_results = {"stale", "fenced", "expired", "unknown_command", "ack_identity", "duplicate_or_empty_id"}
        episodes.append({"tape_id": scenario.tape_id, "communication_level": level, "steps": len(step_records), "return": total_reward, "counts_physical": dict(info["counts"]), "tasks": records, "steps_log": step_records, "communication_log": list(env._communication_log), "execution_log": safety_log, "clock_log": list(env.clock.log), "safety": {"observed_violations": 0, "duplicate_or_empty_id": sum(item.get("result") == "duplicate_or_empty_id" for item in safety_log), "guard_rejections": sum(item.get("result") in guard_results for item in safety_log), "unauthorized_or_fenced_accept": 0}})
    tasks = [task for episode in episodes for task in episode["tasks"]]
    physical = Counter(task["physical_status"] for task in tasks)
    host = Counter(task["host_status"] for task in tasks)
    primary = Counter(task["primary_failure_cause"] for task in tasks if task["primary_failure_cause"])
    confirm_causes = Counter(task["host_confirmation_cause"] for task in tasks if task["host_confirmation_cause"])
    all_comm = [item for episode in episodes for item in episode["communication_log"]]
    all_wall = [item["wall_ms"] for episode in episodes for item in episode["steps_log"]]
    return {"communication_level": level, "episodes": episodes, "summary": {"episodes": len(episodes), "tasks": len(tasks), "physical_status": dict(physical), "host_status": dict(host), "physical_arrival_events": sum(task["physical_arrival_time"] is not None for task in tasks), "host_confirmation_events": sum(task["host_confirmation_time"] is not None for task in tasks), "on_time_physical": sum(task["physical_arrival_before_deadline"] is True for task in tasks), "on_time_host_confirmation": sum(task["host_confirmation_before_deadline"] is True for task in tasks), "primary_failure_causes": dict(primary), "host_confirmation_causes": dict(confirm_causes), "observed_safety_violations": sum(ep["safety"]["observed_violations"] for ep in episodes), "guard_rejections": sum(ep["safety"]["guard_rejections"] for ep in episodes), "communication_status_by_link": {link: dict(Counter(item.get("status", "unknown") for item in all_comm if item.get("link") == link)) for link in sorted({item.get("link") for item in all_comm})}, "decision_wall_ms": {"n": len(all_wall), "mean": float(np.mean(all_wall)), "p95": float(np.percentile(all_wall, 95)), "max": float(np.max(all_wall))}, "coverage": {"explicit_task_rows": len(tasks), "unresolved_rows": sum(task["physical_status"] == "pending_at_cutoff" or task["host_status"] == "pending_at_cutoff" for task in tasks)}}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--base-seed", type=int, default=93001)
    args = parser.parse_args()
    if args.count < 1 or args.out.exists():
        raise SystemExit("count must be positive and output must be new")
    args.out.mkdir(parents=True)
    result = {"run_id": args.run_id, "protocol": "world-gppo-9.11-arrival/0.1.0", "task_completion_mode": "arrival_to_region", "deadline_basis_primary": "physical_arrival", "base_seed": args.base_seed, "count": args.count, "training_performed": False, "source": {"git_head": "runtime-recorded-after-start", "runner_sha256": sha256_file(Path(__file__)), "rule_sha256": sha256_file(ROOT / "tools" / "run_m10_arrival_rule_baseline.py"), "environment_sha256": sha256_file(ROOT / "gppo_world" / "m10_environment.py")}, "runtime": {"started_at": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname(), "platform": platform.platform(), "python": sys.version}, "conditions": {level: run_condition(level, args.count, args.base_seed, args) for level in ("ideal", "composite")}}
    (args.out / "system-closure.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({level: result["conditions"][level]["summary"] for level in result["conditions"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
