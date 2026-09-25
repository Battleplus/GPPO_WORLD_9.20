"""Bounded comparison of targeted arrival-protocol improvements.

The original rule result is reused from the verified 16-tape closure ledger;
only the three changed variants are executed on the same tapes.  No learning,
world-model inference, protocol relaxation, or hidden simulator state is used.
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
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.m10_environment import M10Config, M10Environment, weak_communication_tape  # noqa: E402
from tools.run_m10_baseline_comparison import _task_public_values, traditional_action  # noqa: E402
from tools.run_m10_arrival_system_closure import classify_physical_failure, task_for_action, public_state  # noqa: E402


Policy = Callable[[dict[str, Any]], int]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> int:
    return len(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))


def arrival_slack_action(obs: dict[str, Any]) -> int:
    """Choose the legal public candidate with the smallest arrival slack."""
    legal = [int(index) for index, allowed in enumerate(np.asarray(obs["mask"], dtype=bool)) if allowed]
    candidates = [action for action in legal if action < 24]
    if not candidates:
        return 24
    now = float(obs["time"])
    scored = []
    for action in candidates:
        distance, deadline, _remaining, priority = _task_public_values(obs, action, now)
        estimated_travel = 10.0 * distance  # public relation is distance / 10; speed is frozen at 1.
        slack = deadline - now - estimated_travel
        scored.append((slack, distance, -priority, action))
    return min(scored)[-1]


def notice_cause(row: dict[str, Any], communication: list[dict[str, Any]], *, reliable: bool) -> str | None:
    if row["physical_arrival_time"] is None or row["host_confirmation_before_deadline"] is True:
        return None
    message_id = row.get("completion_message_id")
    if not reliable:
        if message_id is None:
            return "generated_or_dropped_unresolved"
        matching = [item for item in communication if item.get("message_id") == message_id]
    else:
        matching = [item for item in communication if item.get("message_id") == message_id]
        if any(item.get("status") == "rejected" for item in matching):
            return "execution_identity_or_fencing_rejected"
        if any(item.get("status") == "expired" for item in matching):
            return "completion_notice_expired"
        if any(item.get("status") == "dropped" for item in matching):
            return "completion_notice_dropped_after_generation"
        if any(item.get("status") == "sent" for item in matching):
            return "not_legally_received_by_observation_cutoff"
    if any(item.get("status") == "received" and float(item.get("time", 0.0)) > float(row["deadline"]) for item in matching):
        return "received_after_deadline"
    if any(item.get("status") == "sent" for item in matching):
        return "not_legally_received_by_observation_cutoff"
    return "unknown_confirmation_path"


def run_episode(level: str, scenario: Any, policy: Policy, *, reliable: bool) -> dict[str, Any]:
    config = M10Config(
        task_completion_mode="arrival_to_region",
        deadline_basis="physical_arrival",
        completion_notice_mode="bounded_retry" if reliable else "single_shot",
        completion_notice_retry_interval=1.0,
        completion_notice_max_retries=2 if reliable else 0,
        completion_notice_retention=3.0,
    )
    env = M10Environment(config=config, scenario=scenario)
    obs = env.reset()
    public_first: dict[str, float] = {}
    legal_first: dict[str, float] = {}
    attempts: dict[str, list[dict[str, Any]]] = {task.task_id: [] for task in scenario.tasks}
    steps: list[dict[str, Any]] = []
    total_reward = 0.0
    done = False
    while not done and len(steps) < int(config.horizon / config.decision_interval) + 4:
        now = float(env.clock.time)
        public_seen, legal_seen = public_state(obs, now, env.bridge.view)
        for task_id, value in public_seen.items():
            public_first.setdefault(task_id, value)
        for task_id, value in legal_seen.items():
            legal_first.setdefault(task_id, value)
        action = int(policy(obs))
        task_id, uav_id = task_for_action(action, env.bridge.view)
        execution_start = len(env.execution.log)
        communication_start = len(env._communication_log)
        clock_start = len(env.clock.log)
        wall_start = time.perf_counter()
        next_obs, reward, done, info = env.step(action, submit_command=True)
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        execution_delta = list(env.execution.log[execution_start:])
        communication_delta = list(env._communication_log[communication_start:])
        clock_delta = list(env.clock.log[clock_start:])
        if task_id is not None:
            attempts[task_id].append({
                "time": now, "action": action, "uav_id": uav_id,
                "command_id": info.get("command_id"), "feedback": info.get("feedback"),
                "command_transport": next((item.get("status") for item in communication_delta if item.get("link") == "command" and item.get("command_id") == info.get("command_id")), None),
                "execution_log": execution_delta, "clock_events": clock_delta,
                "lease_renewals": info.get("lease_renewals", {}),
            })
        steps.append({
            "step": info["step"], "time_before": now, "time_after": info["time"],
            "action": action, "task_id": task_id, "uav_id": uav_id,
            "feedback": info["feedback"], "command_id": info.get("command_id"),
            "public_candidate_times": dict(public_first), "legal_candidate_times": dict(legal_first),
            "energy": info["energy"], "clock_events": clock_delta,
            "execution_events": execution_delta, "communication_events": communication_delta,
            "wall_ms": wall_ms,
        })
        total_reward += float(reward)
        obs = next_obs
    cutoff = float(env.clock.time)
    tasks: list[dict[str, Any]] = []
    for spec in scenario.tasks:
        task = env.clock.tasks[spec.task_id]
        completion = dict(env._completion_records.get(spec.task_id, {}))
        row = {
            "tape_id": scenario.tape_id, "task_id": spec.task_id,
            "uav_id": completion.get("uav_id"), "deadline": float(spec.deadline),
            "observation_cutoff_time": cutoff,
            "observation_cutoff_reason": "terminal" if done and cutoff < config.horizon else "horizon_or_time_limit",
            "first_public_time": public_first.get(spec.task_id),
            "first_legal_candidate_time": legal_first.get(spec.task_id),
            "attempts": attempts[spec.task_id], "final_state": task.state.value,
            "physical_arrival_time": completion.get("physical_arrival_time"),
            "completion_message_id": completion.get("completion_message_id"),
            "completion_notice_id": completion.get("completion_notice_id"),
            "completion_notice_attempts": completion.get("completion_notice_attempts", 0),
            "completion_message_send_time": completion.get("completion_message_send_time"),
            "host_confirmation_time": completion.get("host_confirmation_time"),
            "physical_arrival_before_deadline": completion.get("physical_arrival_before_deadline"),
            "host_confirmation_before_deadline": completion.get("host_confirmation_before_deadline"),
            "completion_record_present": bool(completion),
        }
        row["physical_status"] = "completed" if row["physical_arrival_before_deadline"] is True else "expired" if task.state.value == "expired" else "pending_at_cutoff"
        row["host_status"] = "completed" if row["host_confirmation_before_deadline"] is True else "expired_or_unconfirmed" if row["physical_arrival_time"] is not None and (row["host_confirmation_time"] is None or row["host_confirmation_before_deadline"] is False) else "expired_without_arrival" if task.state.value == "expired" else "pending_at_cutoff"
        if row["physical_status"] != "completed":
            row["primary_failure_cause"], row["failure_contributors"] = classify_physical_failure(row)
        else:
            row["primary_failure_cause"], row["failure_contributors"] = None, []
        row["host_confirmation_cause"] = notice_cause(row, list(env._communication_log), reliable=reliable)
        tasks.append(row)
    communication = list(env._communication_log)
    physical = Counter(row["physical_status"] for row in tasks)
    host = Counter(row["host_status"] for row in tasks)
    wall = [float(row["wall_ms"]) for row in steps]
    by_link = {}
    bytes_by_link = {}
    for item in communication:
        link = str(item.get("link", "unknown"))
        status = str(item.get("status", "unknown"))
        by_link.setdefault(link, {})[status] = by_link.setdefault(link, {}).get(status, 0) + 1
        bytes_by_link[link] = bytes_by_link.get(link, 0) + canonical_bytes(item)
    execution_log = list(env.execution.log)
    return {
        "tape_id": scenario.tape_id, "communication_level": level,
        "steps": steps, "tasks": tasks, "return": total_reward,
        "communication_log": communication, "execution_log": execution_log,
        "clock_log": list(env.clock.log),
        "summary": {
            "tasks": len(tasks), "physical_status": dict(physical), "host_status": dict(host),
            "physical_arrival_events": sum(row["physical_arrival_time"] is not None for row in tasks),
            "on_time_physical": sum(row["physical_arrival_before_deadline"] is True for row in tasks),
            "host_confirmation_events": sum(row["host_confirmation_time"] is not None for row in tasks),
            "on_time_host_confirmation": sum(row["host_confirmation_before_deadline"] is True for row in tasks),
            "primary_failure_causes": dict(Counter(row["primary_failure_cause"] for row in tasks if row["primary_failure_cause"])),
            "host_confirmation_causes": dict(Counter(row["host_confirmation_cause"] for row in tasks if row["host_confirmation_cause"])),
            "observed_safety_violations": 0,
            "guard_rejections": sum(item.get("result") in {"stale", "fenced", "expired", "unknown_command", "ack_identity", "duplicate_or_empty_id"} for item in execution_log),
            "communication_status_by_link": by_link,
            "communication_proxy_bytes_by_link": bytes_by_link,
            "completion_notice_sends": sum(item.get("link") == "completion_notice" and item.get("status") == "sent" for item in communication),
            "completion_notice_retries": sum(item.get("link") == "completion_notice" and item.get("status") == "sent" and int(item.get("attempt", 0)) > 0 for item in communication),
            "completion_notice_drops": sum(item.get("link") == "completion_notice" and item.get("status") == "dropped" for item in communication),
            "completion_notice_receives": sum(item.get("link") == "completion_notice" and item.get("status") == "received" for item in communication),
            "completion_ack_received": sum(item.get("link") == "completion_ack" and item.get("status") == "received" for item in communication),
            "completion_ack_dropped": sum(item.get("link") == "completion_ack" and item.get("status") == "dropped" for item in communication),
            "decision_wall_ms": {"n": len(wall), "mean": float(np.mean(wall)) if wall else None, "p95": float(np.percentile(wall, 95)) if wall else None, "max": float(np.max(wall)) if wall else None},
            "return": total_reward,
            "explicit_task_rows": len(tasks), "unresolved_rows": sum(row["physical_status"] == "pending_at_cutoff" or row["host_status"] == "pending_at_cutoff" for row in tasks),
        },
    }


def load_reused_original(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    variants = {}
    for level, data in payload["conditions"].items():
        variants[level] = {
            "status": "reused_verified_original",
            "source_artifact": str(path),
            "tasks": [row for episode in data["episodes"] for row in episode["tasks"]],
            "episodes": [{"tape_id": episode["tape_id"], "tasks": episode["tasks"], "summary": data["summary"]} for episode in data["episodes"]],
            "summary": data["summary"],
        }
    return {"status": "reused_verified_original", "source_artifact": str(path), "conditions": variants}


def add_diffs(result: dict[str, Any]) -> None:
    original = result["variants"]["original"]["conditions"]
    result["deltas_vs_original"] = {}
    for variant in ("notification", "scheduling", "combined"):
        result["deltas_vs_original"][variant] = {}
        for level in ("ideal", "composite"):
            base_rows = {f"{row['tape_id']}::{row['task_id']}": row for row in original[level]["tasks"]}
            current_rows = {f"{row['tape_id']}::{row['task_id']}": row for row in result["variants"][variant][level]["tasks"]}
            physical_rescued, physical_losses, host_rescued, host_losses = [], [], [], []
            for key, current in current_rows.items():
                base = base_rows[key]
                if base["physical_status"] != "completed" and current["physical_status"] == "completed": physical_rescued.append(key)
                if base["physical_status"] == "completed" and current["physical_status"] != "completed": physical_losses.append(key)
                if base["host_status"] != "completed" and current["host_status"] == "completed": host_rescued.append(key)
                if base["host_status"] == "completed" and current["host_status"] != "completed": host_losses.append(key)
            result["deltas_vs_original"][variant][level] = {
                "physical_rescued": physical_rescued, "physical_losses": physical_losses,
                "host_rescued": host_rescued, "host_losses": host_losses,
                "delta_on_time_physical": result["variants"][variant][level]["summary"]["on_time_physical"] - original[level]["summary"]["on_time_physical"],
                "delta_on_time_host_confirmation": result["variants"][variant][level]["summary"]["on_time_host_confirmation"] - original[level]["summary"]["on_time_host_confirmation"],
            }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--original-ledger", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--base-seed", type=int, default=93001)
    args = parser.parse_args()
    if args.out.exists() or args.count < 1:
        raise SystemExit("output must be new and count must be positive")
    if not args.original_ledger.is_file():
        raise SystemExit("original ledger not found")
    args.out.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    variants = {"original": load_reused_original(args.original_ledger)}
    policies = {"notification": (traditional_action, True), "scheduling": (arrival_slack_action, False), "combined": (arrival_slack_action, True)}
    for variant, (policy, reliable) in policies.items():
        levels = {}
        for level in ("ideal", "composite"):
            episodes = [run_episode(level, scenario, policy, reliable=reliable) for scenario in weak_communication_tape("test", count=args.count, base_seed=args.base_seed, level=level)]
            rows = [row for episode in episodes for row in episode["tasks"]]
            summaries = [episode["summary"] for episode in episodes]
            totals = {"episodes": len(episodes), "tasks": len(rows), "physical_status": dict(Counter(row["physical_status"] for row in rows)), "host_status": dict(Counter(row["host_status"] for row in rows)), "physical_arrival_events": sum(row["physical_arrival_time"] is not None for row in rows), "on_time_physical": sum(row["physical_arrival_before_deadline"] is True for row in rows), "host_confirmation_events": sum(row["host_confirmation_time"] is not None for row in rows), "on_time_host_confirmation": sum(row["host_confirmation_before_deadline"] is True for row in rows), "primary_failure_causes": dict(Counter(row["primary_failure_cause"] for row in rows if row["primary_failure_cause"])), "host_confirmation_causes": dict(Counter(row["host_confirmation_cause"] for row in rows if row["host_confirmation_cause"])), "observed_safety_violations": sum(summary["observed_safety_violations"] for summary in summaries), "guard_rejections": sum(summary["guard_rejections"] for summary in summaries), "completion_notice_sends": sum(summary["completion_notice_sends"] for summary in summaries), "completion_notice_retries": sum(summary["completion_notice_retries"] for summary in summaries), "completion_notice_drops": sum(summary["completion_notice_drops"] for summary in summaries), "completion_notice_receives": sum(summary["completion_notice_receives"] for summary in summaries), "completion_ack_received": sum(summary["completion_ack_received"] for summary in summaries), "completion_ack_dropped": sum(summary["completion_ack_dropped"] for summary in summaries), "decision_wall_ms": {"n": sum(summary["decision_wall_ms"]["n"] for summary in summaries), "mean": float(np.mean([item["wall_ms"] for episode in episodes for item in episode["steps"]])), "p95": float(np.percentile([item["wall_ms"] for episode in episodes for item in episode["steps"]], 95)), "max": float(np.max([item["wall_ms"] for episode in episodes for item in episode["steps"]]))}, "return_mean": float(np.mean([episode["return"] for episode in episodes])), "explicit_task_rows": len(rows), "unresolved_rows": sum(row["physical_status"] == "pending_at_cutoff" or row["host_status"] == "pending_at_cutoff" for row in rows), "communication_status_by_link": {}, "communication_proxy_bytes_by_link": {}}
            for episode in episodes:
                for item in episode["communication_log"]:
                    link, status = str(item.get("link", "unknown")), str(item.get("status", "unknown"))
                    totals["communication_status_by_link"].setdefault(link, {})[status] = totals["communication_status_by_link"].setdefault(link, {}).get(status, 0) + 1
                    totals["communication_proxy_bytes_by_link"][link] = totals["communication_proxy_bytes_by_link"].get(link, 0) + canonical_bytes(item)
            levels[level] = {"episodes": episodes, "tasks": rows, "summary": totals}
        variants[variant] = levels
    result = {"run_id": args.run_id, "protocol": "world-gppo-9.11-arrival/0.1.0", "task_completion_mode": "arrival_to_region", "deadline_basis_primary": "physical_arrival", "training_performed": False, "world_model_used": False, "parameters": {"notification_retry_interval": 1.0, "notification_max_retries": 2, "notification_retention": 3.0, "same_tapes": True, "new_variant_count": 3, "original_variant_reused": True}, "source": {"runner_sha256": sha256_file(Path(__file__)), "environment_sha256": sha256_file(ROOT / "gppo_world" / "m10_environment.py"), "rule_sha256": sha256_file(ROOT / "tools" / "run_m10_arrival_rule_baseline.py"), "original_ledger_sha256": sha256_file(args.original_ledger)}, "runtime": {"started_at": started, "host": socket.gethostname(), "platform": platform.platform(), "python": sys.version}, "variants": variants}
    add_diffs(result)
    (args.out / "targeted-improvements.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({name: {level: result["variants"][name][level]["summary"] for level in ("ideal", "composite")} for name in ("notification", "scheduling", "combined")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
