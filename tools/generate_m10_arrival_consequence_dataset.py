"""Generate grouped counterfactual labels for the arrival-to-region contract.

This is a simulator-side label generator only. Public Graph-5 observations
are the model input; hidden task/resource state is used only to write labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.graph5 import graph5_from_m10_observation  # noqa: E402
from gppo_world.m10_environment import M10Config, M10Environment, M10Scenario, weak_communication_tape  # noqa: E402


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def arrival_config() -> M10Config:
    return M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")


def compact(action: int, reward: float, done: bool, info: dict[str, Any]) -> dict[str, Any]:
    return {"action": action, "reward": float(reward), "done": bool(done), "time": float(info["time"]),
            "feedback": info["feedback"], "new_events": info["new_events"],
            "communication_delta": info["communication_delta"], "counts": info["counts"],
            "energy": info["energy"], "completion_records": info["completion_records"]}


def branch(scenario: M10Scenario, prefix_actions: list[int], action: int, horizon: int, parent: str, prefix: str, key: str) -> dict[str, Any]:
    config = arrival_config()
    env = M10Environment(config=config, scenario=scenario, exogenous_key=key)
    obs = env.reset()
    prefix_trace = []
    for prefix_action in prefix_actions:
        obs, reward, done, info = env.step(int(prefix_action))
        if done:
            raise RuntimeError("prefix reached terminal state")
        prefix_trace.append(compact(prefix_action, reward, done, info))
    graph = graph5_from_m10_observation(obs).as_dict()
    before_time = float(env.clock.time)
    before_energy = sum(resource.energy for resource in env.clock.resources.values())
    is_noop = action == config.action_count - 1
    task_id = None if is_noop else f"task-{action % config.task_capacity}"
    branch_trace = []
    for index in range(horizon):
        selected = int(action if index == 0 else config.action_count - 1)
        obs, reward, done, info = env.step(selected)
        branch_trace.append(compact(selected, reward, done, info))
        if done:
            break
    task = None if task_id is None else env.clock.tasks[task_id]
    arrivals = [] if task_id is None else [item for item in env.clock.log if item.get("kind") == "arrival" and item.get("task") == task_id]
    arrival = arrivals[0] if arrivals else None
    record = env._completion_records.get(task_id)  # audit-only simulator record, never model input
    after_time = float(env.clock.time)
    deadline_observed = bool(task is not None and (after_time >= float(task.deadline) or arrival is not None))
    failure = "noop" if is_noop else None
    if not is_noop and arrival is None:
        if task.state.value == "expired":
            failure = "not_arrived_before_deadline"
        elif task.assigned_uav is None and any(item.get("result") in ("energy", "resource_unavailable", "lease_expired") for item in env.execution.log):
            failure = "execution_or_energy_failure"
        elif not deadline_observed:
            failure = "censored_window"
        else:
            failure = "not_arrived_before_deadline"
    target = {
        "parent_episode_id": parent,
        "prefix_id": prefix,
        "action": int(action),
        "exogenous_key": key,
        "prediction_horizon_steps": int(horizon),
        "arrival_time": None if arrival is None else float(arrival["time"] - before_time),
        "arrival_before_deadline_physical": None if not deadline_observed else bool(arrival is not None and float(arrival["time"]) <= float(task.deadline)),
        "arrival_before_deadline_host": None if record is None or record.get("host_confirmation_time") is None else bool(record["host_confirmation_before_deadline"]),
        "execution_or_energy_failure": bool(failure == "execution_or_energy_failure"),
        "failure_class": failure,
        "system_energy_delta": float(before_energy - sum(resource.energy for resource in env.clock.resources.values())),
        "label_source": "M10Environment.counterfactual_simulator",
    }
    return {
        "schema": "gppo-arrival-consequence/v1",
        "parent_episode_id": parent,
        "prefix_id": prefix,
        "graph5_t": graph,
        "target": target,
        "label_masks": {"arrival_time": arrival is not None, "arrival_before_deadline_physical": deadline_observed,
                         "arrival_before_deadline_host": record is not None and record.get("host_confirmation_time") is not None,
                         "execution_or_energy_failure": bool(not is_noop and deadline_observed)},
        "label_provenance": {"hidden_state_online": False, "shared_exogenous_key": key,
                              "prefix_trace_sha256": digest(prefix_trace), "branch_trace_sha256": digest(branch_trace),
                              "post_branch_control": "selected action once then periodic NOOP", "deadline_observed_through": after_time},
        "branch_ledger": {"prefix_trace": prefix_trace, "branch_trace": branch_trace,
                           "execution_log": list(env.execution.log), "completion_records": env._completion_records},
    }


def make_split(split: str, count: int, base_seed: int, prefix_steps: int, horizon: int, max_candidates: int) -> list[dict[str, Any]]:
    rows = []
    config = arrival_config()
    for scenario in weak_communication_tape(split, count=count, base_seed=base_seed, level="composite"):
        parent = f"{split}:{scenario.tape_id}"
        prefix = f"{parent}:prefix-{prefix_steps}-public-hash-legal"
        key = f"{scenario.tape_id}|{parent}|{prefix}"
        probe = M10Environment(config=config, scenario=scenario, exogenous_key=key)
        obs = probe.reset()
        actions = []
        for index in range(prefix_steps):
            legal = [i for i, allowed in enumerate(np.asarray(obs["mask"], dtype=bool)) if allowed]
            chosen = legal[int(digest({"parent": parent, "prefix": prefix, "step": index})[:16], 16) % len(legal)]
            actions.append(chosen)
            obs, _, done, _ = probe.step(chosen)
            if done:
                raise RuntimeError("prefix terminated")
        legal = [i for i, allowed in enumerate(np.asarray(obs["mask"], dtype=bool)) if allowed]
        legal = sorted(legal, key=lambda value: digest({"parent": parent, "prefix": prefix, "action": int(value)}))[:max_candidates]
        rows.extend(branch(scenario, actions, int(action), horizon, parent, prefix, key) for action in legal)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count-per-split", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=93001)
    parser.add_argument("--prefix-steps", type=int, default=2)
    parser.add_argument("--horizon-steps", type=int, default=6)
    parser.add_argument("--max-candidates-per-prefix", type=int, default=25)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing non-empty output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "gppo-arrival-consequence/v1", "protocol": "world-gppo-9.11-arrival/0.1.0",
                "task_completion_mode": "arrival_to_region", "deadline_basis_labels": ["physical_arrival", "host_confirmation"],
                "prediction_horizon_steps": args.horizon_steps, "shared_exogenous_randomness": True,
                "splits": {}, "generation": vars(args) | {"out": str(args.out)}}
    for split, seed in (("train", args.base_seed), ("validation", args.base_seed + 1000)):
        rows = make_split(split, args.count_per_split, seed, args.prefix_steps, args.horizon_steps, args.max_candidates_per_prefix)
        path = args.out / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
        manifest["splits"][split] = {"path": path.name, "records": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                      "parents": len({row["parent_episode_id"] for row in rows}),
                                      "prefixes": len({(row["parent_episode_id"], row["prefix_id"]) for row in rows})}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "splits": manifest["splits"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
