"""Generate grouped M-10 Graph-5 candidate-consequence supervision.

This is a simulator-side label generator, not a policy or training command.
Every candidate branch starts from the same public prefix and receives the
same ``exogenous_key``.  Hidden environment state is read only after the
branch for labels and is never serialized as a model input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gppo_world.consequence_data import record_identity_sha256  # noqa: E402
from gppo_world.graph5 import graph5_from_m10_observation  # noqa: E402
from gppo_world.m10_environment import M10Environment, M10Scenario, scenario_tape, weak_communication_tape  # noqa: E402


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    path.write_bytes(text.encode("utf-8"))
    sha = hashlib.sha256(text.encode()).hexdigest()
    return {"path": path.name, "sha256": sha, "records": len(rows), "identity_sha256": record_identity_sha256(rows)}


def state(env: M10Environment) -> tuple[float, dict[str, float], dict[str, str]]:
    return (
        float(env.clock.time),
        {task_id: float(task.service) for task_id, task in env.clock.tasks.items()},
        {task_id: task.state.value for task_id, task in env.clock.tasks.items()},
    )


def branch(scenario: M10Scenario, prefix_actions: list[int], action: int, horizon: int, parent_id: str, prefix_id: str, exogenous_key: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one branch from a prefix under the caller-selected random stream."""

    exogenous_key = exogenous_key or f"{scenario.tape_id}|{parent_id}|{prefix_id}"
    env = M10Environment(scenario=scenario, exogenous_key=exogenous_key)
    obs = env.reset()
    prefix_trace: list[dict[str, Any]] = []
    for prefix_action in prefix_actions:
        obs, reward, done, info = env.step(prefix_action)
        prefix_trace.append({"action": prefix_action, "reward": reward, "done": done, "info": info})
        if done:
            raise RuntimeError("prefix reached terminal state; choose a shorter fixed prefix")
    graph = graph5_from_m10_observation(obs).as_dict()
    before_time, before_service, _ = state(env)
    before_energy = sum(float(resource.energy) for resource in env.clock.resources.values())
    trace = list(prefix_trace)
    for step_index in range(horizon):
        selected = action if step_index == 0 else env.config.action_count - 1
        obs, reward, done, info = env.step(selected)
        trace.append({"action": selected, "reward": reward, "done": done, "info": info})
        if done and step_index + 1 < horizon:
            break
    after_time, after_service, after_states = state(env)
    after_energy = sum(float(resource.energy) for resource in env.clock.resources.values())
    task_index = action % env.config.task_capacity
    task_id = f"task-{task_index}"
    uav_id = f"uav-{action // env.config.task_capacity}" if action < env.config.action_count - 1 else None
    branch_travel = [item for item in env.clock.log if item.get("kind") == "travel" and item.get("task") == task_id and item.get("resource") == uav_id]
    travel_time = float(sum(float(item["end"]) - float(item["start"]) for item in branch_travel))
    deadline = float(env.clock.tasks[task_id].deadline)
    deadline_observed = after_time >= deadline
    arrived = bool(branch_travel) and after_time >= float(branch_travel[-1]["end"])
    record = {
        "parent_episode_id": parent_id,
        "prefix_id": prefix_id,
        "graph5_t": graph,
        "target": {
            "episode_id": parent_id,
            "decision_index": len(prefix_actions),
            "action": action,
            "horizon_steps": horizon,
            "exogenous_key": exogenous_key,
            "travel_time": travel_time,
            "service_progress": float(after_service[task_id] - before_service[task_id]),
            "energy_delta": float(before_energy - after_energy),
            "deadline_risk": float(after_states[task_id] != "completed"),
            "source": "simulator-counterfactual",
            "hidden_state_used_for_label": True,
        },
        "label_masks": {
            "travel_time": arrived,
            "service_progress": True,
            "energy_delta": True,
            "deadline_risk": deadline_observed,
        },
        "label_provenance": {
            "source": "M10Environment.simulator-counterfactual",
            "shared_exogenous_key": exogenous_key,
            "hidden_state_online": False,
            "branch_trace_sha256": digest(trace),
            "prefix_trace_sha256": digest(prefix_trace),
            "task_id": task_id,
            "uav_id": uav_id,
            "deadline_observed_through": after_time,
        },
    }
    ledger = {"parent_episode_id": parent_id, "prefix_id": prefix_id, "action": action, "exogenous_key": exogenous_key, "trace": trace}
    return record, ledger


def make_split(split: str, count: int, base_seed: int, prefix_steps: int, horizon: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scenarios = weak_communication_tape(split, count=count, base_seed=base_seed, level="composite")
    rows: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for scenario in scenarios:
        parent_id = f"{split}:{scenario.tape_id}"
        prefix_id = f"{parent_id}:prefix-{prefix_steps}"
        prefix_actions = [M10Environment(scenario=scenario).config.action_count - 1] * prefix_steps
        shared_key = f"{scenario.tape_id}|{parent_id}|{prefix_id}"
        probe = M10Environment(scenario=scenario, exogenous_key=shared_key)
        obs = probe.reset()
        for prefix_action in prefix_actions:
            obs, _, done, _ = probe.step(prefix_action)
            if done:
                raise RuntimeError("fixed prefix terminated before candidate branching")
        legal_actions = [index for index, allowed in enumerate(np.asarray(obs["mask"], dtype=bool)) if allowed]
        for action in legal_actions:
            row, branch_ledger = branch(scenario, prefix_actions, action, horizon, parent_id, prefix_id, shared_key)
            rows.append(row)
            ledger.append(branch_ledger)
    return rows, ledger


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count-per-split", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=91011)
    parser.add_argument("--prefix-steps", type=int, default=2)
    parser.add_argument("--horizon-steps", type=int, default=1)
    parser.add_argument("--protocol", default="world-gppo-9.11-consequence/0.1.0")
    args = parser.parse_args()
    if min(args.count_per_split, args.prefix_steps, args.horizon_steps) < 1:
        parser.error("count-per-split, prefix-steps, and horizon-steps must be positive")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    ledgers: dict[str, Any] = {}
    for split in ("train", "validation", "test", "ood"):
        rows, ledger = make_split(split, args.count_per_split, args.base_seed, args.prefix_steps, args.horizon_steps)
        files[split] = dump_jsonl(out / f"{split}.jsonl", rows)
        ledger_path = out / f"{split}.branches.jsonl"
        ledger_text = "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in ledger)
        ledger_path.write_bytes(ledger_text.encode("utf-8"))
        ledgers[split] = {"path": ledger_path.name, "sha256": hashlib.sha256(ledger_text.encode()).hexdigest(), "records": len(ledger)}
    manifest = {"schema": "gppo-consequence-dataset/v2", "protocol": args.protocol, "observation_contract": "m10-graph5-5type-25action", "prediction_horizon_steps": args.horizon_steps, "files": files, "branch_ledgers": ledgers, "generation": {"generator": "tools/generate_m10_consequence_dataset.py", "prefix_steps": args.prefix_steps, "count_per_split": args.count_per_split, "base_seed": args.base_seed, "shared_exogenous_randomness": True, "hidden_state_online": False}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
