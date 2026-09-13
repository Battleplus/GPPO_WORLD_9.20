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
from typing import Any, Sequence

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


def compact_transition(action: int, reward: float, done: bool, info: dict[str, Any]) -> dict[str, Any]:
    """Keep replay-relevant deltas without duplicating cumulative logs each step."""

    return {
        "action": action,
        "reward": reward,
        "done": done,
        "info": {
            key: info[key]
            for key in (
                "feedback",
                "command_submitted",
                "command_id",
                "lease_renewal",
                "lease_renewals",
                "lease_renewal_delivery_results",
                "active_continuations",
                "step",
                "time",
                "counts",
                "energy",
                "tasks",
                "task_service",
                "new_events",
                "communication_delta",
                "policy_version",
                "trigger_flags",
                "terminated",
                "truncated",
                "episode_end_reason",
            )
        },
    }


def branch(scenario: M10Scenario, prefix_actions: list[int], action: int, horizon: int, parent_id: str, prefix_id: str, exogenous_key: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one branch from a prefix under the caller-selected random stream."""

    exogenous_key = exogenous_key or f"{scenario.tape_id}|{parent_id}|{prefix_id}"
    env = M10Environment(scenario=scenario, exogenous_key=exogenous_key)
    obs = env.reset()
    prefix_trace: list[dict[str, Any]] = []
    for prefix_action in prefix_actions:
        obs, reward, done, info = env.step(prefix_action)
        prefix_trace.append(compact_transition(prefix_action, reward, done, info))
        if done:
            raise RuntimeError("prefix reached terminal state; choose a shorter fixed prefix")
    graph = graph5_from_m10_observation(obs).as_dict()
    before_time, before_service, _ = state(env)
    before_energy = sum(float(resource.energy) for resource in env.clock.resources.values())
    trace = list(prefix_trace)
    for step_index in range(horizon):
        selected = action if step_index == 0 else env.config.action_count - 1
        obs, reward, done, info = env.step(selected)
        trace.append(compact_transition(selected, reward, done, info))
        if done and step_index + 1 < horizon:
            break
    after_time, after_service, after_states = state(env)
    after_energy = sum(float(resource.energy) for resource in env.clock.resources.values())
    is_noop = action == env.config.action_count - 1
    task_index = None if is_noop else action % env.config.task_capacity
    task_id = None if task_index is None else f"task-{task_index}"
    uav_id = None if is_noop else f"uav-{action // env.config.task_capacity}"
    branch_travel = [item for item in env.clock.log if item.get("kind") == "travel" and item.get("task") == task_id and item.get("resource") == uav_id]
    travel_time = float(sum(float(item["end"]) - float(item["start"]) for item in branch_travel))
    deadline = None if task_id is None else float(env.clock.tasks[task_id].deadline)
    deadline_observed = deadline is not None and after_time >= deadline
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
            "service_progress": 0.0 if task_id is None else float(after_service[task_id] - before_service[task_id]),
            "energy_delta": float(before_energy - after_energy),
            "deadline_risk": 0.0 if task_id is None else float(after_states[task_id] != "completed"),
            "source": "simulator-counterfactual",
            "hidden_state_used_for_label": True,
        },
        "label_masks": {
            "travel_time": arrived,
            "service_progress": not is_noop,
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
            "outcome_scope": "system-energy-only" if is_noop else "selected-uav-task",
            "deadline_observed_through": after_time,
        },
    }
    ledger = {
        "parent_episode_id": parent_id,
        "prefix_id": prefix_id,
        "action": action,
        "exogenous_key": exogenous_key,
        "prefix_steps": len(prefix_actions),
        "prefix_actions": prefix_actions,
        "prefix_trace_sha256": digest(prefix_trace),
        "trace": trace,
    }
    return record, ledger


def make_split(
    split: str,
    count: int,
    base_seed: int,
    prefix_steps: int | Sequence[int],
    horizon: int,
    *,
    max_candidates_per_prefix: int = 25,
    prefix_policy: str = "noop",
    scenario_name: str = "mixed",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prefixes = [prefix_steps] if isinstance(prefix_steps, int) else [int(value) for value in prefix_steps]
    if not prefixes or min(prefixes) < 1 or len(prefixes) != len(set(prefixes)):
        raise ValueError("prefix_steps must contain unique positive values")
    if max_candidates_per_prefix < 1:
        raise ValueError("max_candidates_per_prefix must be positive")
    if prefix_policy not in {"noop", "public-hash-legal"}:
        raise ValueError("unknown prefix policy")
    scenarios = weak_communication_tape(split, count=count, base_seed=base_seed, level="composite", name=scenario_name)
    rows: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for scenario in scenarios:
        parent_id = f"{split}:{scenario.tape_id}"
        for prefix_count in prefixes:
            prefix_id = f"{parent_id}:prefix-{prefix_count}-{prefix_policy}"
            shared_key = f"{scenario.tape_id}|{parent_id}|{prefix_id}"
            probe = M10Environment(scenario=scenario, exogenous_key=shared_key)
            obs = probe.reset()
            prefix_actions: list[int] = []
            for step_index in range(prefix_count):
                legal = [index for index, allowed in enumerate(np.asarray(obs["mask"], dtype=bool)) if allowed]
                if prefix_policy == "noop":
                    prefix_action = probe.config.action_count - 1
                else:
                    selector = digest({"tape_id": scenario.tape_id, "prefix_steps": prefix_count, "step": step_index})
                    prefix_action = legal[int(selector[:16], 16) % len(legal)]
                prefix_actions.append(prefix_action)
                obs, _, done, _ = probe.step(prefix_action)
                if done:
                    raise RuntimeError("fixed prefix terminated before candidate branching")
            legal_actions = [index for index, allowed in enumerate(np.asarray(obs["mask"], dtype=bool)) if allowed]
            if len(legal_actions) > max_candidates_per_prefix:
                legal_actions = sorted(
                    legal_actions,
                    key=lambda action: digest({"parent": parent_id, "prefix": prefix_id, "action": action}),
                )[:max_candidates_per_prefix]
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
    parser.add_argument("--prefix-steps", type=int, nargs="+", default=[2])
    parser.add_argument("--horizon-steps", type=int, default=1)
    parser.add_argument("--max-candidates-per-prefix", type=int, default=25)
    parser.add_argument("--prefix-policy", choices=("noop", "public-hash-legal"), default="noop")
    parser.add_argument("--ood-scenario", choices=("mixed", "energy_insufficient", "uav_damage", "communication_interrupt"), default="mixed")
    parser.add_argument("--protocol", default="world-gppo-9.11-consequence/0.1.0")
    args = parser.parse_args()
    if min(args.count_per_split, min(args.prefix_steps), args.horizon_steps, args.max_candidates_per_prefix) < 1:
        parser.error("count-per-split, prefix-steps, and horizon-steps must be positive")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    ledgers: dict[str, Any] = {}
    for split in ("train", "validation", "test", "ood"):
        scenario_name = args.ood_scenario if split == "ood" else "mixed"
        rows, ledger = make_split(
            split,
            args.count_per_split,
            args.base_seed,
            args.prefix_steps,
            args.horizon_steps,
            max_candidates_per_prefix=args.max_candidates_per_prefix,
            prefix_policy=args.prefix_policy,
            scenario_name=scenario_name,
        )
        files[split] = dump_jsonl(out / f"{split}.jsonl", rows)
        ledger_path = out / f"{split}.branches.jsonl"
        ledger_text = "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in ledger)
        ledger_path.write_bytes(ledger_text.encode("utf-8"))
        ledgers[split] = {"path": ledger_path.name, "sha256": hashlib.sha256(ledger_text.encode()).hexdigest(), "records": len(ledger)}
    manifest = {
        "schema": "gppo-consequence-dataset/v2",
        "protocol": args.protocol,
        "observation_contract": "m10-graph5-5type-25action-global27",
        "prediction_horizon_steps": args.horizon_steps,
        "files": files,
        "branch_ledgers": ledgers,
        "generation": {
            "generator": "tools/generate_m10_consequence_dataset.py",
            "prefix_steps": args.prefix_steps,
            "prefix_policy": args.prefix_policy,
            "count_per_split": args.count_per_split,
            "max_candidates_per_prefix": args.max_candidates_per_prefix,
            "base_seed": args.base_seed,
            "in_distribution_scenario": "mixed",
            "ood_scenario": args.ood_scenario,
            "shared_exogenous_randomness": True,
            "hidden_state_online": False,
            "noop_labels": "task-specific travel/service/deadline masked; system energy observed",
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
