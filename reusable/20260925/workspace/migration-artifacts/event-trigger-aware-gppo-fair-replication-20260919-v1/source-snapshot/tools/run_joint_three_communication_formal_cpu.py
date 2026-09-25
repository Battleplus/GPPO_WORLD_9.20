"""Run the preregistered A-D x I/W1/W2 CPU matrix.

This is an orchestration entry point, not a new training method.  It freezes
the three communication conditions and the parent-tape identities before any
optimizer update, then runs each group/seed from a fresh initialization.  A
separate evaluator reads the final transactions only after training freezes.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import argparse
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.joint_gppo import JOINT_PROTOCOL, ActionConditionedTemporalWorldModel, JointGraphPreferencePolicy, JointTrainConfig
from gppo_world.joint_training import GROUPS, run_group
from gppo_world.m10_environment import M10Config, M10Environment, formal_three_condition_tape, scenario_to_dict
from tools.run_m10_baseline_comparison import traditional_action


RUN_NAME = "joint-four-module-three-communication-formal-cpu-20260916-v1"
CONDITIONS = ("I", "W1", "W2")
GROUP_ORDER = ("A", "B", "C", "D")
SEEDS = (1101, 2203, 3307)
SPLITS = {"train": (128, 851001), "validation": (32, 851002), "final_test": (64, 851003)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def interleave(parts: list[tuple[object, ...]]) -> tuple[object, ...]:
    result = []
    for index in range(max(map(len, parts))):
        for part in parts:
            if index < len(part):
                result.append(part[index])
    return tuple(result)


def build_tapes(root: Path) -> dict[str, dict[str, tuple[object, ...]]]:
    tapes: dict[str, dict[str, tuple[object, ...]]] = {}
    for split, (count, base_seed) in SPLITS.items():
        tapes[split] = {
            condition: formal_three_condition_tape(
                split, count=count, base_seed=base_seed, condition=condition,
            )
            for condition in CONDITIONS
        }
    serialized = {
        split: {condition: [scenario_to_dict(item) for item in scenarios]
                for condition, scenarios in conditions.items()}
        for split, conditions in tapes.items()
    }
    write_json(root / "formal-tapes.json", serialized)
    parent_rows = {}
    for split, conditions in tapes.items():
        reference = conditions["I"]
        parent_rows[split] = []
        for index, base in enumerate(reference):
            parent_id = f"{split}|parent-index-{index:04d}|seed-{base.seed}"
            parent_rows[split].append({
                "parent_id": parent_id,
                "base_seed": base.seed,
                "condition_tape_ids": {condition: conditions[condition][index].tape_id for condition in CONDITIONS},
                "task_content_sha256": hashlib.sha256(json.dumps(
                    [asdict(task) for task in base.tasks], sort_keys=True, separators=(",", ":")
                ).encode("utf-8")).hexdigest(),
            })
    write_json(root / "parent-tape-index.json", parent_rows)
    all_parent_ids = [row["parent_id"] for rows in parent_rows.values() for row in rows]
    if len(all_parent_ids) != len(set(all_parent_ids)):
        raise RuntimeError("formal parent tape identities overlap")
    condition_specs = {
        "I": {"name": "ideal", "telemetry_extra_delay": 0.0, "telemetry_loss_probability": 0.0,
               "command_loss_probability": 0.0, "ack_loss_probability": 0.0,
               "disconnect": "base mixed tape event schedule"},
        "W1": {"name": "light", "telemetry_extra_delay": 1.0, "telemetry_loss_probability": 0.05,
                "command_loss_probability": 0.02, "ack_loss_probability": 0.02,
                "disconnect": "one addressed UAV, start uniformly in [20%,40%] of 18.0, duration 2.0"},
        "W2": {"name": "moderate", "telemetry_extra_delay": 2.0, "telemetry_loss_probability": 0.15,
                "command_loss_probability": 0.05, "ack_loss_probability": 0.05,
                "disconnect": "one addressed UAV, start uniformly in [20%,40%] of 18.0, duration 4.0"},
    }
    write_json(root / "generation-audit.json", {
        "schema": "world-gppo-9.11-three-communication-formal-tapes/1.0.0",
        "split_counts": {split: count for split, (count, _) in SPLITS.items()},
        "generation_seeds": {split: seed for split, (_, seed) in SPLITS.items()},
        "conditions": condition_specs,
        "task_protocol": {"task_completion_mode": "arrival_to_region", "deadline_basis": "physical_arrival", "horizon": 18.0, "decision_interval": 1.0},
        "same_parent_across_conditions": True,
        "cross_split_parent_identity_disjoint": True,
        "historical_index_coverage": "Only source/runs manifests present in this deployment are auditable; unavailable historical raw tape bytes are not claimed deduplicated.",
        "selection": "No result-, failure-, or success-based filtering; all generated parents retained.",
        "formal_tapes_sha256": sha256(root / "formal-tapes.json"),
    })
    return tapes


def make_initialization(root: Path, seed: int, env_config: M10Config) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    policy = JointGraphPreferencePolicy(env_config, history=True)
    world = ActionConditionedTemporalWorldModel()
    action_rng_seed = seed + 0x4A11CE
    torch.manual_seed(action_rng_seed)
    action_rng = torch.get_rng_state().clone()
    payload = {
        "seed": seed,
        "policy_state_dict": policy.state_dict(),
        "world_state_dict": world.state_dict(),
        "action_rng_state": action_rng,
        "protocol": JOINT_PROTOCOL,
        "policy_parameter_count": sum(x.numel() for x in policy.parameters()),
        "world_parameter_count": sum(x.numel() for x in world.parameters()),
    }
    path = root / "initialization" / f"seed-{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {"path": str(path.relative_to(root)), "sha256": sha256(path), "policy_parameter_count": payload["policy_parameter_count"], "world_parameter_count": payload["world_parameter_count"], "action_rng_seed": action_rng_seed}


def micro_check(root: Path, tapes: dict[str, dict[str, tuple[object, ...]]], env_config: M10Config) -> dict[str, object]:
    check_root = root / "communication-check"
    check_root.mkdir(parents=True, exist_ok=False)
    rows = {}
    for condition in CONDITIONS:
        out = check_root / condition
        out.mkdir()
        steps = 0
        terminal_rows = []
        for scenario in tapes["validation"][condition][:1]:
            env = M10Environment(env_config, scenario)
            obs = env.reset()
            done = False
            info = {}
            while not done and steps < 256:
                action = int(traditional_action(obs))
                obs, _, done, info = env.step(action)
                steps += 1
            terminal_rows.append({"tape_id": scenario.tape_id, "steps": steps, "done": bool(done), "counts": info.get("counts", {}), "communication_entries": len(getattr(env, "_communication_log", []))})
        write_json(out / "resolved-config.json", {"run_id": f"formal-communication-check-{condition}-v1", "condition": condition, "updates": 0, "max_environment_steps_total": 256, "implementation": "direct environment/message-path check; no training runner"})
        rows[condition] = {"environment_steps": steps, "policy_updates": 0, "world_updates": 0, "status": "completed", "terminal_rows": terminal_rows}
    write_json(check_root / "summary.json", {"status": "completed", "rows": rows, "scope": "no optimizer updates; communication path/terminal-state smoke only"})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--run-name", default=RUN_NAME)
    args = parser.parse_args()
    root = ROOT / "runs" / args.run_name
    if root.exists():
        raise SystemExit(f"refusing to overwrite existing formal run root: {root}")
    root.mkdir(parents=True)
    env_config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    tapes = build_tapes(root)
    init = {str(seed): make_initialization(root, seed, env_config) for seed in SEEDS}
    write_json(root / "protocol-and-budget.json", {
        "protocol": JOINT_PROTOCOL, "env_config": asdict(env_config),
        "device": args.device, "threads": {"intra_op": 4, "inter_op": 1},
        "groups": {key: asdict(GROUPS[key]) for key in GROUP_ORDER},
        "conditions": CONDITIONS, "seeds": SEEDS,
        "budget_per_run": {"environment_steps": 16384, "policy_optimizer_steps": 128, "world_optimizer_steps": {"A": 0, "B": 0, "C": 1024, "D": 1024}, "wall_seconds": 2400},
        "total_budget": {"environment_steps": 196608, "policy_optimizer_steps": 1536, "world_optimizer_steps": 6144, "training_wall_seconds": 28800},
        "final_checkpoint_policy": "fixed final 16384-step transaction; no validation selection",
        "source_entry_sha256": sha256(Path(__file__)), "initializations": init,
        "status": "pre_registered_before_training",
    })
    check = micro_check(root, tapes, env_config)
    write_json(root / "run-status.json", {"status": "micro_check_completed", "micro_check": check, "training_started": False})
    train_lists = {condition: tapes["train"][condition] for condition in CONDITIONS}
    # Interleave conditions so the fixed training tape is balanced at rollout scale.
    train_scenarios = interleave([tuple(train_lists[condition]) for condition in CONDITIONS])
    summaries = {}
    started = time.perf_counter()
    for seed in SEEDS:
        initialization = torch.load(root / "initialization" / f"seed-{seed}.pt", map_location="cpu", weights_only=False)
        for condition in CONDITIONS:
            for group in GROUP_ORDER:
                out = root / "training" / f"seed-{seed}" / condition / group
                out.mkdir(parents=True, exist_ok=False)
                run_id = f"formal-{condition}-{group}-seed-{seed}-v1"
                write_json(out / "resolved-config.json", {
                    "run_id": run_id, "group": group, "seed": seed, "condition": condition,
                    "protocol": JOINT_PROTOCOL, "environment": asdict(env_config),
                    "training": asdict(JointTrainConfig(seed=seed, rollout_steps=128)),
                    "initialization_sha256": init[str(seed)]["sha256"],
                    "training_tape_sha256": sha256(root / "formal-tapes.json"),
                    "balanced_condition_sampling": True,
                    "final_test_frozen_before_read": True,
                })
                max_world = 1024 if group in ("C", "D") else 0
                initial_policy = initialization["policy_state_dict"] if group in ("B", "C", "D") else None
                initial_world = initialization["world_state_dict"] if group in ("C", "D") else None
                summary = run_group(
                    group=group, run_id=run_id, output_dir=out, device=args.device, env_config=env_config,
                    train_config=JointTrainConfig(seed=seed, rollout_steps=128), max_steps=16384,
                    max_policy_updates=128, max_world_updates=max_world, wall_seconds=2400.0,
                    enforce_smoke_limits=False, verify_recovery_update=False, scenarios=train_scenarios,
                    initial_policy_state_dict=initial_policy, initial_world_state_dict=initial_world,
                    initial_action_rng_state=initialization["action_rng_state"] if group in ("B", "C", "D") else None,
                )
                stop_reason = summary.get("stop_reason") or summary.get("metadata", {}).get("stop_reason")
                if stop_reason is None and int(summary.get("environment_steps", 0)) >= 16384:
                    stop_reason = "environment_step_budget"
                wall_seconds = summary.get("wall_seconds") or summary.get("metadata", {}).get("wall_seconds")
                summaries[f"{seed}/{condition}/{group}"] = {
                    "status": summary.get("status"), "environment_steps": summary.get("environment_steps"),
                    "policy_optimizer_steps": summary.get("policy_optimizer_steps"), "world_optimizer_steps": summary.get("world_optimizer_steps"),
                    "stop_reason": stop_reason, "wall_seconds": wall_seconds,
                    "run_dir": str(out.relative_to(root)),
                }
                write_json(out / "formal-group-summary.json", summaries[f"{seed}/{condition}/{group}"])
                write_json(root / "training-progress.json", {"completed_runs": summaries, "next": "continue in fixed seed/condition/group order", "elapsed_seconds": time.perf_counter() - started})
    write_json(root / "training-summary.json", {"status": "completed", "runs": summaries, "actual_total_wall_seconds": time.perf_counter() - started})
    print(json.dumps({"run_root": str(root), "status": "training_completed", "runs": len(summaries)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
