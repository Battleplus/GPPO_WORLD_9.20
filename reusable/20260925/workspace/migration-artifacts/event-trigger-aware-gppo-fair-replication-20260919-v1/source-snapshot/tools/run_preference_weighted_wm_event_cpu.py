"""Run the bounded WB/WC/WD re-integration validation on frozen tapes.

WB is the previously validated per-transition preference-weighted PPO arm.
WC adds the action-conditioned temporal world model with the historical
shared-gradient semantics, while WD additionally enables event features and
the event auxiliary loss.  This runner deliberately never reads final-test.

The correctness phase is a separate run directory and is completed before
the nine formal runs.  Validation curves are replayed after each immutable
4096-step checkpoint; they are recorded only and never used for selection.
"""

from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.joint_gppo import JOINT_PROTOCOL, JointTrainConfig
from gppo_world.joint_training import GROUPS, run_group
from gppo_world.m10_environment import M10Config, scenario_from_dict
from tools.evaluate_joint_four_group_pilot import _episode, _load_policy, _summarize


SEEDS = (1101, 2203, 3307)
GROUPS_TO_RUN = ("WB", "WC", "WD")
CONDITIONS = ("I", "W1", "W2")
PREFERENCES = ((0.2, 0.8), (0.5, 0.5), (0.8, 0.2))
EXPECTED_FORMAL_TAPES_SHA256 = "5d1cfaed3683e3e1c512f4ff67d1f93566dae2b418942b67df3f1c81cb5eaea7"
# The value above is intentionally checked against the supplied frozen tape
# file.  A typo in the expected identity must stop the run rather than permit
# an accidental tape substitution.


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
    result: list[object] = []
    for index in range(max(map(len, parts))):
        for part in parts:
            if index < len(part):
                result.append(part[index])
    return tuple(result)


def load_frozen_tapes(formal_root: Path):
    tape_path = formal_root / "formal-tapes.json"
    actual = sha256(tape_path)
    if actual != EXPECTED_FORMAL_TAPES_SHA256:
        raise SystemExit(f"frozen formal-tapes hash mismatch: {actual}")
    payload = json.loads(tape_path.read_text(encoding="utf-8"))
    required = {"train": 128, "validation": 32}
    for split, count in required.items():
        if split not in payload or any(len(payload[split][condition]) != count for condition in CONDITIONS):
            raise SystemExit(f"frozen tape count mismatch in {split}")
    # Explicitly do not load final_test.  Keeping the field out of this object
    # makes accidental final-test use a code-level failure.
    tapes = {
        split: {condition: tuple(scenario_from_dict(item) for item in payload[split][condition])
                for condition in CONDITIONS}
        for split in required
    }
    return tapes, actual


def checkpoint_at(run_dir: Path, step: int) -> Path:
    matches = sorted((run_dir / "transactions").glob(f"txn-step-{step:08d}-policy-*-world-*.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one immutable checkpoint at {run_dir} step {step}, found {len(matches)}")
    return matches[0]


def evaluate_node(*, run_dir: Path, group: str, step: int, validation, env_config: M10Config,
                  train_config: JointTrainConfig) -> dict[str, object]:
    checkpoint = checkpoint_at(run_dir, step)
    device = torch.device("cpu")
    policy, world = _load_policy(group, checkpoint, env_config, device)
    preferences: dict[str, object] = {}
    for preference in PREFERENCES:
        key = "(" + ",".join(f"{value:.1f}" for value in preference) + ")"
        rows = [_episode(group, policy, world, scenario, env_config, train_config, device, preference)
                for scenario in validation]
        preferences[key] = {
            "preference": preference,
            "summary": _summarize(rows),
            "episodes": rows,
        }
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "preferences": preferences,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--wall-seconds", type=float, default=6 * 3600)
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()
    if args.wall_seconds <= 0 or args.wall_seconds > 6 * 3600:
        raise SystemExit("invalid total wall-clock budget")
    formal_root = args.formal_root.resolve()
    output_root = args.out.resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output_root}")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    tapes, tape_hash = load_frozen_tapes(formal_root)
    train_scenarios = interleave([tapes["train"][condition] for condition in CONDITIONS])
    validation = tuple(item for condition in CONDITIONS for item in tapes["validation"][condition])
    env_config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    train_config_template = JointTrainConfig(rollout_steps=128)
    output_root.mkdir(parents=True, exist_ok=False)
    write_json(output_root / "protocol-and-budget.json", {
        "schema": "joint-preference-weighted-wm-event-cpu/1.0.0",
        "protocol": JOINT_PROTOCOL,
        "groups": {group: asdict(GROUPS[group]) for group in GROUPS_TO_RUN},
        "seeds": SEEDS, "conditions": CONDITIONS,
        "formal_root": str(formal_root), "formal_tapes_sha256": tape_hash,
        "train_schedule": "fixed I/W1/W2 interleave; condition is not an extra training multiplier",
        "validation": "fixed train/validation tapes only; replay at 4096-step nodes; no final-test read",
        "environment": asdict(env_config),
        "budget_per_run": {"environment_steps": 16384, "rollout_steps": 128,
                            "policy_updates": 128, "world_updates": {"WB": 0, "WC": 1024, "WD": 1024},
                            "wall_seconds": 40 * 60},
        "total_budget": {"environment_steps": 147456, "policy_updates": 1152,
                          "world_updates": 6144, "wall_seconds": 6 * 3600},
        "correctness_budget": {"environment_steps": 384, "policy_updates": 12,
                               "world_updates": 32, "wall_seconds": 20 * 60},
        "WB_definition": "W standard clipped PPO actor with per-sample A_scalar=sum_i p_i A_i",
        "WC_definition": "WB plus action-conditioned temporal world model, event input/loss disabled",
        "WD_definition": "WC plus event feature consumption and event auxiliary loss",
        "final_test_read": False,
        "runtime": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                    "numpy": np.__version__, "threads": {"intra_op": 4, "inter_op": 1}},
        "status": "pre_registered_before_training",
    })
    write_json(output_root / "run-status.json", {"status": "pre_registered", "training_started": False})
    summaries: dict[str, object] = {}
    validation_curves: dict[str, object] = {}
    started = time.perf_counter()

    if not args.skip_correctness:
        correctness_root = output_root / "correctness"
        correctness_root.mkdir(parents=True, exist_ok=False)
        correctness_results: dict[str, object] = {}
        for index, group in enumerate(GROUPS_TO_RUN):
            seed = SEEDS[0]
            run_dir = correctness_root / group
            # 128 steps per group gives 384 total correctness transitions;
            # world groups receive 8 updates each, within the registered cap.
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            init_path = formal_root / "initialization" / f"seed-{seed}.pt"
            init = torch.load(init_path, map_location="cpu", weights_only=False)
            summary = run_group(
                group=group, run_id=f"weighted-wm-event-correctness-{group}-seed-{seed}-v1",
                output_dir=run_dir, device=args.device, env_config=env_config,
                train_config=JointTrainConfig(seed=seed, rollout_steps=128),
                max_steps=128, max_policy_updates=1, max_world_updates=0 if group == "WB" else 8,
                wall_seconds=20 * 60, enforce_smoke_limits=True, verify_recovery_update=True,
                scenarios=train_scenarios, initial_policy_state_dict=init["policy_state_dict"],
                initial_action_rng_state=init["action_rng_state"],
            )
            correctness_results[group] = summary
        write_json(output_root / "correctness-summary.json", {
            "status": "passed", "results": correctness_results,
            "scope": "one 128-step run per group; separate from formal training",
        })
        write_json(output_root / "run-status.json", {"status": "correctness_passed", "training_started": False})

    for seed in SEEDS:
        init_path = formal_root / "initialization" / f"seed-{seed}.pt"
        init = torch.load(init_path, map_location="cpu", weights_only=False)
        base_policy = init["policy_state_dict"]
        action_rng_state = init["action_rng_state"]
        for group in GROUPS_TO_RUN:
            run_key = f"{seed}/{group}"
            run_dir = output_root / "training" / f"seed-{seed}" / group
            write_json(run_dir / "resolved-config.json", {
                "run_id": f"weighted-wm-event-{group}-seed-{seed}-v1", "group": group, "seed": seed,
                "protocol": JOINT_PROTOCOL, "environment": asdict(env_config),
                "training": asdict(JointTrainConfig(seed=seed, rollout_steps=128)),
                "initialization_source": str(init_path), "initialization_sha256": sha256(init_path),
                "formal_tapes_sha256": tape_hash, "condition_schedule": CONDITIONS,
                "validation_replay_nodes": (4096, 8192, 12288, 16384),
                "final_test_read": False,
                "group_semantics": asdict(GROUPS[group]),
            })
            summary = run_group(
                group=group, run_id=f"weighted-wm-event-{group}-seed-{seed}-v1", output_dir=run_dir,
                device=args.device, env_config=env_config,
                train_config=JointTrainConfig(seed=seed, rollout_steps=128),
                max_steps=16384, max_policy_updates=128,
                max_world_updates=0 if group == "WB" else 1024,
                wall_seconds=40 * 60, enforce_smoke_limits=False,
                verify_recovery_update=False, scenarios=train_scenarios,
                initial_policy_state_dict=base_policy, initial_action_rng_state=action_rng_state,
            )
            summaries[run_key] = {
                "status": summary.get("status"), "environment_steps": summary.get("environment_steps"),
                "policy_optimizer_steps": summary.get("policy_optimizer_steps"),
                "world_optimizer_steps": summary.get("world_optimizer_steps"),
                "stop_reason": summary.get("stop_reason"), "wall_seconds": summary.get("wall_seconds"),
                "run_dir": str(run_dir.relative_to(output_root)),
                "checkpoint_sha256": summary.get("checkpoint_sha256"),
            }
            write_json(run_dir / "weighted-group-summary.json", summaries[run_key])
            write_json(output_root / "training-progress.json", {
                "status": "running", "completed_runs": summaries,
                "elapsed_seconds": time.perf_counter() - started,
            })
            # Replay the immutable nodes after this run.  This is read-only,
            # uses only validation, and cannot influence the next run.
            nodes: dict[str, object] = {}
            for step in (4096, 8192, 12288, 16384):
                nodes[str(step)] = evaluate_node(
                    run_dir=run_dir, group=group, step=step, validation=validation,
                    env_config=env_config, train_config=JointTrainConfig(seed=seed, rollout_steps=128),
                )
            validation_curves[run_key] = nodes
            write_json(output_root / "validation-curves.json", validation_curves)
            if time.perf_counter() - started >= args.wall_seconds:
                write_json(output_root / "run-status.json", {"status": "stopped_at_total_wall_clock", "completed_runs": summaries})
                raise SystemExit("total wall-clock budget reached")

    write_json(output_root / "training-summary.json", {
        "status": "completed", "runs": summaries,
        "actual_total_wall_seconds": time.perf_counter() - started,
        "final_test_read": False,
    })
    write_json(output_root / "run-status.json", {
        "status": "completed", "completed_runs": len(summaries),
        "actual_total_wall_seconds": time.perf_counter() - started,
        "final_test_read": False,
    })
    print(json.dumps({"run_root": str(output_root), "status": "completed", "runs": len(summaries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
