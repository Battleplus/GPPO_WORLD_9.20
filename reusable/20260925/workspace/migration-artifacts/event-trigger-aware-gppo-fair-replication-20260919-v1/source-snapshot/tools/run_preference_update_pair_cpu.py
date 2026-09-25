"""Run the bounded P/W preference-update localization experiment on the server CPU.

P is the existing PreCo-adapted preference update.  W keeps the same
preference-conditioned policy, vector critic, data and budgets, but uses the
standard clipped PPO actor surrogate with per-transition scalarized GAE.
This runner consumes the already frozen mixed I/W1/W2 training tapes and does
not read final-test results.
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


SEEDS = (1101, 2203, 3307)
GROUPS_TO_RUN = ("P", "W")
CONDITIONS = ("I", "W1", "W2")
EXPECTED_FORMAL_TAPES_SHA256 = "5d1cfaed3683e3e1c512f4ff67d1f93566dae2b418942b67df3f1c81cb5eaea7"


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


def load_frozen_tapes(formal_root: Path) -> tuple[dict[str, dict[str, tuple[object, ...]]], str]:
    tape_path = formal_root / "formal-tapes.json"
    actual_hash = sha256(tape_path)
    if actual_hash != EXPECTED_FORMAL_TAPES_SHA256:
        raise SystemExit(f"frozen formal-tapes hash mismatch: {actual_hash}")
    payload = json.loads(tape_path.read_text(encoding="utf-8"))
    tapes = {
        split: {
            condition: tuple(scenario_from_dict(item) for item in rows)
            for condition, rows in conditions.items()
        }
        for split, conditions in payload.items()
    }
    expected = {"train": 128, "validation": 32, "final_test": 64}
    for split, count in expected.items():
        if any(len(tapes[split][condition]) != count for condition in CONDITIONS):
            raise SystemExit(f"frozen tape count mismatch in {split}")
    return tapes, actual_hash


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--wall-seconds", type=float, default=2400.0)
    args = parser.parse_args()
    formal_root = args.formal_root.resolve()
    output_root = args.out.resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output_root}")
    if args.wall_seconds <= 0 or args.wall_seconds > 4 * 3600:
        raise SystemExit("invalid wall-clock budget")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    tapes, tapes_hash = load_frozen_tapes(formal_root)
    train_scenarios = interleave([tapes["train"][condition] for condition in CONDITIONS])
    env_config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    output_root.mkdir(parents=True, exist_ok=False)
    write_json(output_root / "protocol-and-budget.json", {
        "schema": "joint-preference-update-pair-cpu/1.0.0",
        "protocol": JOINT_PROTOCOL,
        "groups": {group: asdict(GROUPS[group]) for group in GROUPS_TO_RUN},
        "seeds": SEEDS, "conditions": CONDITIONS,
        "source_formal_root": str(formal_root), "formal_tapes_sha256": tapes_hash,
        "training_tape_schedule": "frozen mixed I/W1/W2 episode interleave",
        "environment": asdict(env_config),
        "budget_per_run": {"environment_steps": 16384, "rollout_steps": 128,
                            "policy_updates": 128, "world_updates": 0,
                            "wall_seconds": args.wall_seconds},
        "total_budget": {"environment_steps": 98304, "policy_updates": 768,
                          "world_updates": 0, "wall_seconds": 4 * 3600},
        "P_definition": "existing PreCo adaptation with vector-return similarity and local PPO surrogates",
        "W_definition": "standard clipped PPO actor with per-sample A_scalar=sum_i p_i A_i; same vector critic/GAE and scales",
        "final_test": "not run; old final-test is read-only historical evidence",
        "runtime": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__, "threads": {"intra_op": 4, "inter_op": 1}},
        "status": "pre_registered_before_training",
    })
    write_json(output_root / "run-status.json", {"status": "pre_registered", "training_started": False})
    summaries: dict[str, object] = {}
    started = time.perf_counter()
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # Reuse the already verified initialization only as a fresh, untrained
        # starting point; no final checkpoint or validation outcome is loaded.
        base_policy = None
        action_rng_state = None
        init_path = formal_root / "initialization" / f"seed-{seed}.pt"
        if init_path.is_file():
            init = torch.load(init_path, map_location="cpu", weights_only=False)
            base_policy = init["policy_state_dict"]
            action_rng_state = init["action_rng_state"]
            init_hash = sha256(init_path)
        else:
            init_hash = None
        for group in GROUPS_TO_RUN:
            run_key = f"{seed}/{group}"
            run_dir = output_root / "training" / f"seed-{seed}" / group
            write_json(run_dir / "resolved-config.json", {
                "run_id": f"preference-pair-{group}-seed-{seed}-v1", "group": group, "seed": seed,
                "protocol": JOINT_PROTOCOL, "environment": asdict(env_config),
                "training": asdict(JointTrainConfig(seed=seed, rollout_steps=128)),
                "initialization_source": str(init_path), "initialization_sha256": init_hash,
                "training_tape_sha256": tapes_hash, "condition_schedule": CONDITIONS,
                "final_test_read": False,
            })
            summary = run_group(
                group=group, run_id=f"preference-pair-{group}-seed-{seed}-v1", output_dir=run_dir,
                device="cpu", env_config=env_config,
                train_config=JointTrainConfig(seed=seed, rollout_steps=128),
                max_steps=16384, max_policy_updates=128, max_world_updates=0,
                wall_seconds=args.wall_seconds, enforce_smoke_limits=False,
                verify_recovery_update=False, scenarios=train_scenarios,
                initial_policy_state_dict=base_policy,
                initial_action_rng_state=action_rng_state,
            )
            summaries[run_key] = {
                "status": summary.get("status"), "environment_steps": summary.get("environment_steps"),
                "policy_optimizer_steps": summary.get("policy_optimizer_steps"),
                "world_optimizer_steps": summary.get("world_optimizer_steps"),
                "stop_reason": summary.get("stop_reason"), "wall_seconds": summary.get("wall_seconds"),
                "run_dir": str(run_dir.relative_to(output_root)),
            }
            write_json(run_dir / "pair-group-summary.json", summaries[run_key])
            write_json(output_root / "training-progress.json", {
                "status": "running", "completed_runs": summaries,
                "elapsed_seconds": time.perf_counter() - started,
            })
    write_json(output_root / "training-summary.json", {
        "status": "completed", "runs": summaries,
        "actual_total_wall_seconds": time.perf_counter() - started,
    })
    write_json(output_root / "run-status.json", {
        "status": "completed", "completed_runs": len(summaries),
        "actual_total_wall_seconds": time.perf_counter() - started,
    })
    print(json.dumps({"run_root": str(output_root), "status": "completed", "runs": len(summaries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
