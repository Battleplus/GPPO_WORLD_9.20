"""Run the corrected 12-run mixed-communication formal matrix.

Each A/B/C/D x seed run consumes one interleaved episode schedule containing
the frozen I/W1/W2 tape variants.  Communication conditions are strata of a
single 16,384-step run, not separate training runs.
"""

from __future__ import annotations

from dataclasses import asdict
import faulthandler
import hashlib
import json
from pathlib import Path
import argparse
import sys
import threading
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.joint_gppo import JOINT_PROTOCOL, JointTrainConfig
from gppo_world.joint_training import GROUPS, run_group
from gppo_world.m10_environment import M10Config
from tools.run_joint_three_communication_formal_cpu import (
    CONDITIONS, GROUP_ORDER, SEEDS, SPLITS, build_tapes, interleave,
    make_initialization, micro_check,
)


DEFAULT_RUN_NAME = "joint-four-module-three-communication-mixed-formal-gpu-20260916-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


class Heartbeat:
    def __init__(self, path: Path, root: Path) -> None:
        self.path = path
        self.root = root
        self.started = time.time()
        self.current: dict[str, object] = {"stage": "initializing"}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="matrix-heartbeat", daemon=True)

    def set(self, **fields: object) -> None:
        self.current = {**self.current, **fields, "updated_at": time.time()}
        write_json(self.path, {"status": "alive", "elapsed_seconds": time.time() - self.started, **self.current})

    def _loop(self) -> None:
        while not self.stop.wait(20.0):
            self.set()

    def start(self) -> None:
        self.set(stage="started")
        self.thread.start()

    def close(self, status: str, **fields: object) -> None:
        self.stop.set()
        self.set(stage="finished", status=status, **fields)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    args = parser.parse_args()
    faulthandler.enable()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    root = ROOT / "runs" / args.run_name
    if root.exists():
        raise SystemExit(f"refusing to overwrite existing corrected matrix root: {root}")
    root.mkdir(parents=True)
    env_config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    heartbeat = Heartbeat(root / "supervisor-heartbeat.json", root)
    heartbeat.start()
    started = time.perf_counter()
    try:
        heartbeat.set(stage="freeze_tapes")
        tapes = build_tapes(root)
        init = {str(seed): make_initialization(root, seed, env_config) for seed in SEEDS}
        write_json(root / "protocol-and-budget.json", {
            "protocol": JOINT_PROTOCOL,
            "matrix_schema": "12 runs; each run interleaves I/W1/W2 by episode",
            "conditions": CONDITIONS,
            "seeds": SEEDS,
            "groups": {key: asdict(GROUPS[key]) for key in GROUP_ORDER},
            "training_schedule": "fixed episode-level round-robin I,W1,W2; no condition-specific directory runs",
            "env_config": asdict(env_config),
            "budget_per_run": {"environment_steps": 16384, "policy_optimizer_steps": 128,
                               "world_optimizer_steps": {"A": 0, "B": 0, "C": 1024, "D": 1024},
                               "wall_seconds": 2400},
            "total_budget": {"environment_steps": 196608, "policy_optimizer_steps": 1536,
                             "world_optimizer_steps": 6144, "training_wall_seconds": 28800},
            "final_checkpoint_policy": "fixed final 16384-step transaction; no validation selection",
            "source_entry_sha256": sha256(Path(__file__)),
            "initializations": init,
            "device": args.device,
            "threads": {"intra_op": 4, "inter_op": 1},
            "status": "pre_registered_before_training",
        })
        heartbeat.set(stage="communication_micro_check")
        check = micro_check(root, tapes, env_config)
        write_json(root / "run-status.json", {"status": "micro_check_completed", "micro_check": check, "training_started": False})
        train_scenarios = interleave([tuple(tapes["train"][condition]) for condition in CONDITIONS])
        summaries: dict[str, object] = {}
        for seed in SEEDS:
            initialization = torch.load(root / "initialization" / f"seed-{seed}.pt", map_location="cpu", weights_only=False)
            for group in GROUP_ORDER:
                run_key = f"{seed}/{group}"
                out = root / "training" / f"seed-{seed}" / group
                run_id = f"formal-mixed-{group}-seed-{seed}-v1"
                write_json(out / "resolved-config.json", {
                    "run_id": run_id, "group": group, "seed": seed,
                    "protocol": JOINT_PROTOCOL, "environment": asdict(env_config),
                    "training": asdict(JointTrainConfig(seed=seed, rollout_steps=128)),
                    "initialization_sha256": init[str(seed)]["sha256"],
                    "training_tape_sha256": sha256(root / "formal-tapes.json"),
                    "condition_schedule": ["I", "W1", "W2"],
                    "condition_schedule_type": "episode_round_robin",
                    "scenario_count": len(train_scenarios),
                    "scenario_ids": [item.tape_id for item in train_scenarios],
                    "final_test_frozen_before_read": True,
                })
                heartbeat.set(stage="training", seed=seed, group=group, run_key=run_key,
                              completed_runs=len(summaries))
                max_world = 1024 if group in ("C", "D") else 0
                initial_policy = initialization["policy_state_dict"] if group in ("B", "C", "D") else None
                initial_world = initialization["world_state_dict"] if group in ("C", "D") else None
                summary = run_group(
                    group=group, run_id=run_id, output_dir=out, device=args.device,
                    env_config=env_config, train_config=JointTrainConfig(seed=seed, rollout_steps=128),
                    max_steps=16384, max_policy_updates=128, max_world_updates=max_world,
                    wall_seconds=2400.0, enforce_smoke_limits=False, verify_recovery_update=False,
                    scenarios=train_scenarios, initial_policy_state_dict=initial_policy,
                    initial_world_state_dict=initial_world,
                    initial_action_rng_state=initialization["action_rng_state"] if group in ("B", "C", "D") else None,
                )
                summaries[run_key] = {
                    "status": summary.get("status"), "environment_steps": summary.get("environment_steps"),
                    "policy_optimizer_steps": summary.get("policy_optimizer_steps"),
                    "world_optimizer_steps": summary.get("world_optimizer_steps"),
                    "stop_reason": summary.get("stop_reason"), "wall_seconds": summary.get("wall_seconds"),
                    "run_dir": str(out.relative_to(root)),
                }
                write_json(out / "formal-group-summary.json", summaries[run_key])
                write_json(root / "training-progress.json", {
                    "status": "running", "completed_runs": summaries,
                    "next": "fixed seed/group order", "elapsed_seconds": time.perf_counter() - started,
                })
        write_json(root / "training-summary.json", {
            "status": "completed", "runs": summaries,
            "actual_total_wall_seconds": time.perf_counter() - started,
        })
        heartbeat.close("completed", completed_runs=len(summaries))
        print(json.dumps({"run_root": str(root), "status": "training_completed", "runs": len(summaries)}, indent=2))
        return 0
    except BaseException as exc:
        heartbeat.close("failed", failure_type=type(exc).__name__, failure=str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
