"""Evaluate immutable gradient-isolation checkpoints at registered budget nodes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from gppo_world.joint_gppo import JointTrainConfig
from gppo_world.joint_training import _atomic_json
from gppo_world.m10_environment import M10Config, scenario_from_dict
from tools.evaluate_joint_four_group_pilot import _episode, _load_policy, _summarize, _transaction


RUN_ROOT = ROOT / "runs" / "joint-gradient-isolation-cpu-20260916-v1"
PILOT_ROOT = RUN_ROOT / "pilot"
GROUPS = ("B0", "CS", "CI", "DI")
PREFERENCES = ((0.2, 0.8), (0.5, 0.5), (0.8, 0.2))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    tape_payload = json.loads((ROOT / "runs" / "joint-four-group-single-seed-pilot-cpu-20260916-v1" / "frozen-tapes.json").read_text(encoding="utf-8"))
    validation = [scenario_from_dict(item) for item in tape_payload["validation"]]
    config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    train_config = JointTrainConfig(seed=1101, rollout_steps=128)
    output = {"status": "completed", "groups": {}, "scope": "immutable checkpoint replay; no parameter updates"}
    started = time.perf_counter()
    for group in GROUPS:
        output["groups"][group] = {}
        for step in (1024, 2048, 3072, 4096):
            checkpoint = _transaction(PILOT_ROOT / group, step)
            output["groups"][group][str(step)] = {
                "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "preferences": {},
            }
            for preference in PREFERENCES:
                key = "p" + "_".join(str(value).replace(".", "", 1) for value in preference)
                policy, world = _load_policy(group, checkpoint, config, torch.device("cpu"))
                rows = [_episode(group, policy, world, scenario, config, train_config, torch.device("cpu"), preference)
                        for scenario in validation]
                output["groups"][group][str(step)]["preferences"][key] = _summarize(rows)
    output["elapsed_seconds"] = time.perf_counter() - started
    _atomic_json(RUN_ROOT / "validation-curves.json", output)
    print(json.dumps({"path": str(RUN_ROOT / "validation-curves.json"), "elapsed_seconds": output["elapsed_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
