"""Replay CS-R/DS immutable checkpoints at each registered validation node."""

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

RUN_ROOT = ROOT / "runs" / "joint-shared-gradient-paired-cpu-20260916-v2"
TAPE_ROOT = ROOT / "runs" / "joint-four-group-single-seed-pilot-cpu-20260916-v1"
GROUPS = ("CS", "DS")
PREFERENCES = ((0.2, 0.8), (0.5, 0.5), (0.8, 0.2))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    tapes = json.loads((TAPE_ROOT / "frozen-tapes.json").read_text(encoding="utf-8"))
    validation = [scenario_from_dict(item) for item in tapes["validation"]]
    env_config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    train_config = JointTrainConfig(seed=1101, rollout_steps=128)
    output = {
        "schema": "joint-shared-gradient-validation-curve/1.0.0",
        "scope": "16 frozen validation parents; ideal communication; immutable replay; no updates",
        "groups": {},
    }
    started = time.perf_counter()
    for group in GROUPS:
        output["groups"][group] = {}
        for step in (1024, 2048, 3072, 4096):
            checkpoint = _transaction(RUN_ROOT / "pair-training" / group, step)
            policy, world = _load_policy(group, checkpoint, env_config, torch.device("cpu"))
            node = {
                "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "preferences": {},
            }
            for preference in PREFERENCES:
                key = "p" + "_".join(str(value).replace(".", "", 1) for value in preference)
                rows = [_episode(group, policy, world, scenario, env_config, train_config, torch.device("cpu"), preference)
                        for scenario in validation]
                node["preferences"][key] = {"preference": preference, "summary": _summarize(rows), "episodes": rows}
            output["groups"][group][str(step)] = node
    output["elapsed_seconds"] = time.perf_counter() - started
    _atomic_json(RUN_ROOT / "validation-curves.json", output)
    print(json.dumps({"path": str(RUN_ROOT / "validation-curves.json"), "elapsed_seconds": output["elapsed_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
