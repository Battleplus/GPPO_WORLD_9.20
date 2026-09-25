"""Evaluate frozen P/W checkpoints on the original validation tapes only.

This is a read-only development evaluation.  It never loads final-test data and
does not update model parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.joint_gppo import JointTrainConfig
from gppo_world.joint_training import GROUPS
from gppo_world.m10_environment import M10Config, M10Environment, scenario_from_dict
from tools import evaluate_joint_four_group_pilot as evaluator

PREFERENCES = ((0.2, 0.8), (0.5, 0.5), (0.8, 0.2))
SEEDS = (1101, 2203, 3307)
GROUP_NAMES = ("P", "W")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint(pair_root: Path, seed: int, group: str) -> Path:
    run_dir = pair_root / "source" / "runs" / "preference-update-pair-cpu-20260916-v1" / "training" / f"seed-{seed}" / group
    matches = sorted((run_dir / "transactions").glob("txn-step-00016384-policy-000128-world-000000.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one final checkpoint for {seed}/{group}, found {len(matches)}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-root", required=True, type=Path)
    parser.add_argument("--formal-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    tapes_path = args.formal_root / "formal-tapes.json"
    tapes = json.loads(tapes_path.read_text(encoding="utf-8"))
    print("validation_types", {condition: (type(tapes["validation"][condition]).__name__, type(tapes["validation"][condition][0]).__name__) for condition in ("I", "W1", "W2")}, flush=True)
    validation = [
        scenario_from_dict(item)
        for condition in ("I", "W1", "W2")
        for item in tapes["validation"][condition]
    ]
    env_config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    train_config = JointTrainConfig(seed=1101, rollout_steps=128)
    device = torch.device("cpu")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    results = {
        "schema": "joint-preference-update-pair-validation/1.0.0",
        "scope": "fixed original validation tapes; development evidence only; no final-test",
        "pair_root": str(args.pair_root),
        "formal_root": str(args.formal_root),
        "validation_tape_sha256": sha256(tapes_path),
        "groups": {},
        "evaluation_wall_seconds": None,
    }
    started = time.perf_counter()
    for seed in SEEDS:
        seed_key = str(seed)
        results["groups"][seed_key] = {}
        for group in GROUP_NAMES:
            checkpoint = _checkpoint(args.pair_root, seed, group)
            policy, world = evaluator._load_policy(group, checkpoint, env_config, device)
            preference_rows = {}
            for pref in PREFERENCES:
                key = "(" + ",".join(f"{x:.1f}" for x in pref) + ")"
                episode_rows = [
                    evaluator._episode(group, policy, world, scenario, env_config, train_config, device, pref)
                    for scenario in validation
                ]
                preference_rows[key] = {
                    "preference": pref,
                    "episodes": episode_rows,
                    "summary": evaluator._summarize(episode_rows),
                }
            results["groups"][seed_key][group] = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "preferences": preference_rows,
            }
    results["evaluation_wall_seconds"] = time.perf_counter() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.out),
        "evaluation_wall_seconds": results["evaluation_wall_seconds"],
        "validation_tape_sha256": results["validation_tape_sha256"],
        "groups": {
            seed: {group: {pref: block["summary"].get("physical_on_time")
                           for pref, block in payload[group]["preferences"].items()}
                   for group in GROUP_NAMES}
            for seed, payload in results["groups"].items()
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
