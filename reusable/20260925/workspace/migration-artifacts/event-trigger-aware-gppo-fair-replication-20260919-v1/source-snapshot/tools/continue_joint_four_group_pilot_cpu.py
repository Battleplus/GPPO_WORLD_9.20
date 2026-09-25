"""Continue the bounded CPU A-D pilot after a completed group without rerunning it."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from gppo_world.joint_gppo import JOINT_PROTOCOL, JointTrainConfig
from gppo_world.joint_training import GROUPS, _atomic_json, run_group, verify_commit_pointer
from gppo_world.m10_environment import M10Config, scenario_from_dict


RUN_ROOT = ROOT / "runs" / "joint-four-group-single-seed-pilot-cpu-20260916-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if not RUN_ROOT.is_dir():
        raise SystemExit(f"pilot root is missing: {RUN_ROOT}")
    frozen = json.loads((RUN_ROOT / "frozen-tapes.json").read_text(encoding="utf-8"))
    train_tapes = tuple(scenario_from_dict(item) for item in frozen["train"])
    validation_tapes = tuple(scenario_from_dict(item) for item in frozen["validation"])
    if len(train_tapes) != 32 or len(validation_tapes) != 16:
        raise SystemExit("frozen pilot tape counts do not match the registered CPU pilot")
    a_summary_path = RUN_ROOT / "A" / "pilot-group-summary.json"
    if not a_summary_path.is_file():
        raise SystemExit("completed A summary is missing; refusing to infer or rerun A")
    a_summary = json.loads(a_summary_path.read_text(encoding="utf-8"))
    if not (a_summary.get("status") == "completed"
            and int(a_summary.get("environment_steps", -1)) == 4096
            and int(a_summary.get("policy_optimizer_steps", -1)) == 32
            and int(a_summary.get("world_optimizer_steps", -1)) == 0):
        raise SystemExit("A is not a complete registered pilot group")
    a_commit = verify_commit_pointer(RUN_ROOT / "A")
    if a_commit["last_committed_step"] != 4095 or a_commit["last_committed_policy_update"] != 32:
        raise SystemExit("A commit pointer is not at the registered pilot boundary")

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    env_config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    train_config = JointTrainConfig(seed=1101, rollout_steps=128)
    started = time.perf_counter()
    summaries = {"A": a_summary}
    for group in ("B", "C", "D"):
        out = RUN_ROOT / group
        if out.exists():
            raise SystemExit(f"refusing to overwrite existing group directory: {out}")
        elapsed = float(a_summary.get("pilot_wall_seconds_including_group_restore", 0.0)) + (time.perf_counter() - started)
        remaining = min(3600.0, 14400.0 - elapsed)
        if remaining <= 0:
            raise SystemExit(f"shared pilot wall-clock budget exhausted before group {group}")
        out.mkdir(parents=True, exist_ok=False)
        run_id = f"joint-four-group-pilot-cpu-{group}-seed1101-v1"
        _atomic_json(out / "resolved-config.json", {
            "protocol": JOINT_PROTOCOL, "group": group,
            "group_config": asdict(GROUPS[group]),
            "environment": asdict(env_config), "training": asdict(train_config),
            "run_id": run_id, "train_tape_sha256": sha256(RUN_ROOT / "frozen-tapes.json"),
            "train_tape_ids": [item.tape_id for item in train_tapes],
            "new_initialization": True, "pilot_checkpoint_used_as_start": False,
            "continuation_of_completed_A": True,
        })
        wm_cap = 256 if group in ("C", "D") else 0
        summary = run_group(
            group=group, run_id=run_id, output_dir=out, device="cpu",
            env_config=env_config, train_config=train_config,
            max_steps=4096, max_policy_updates=32, max_world_updates=wm_cap,
            wall_seconds=remaining, enforce_smoke_limits=False,
            verify_recovery_update=False, scenarios=train_tapes,
        )
        commit = verify_commit_pointer(out)
        exact = (summary.get("status") == "completed"
                 and int(summary.get("environment_steps", -1)) == 4096
                 and int(summary.get("policy_optimizer_steps", -1)) == 32
                 and int(summary.get("world_optimizer_steps", -1)) == wm_cap
                 and commit["last_committed_step"] == 4095
                 and commit["last_committed_policy_update"] == 32)
        if not exact:
            summary["status"] = "incomplete"
            summary["stop_reason"] = "registered_budget_not_reached"
            summary["transaction_commit"] = commit
            _atomic_json(out / "pilot-group-summary.json", summary)
            raise RuntimeError(f"group {group} did not complete its exact registered pilot budget")
        summary["stop_reason"] = "environment_step_budget"
        summary["transaction_commit"] = commit
        summary["pilot_wall_seconds_including_group_restore"] = float(a_summary.get("pilot_wall_seconds_including_group_restore", 0.0)) + (time.perf_counter() - started)
        summary["actual_policy_optimizer_calls"] = int(summary.get("policy_optimizer_steps", 0))
        summary["actual_world_optimizer_calls"] = int(summary.get("world_optimizer_steps", 0))
        summaries[group] = summary
        _atomic_json(out / "pilot-group-summary.json", summary)

    total = {
        "status": "completed", "continuation": True, "groups": summaries,
        "actual_training_wall_seconds": float(a_summary.get("pilot_wall_seconds_including_group_restore", 0.0)) + (time.perf_counter() - started),
        "registered_training_wall_cap_seconds": 14400,
        "environment_steps_total": sum(int(x.get("environment_steps", 0)) for x in summaries.values()),
        "policy_optimizer_calls_total": sum(int(x.get("actual_policy_optimizer_calls", x.get("policy_optimizer_steps", 0))) for x in summaries.values()),
        "world_optimizer_calls_total": sum(int(x.get("actual_world_optimizer_calls", x.get("world_optimizer_steps", 0))) for x in summaries.values()),
        "device": "CPU", "threads": 4, "validation_tape_count": len(validation_tapes),
    }
    _atomic_json(RUN_ROOT / "pilot-training-summary.json", total)
    print(json.dumps({"run_root": str(RUN_ROOT), "groups": {
        key: {"status": value.get("status"), "steps": value.get("environment_steps"),
              "policy_updates": value.get("policy_optimizer_steps"),
              "world_updates": value.get("world_optimizer_steps"),
              "stop_reason": value.get("stop_reason")}
        for key, value in summaries.items()
    }}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
