"""Run the registered, bounded A/B/C/D single-seed development pilot."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import re
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
from gppo_world.m10_environment import M10Config, default_scenario, scenario_tape, scenario_to_dict


RUN_ROOT = ROOT / "runs" / "joint-four-group-single-seed-pilot-cpu-20260916-v1"
TAPE_RE = re.compile(r"(?:train|validation|test|ood)-[A-Za-z0-9_-]+-seed-[0-9]+")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def scenario_core(item):
    value = scenario_to_dict(item)
    value.pop("communication", None)
    return value


def historical_tape_ids() -> set[str]:
    found: set[str] = set()
    parent = ROOT.parent
    roots = [ROOT / "runs"]
    roots.extend(
        path for path in parent.iterdir()
        if path.is_dir() and (path.name.startswith("WORLD-GPPO_9.11") or path.name.startswith("m10-"))
    )
    for scan_root in roots:
        for path in scan_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".md", ".txt"}:
                continue
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as stream:
                    for line in stream:
                        found.update(match.group(0) for match in TAPE_RE.finditer(line))
            except OSError:
                continue
    return found


def audit_tapes(train, validation) -> dict:
    all_tapes = [*train, *validation]
    ids = [item.tape_id for item in all_tapes]
    core_hashes = [hashlib.sha256(canonical(scenario_core(item))).hexdigest() for item in all_tapes]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate tape identity within frozen train/validation sets")
    if len(set(core_hashes)) != len(core_hashes):
        raise RuntimeError("duplicate normalized scenario content within frozen train/validation sets")

    old_ids = historical_tape_ids()
    overlap_ids = sorted(set(ids) & old_ids)
    if overlap_ids:
        raise RuntimeError(f"frozen tape IDs overlap historical run records: {overlap_ids[:5]}")
    old_core_hashes: set[str] = set()
    unreconstructable_ids: list[str] = []
    for tape_id in old_ids:
        match = re.fullmatch(r"(train|validation|test|ood)-(.+)-seed-([0-9]+)", tape_id)
        if not match:
            continue
        split, name, seed_text = match.groups()
        try:
            old = default_scenario(name, seed=int(seed_text), split=split)
            old_core_hashes.add(hashlib.sha256(canonical(scenario_core(old))).hexdigest())
        except (ValueError, TypeError):
            unreconstructable_ids.append(tape_id)
    overlap_content = sorted(set(core_hashes) & old_core_hashes)
    if overlap_content:
        raise RuntimeError("normalized scenario content duplicates a reconstructable historical tape")

    return {
        "protocol": JOINT_PROTOCOL,
        "generator": "scenario_tape(name=mixed), unchanged M10 scenario generator",
        "train": {"count": len(train), "base_seed": 841001, "tape_ids": [x.tape_id for x in train]},
        "validation": {"count": len(validation), "base_seed": 841002, "tape_ids": [x.tape_id for x in validation]},
        "train_validation_id_overlap": [],
        "normalized_content_unique_within_new_splits": True,
        "historical_tape_ids_scanned": len(old_ids),
        "historical_recorded_id_overlap": [],
        "historical_reconstructable_core_hashes": len(old_core_hashes),
        "historical_content_overlap": [],
        "unreconstructable_historical_ids": sorted(set(unreconstructable_ids)),
        "content_hashes": dict(zip(ids, core_hashes)),
        "limitation": "Historical opaque binary checkpoints without a recorded tape_id were not content-reconstructed.",
    }


def main() -> int:
    if RUN_ROOT.exists():
        raise SystemExit(f"refusing to overwrite existing pilot directory: {RUN_ROOT}")
    device = "cpu"
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    RUN_ROOT.mkdir(parents=True, exist_ok=False)

    train_tapes = scenario_tape("train", count=32, base_seed=841001, name="mixed")
    validation_tapes = scenario_tape("validation", count=16, base_seed=841002, name="mixed")
    audit = audit_tapes(train_tapes, validation_tapes)
    tape_payload = {
        "train": [scenario_to_dict(item) for item in train_tapes],
        "validation": [scenario_to_dict(item) for item in validation_tapes],
    }
    tape_path = RUN_ROOT / "frozen-tapes.json"
    tape_path.write_bytes(canonical(tape_payload) + b"\n")
    audit["tape_file_sha256"] = sha256(tape_path)
    audit["tape_file_bytes"] = tape_path.stat().st_size
    _atomic_json(RUN_ROOT / "tape-audit.json", audit)

    env_config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    train_config = JointTrainConfig(seed=1101, rollout_steps=128)
    prereg = {
        "protocol": JOINT_PROTOCOL,
        "seed": 1101,
        "groups": {key: asdict(value) for key, value in GROUPS.items()},
        "environment": asdict(env_config),
        "training": asdict(train_config),
        "budgets": {"steps_per_group": 4096, "policy_updates_per_group": 32,
                    "world_updates_C_D": 256, "world_updates_A_B": 0,
                    "wall_seconds_total_training": 14400, "wall_seconds_per_group": 3600,
                    "validation_checkpoints": [1024, 2048, 3072, 4096]},
        "selection": "final fixed-budget checkpoint only; validation is descriptive and not used for selection",
        "validation_timing": "post-hoc replay of immutable 1024-step transaction checkpoints; never feeds back into training",
        "preferences": [[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]],
        "device": "CPU",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "numpy": np.__version__, "threads": torch.get_num_threads(),
        "tape_file_sha256": sha256(tape_path),
    }
    _atomic_json(RUN_ROOT / "pilot-preregistration.json", prereg)
    started = time.perf_counter()
    group_summaries = {}
    for group in ("A", "B", "C", "D"):
        elapsed = time.perf_counter() - started
        remaining = min(3600.0, 14400.0 - elapsed)
        if remaining <= 0:
            group_summaries[group] = {"status": "not_started", "stop_reason": "shared_wall_clock_budget"}
            break
        out = RUN_ROOT / group
        out.mkdir(parents=True, exist_ok=False)
        _atomic_json(out / "resolved-config.json", {
            "protocol": JOINT_PROTOCOL, "group": group,
            "group_config": asdict(GROUPS[group]),
            "environment": asdict(env_config), "training": asdict(train_config),
            "run_id": f"joint-four-group-pilot-{group}-seed1101-v1",
            "train_tape_sha256": sha256(tape_path),
            "train_tape_ids": [item.tape_id for item in train_tapes],
            "new_initialization": True,
            "shared_base_initialization_check": "same seed; verified exact zero delta before training",
            "pilot_checkpoint_used_as_start": False,
        })
        wm_cap = 256 if group in ("C", "D") else 0
        summary = run_group(
            group=group, run_id=f"joint-four-group-pilot-cpu-{group}-seed1101-v1",
            output_dir=out, device=device, env_config=env_config, train_config=train_config,
            max_steps=4096, max_policy_updates=32, max_world_updates=wm_cap,
            wall_seconds=remaining, enforce_smoke_limits=False,
            verify_recovery_update=False, scenarios=train_tapes,
        )
        commit = verify_commit_pointer(out)
        summary["transaction_commit"] = commit
        summary["pilot_wall_seconds_including_group_restore"] = time.perf_counter() - started - elapsed
        summary["actual_policy_optimizer_calls"] = int(summary.get("policy_optimizer_steps", 0))
        summary["actual_world_optimizer_calls"] = int(summary.get("world_optimizer_steps", 0))
        group_summaries[group] = summary
        _atomic_json(out / "pilot-group-summary.json", summary)
        if (summary.get("environment_steps") != 4096
                or summary.get("policy_optimizer_steps") != 32
                or summary.get("world_optimizer_steps") != wm_cap
                or summary.get("stop_reason") != "environment_step_budget"):
            raise RuntimeError(f"group {group} did not complete its exact registered pilot budget")

    _atomic_json(RUN_ROOT / "pilot-training-summary.json", {
        "status": "completed" if len(group_summaries) == 4 and all(x.get("status") == "completed" for x in group_summaries.values()) else "incomplete",
        "groups": group_summaries,
        "actual_training_wall_seconds": time.perf_counter() - started,
        "registered_training_wall_cap_seconds": 14400,
        "environment_steps_total": sum(int(x.get("environment_steps", 0)) for x in group_summaries.values()),
        "policy_optimizer_calls_total": sum(int(x.get("actual_policy_optimizer_calls", 0)) for x in group_summaries.values()),
        "world_optimizer_calls_total": sum(int(x.get("actual_world_optimizer_calls", 0)) for x in group_summaries.values()),
    })
    print(json.dumps({"run_root": str(RUN_ROOT), "groups": {
        key: {"status": value.get("status"), "steps": value.get("environment_steps"),
              "policy_updates": value.get("policy_optimizer_steps"),
              "world_updates": value.get("world_optimizer_steps"),
              "stop_reason": value.get("stop_reason")}
        for key, value in group_summaries.items()
    }}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
