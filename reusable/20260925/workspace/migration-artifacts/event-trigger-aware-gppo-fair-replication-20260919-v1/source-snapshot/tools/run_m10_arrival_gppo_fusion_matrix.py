"""Run the frozen arrival-protocol GPPO fusion multi-seed validation matrix.

This runner starts policy training from scratch for the three learning groups,
keeps the arrival consequence model frozen, and evaluates all groups on one
new, pre-registered 64-tape test set.  It deliberately does not train a world
model, tune on test outcomes, or run the previously rejected variants.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.arrival_gppo_fusion import (  # noqa: E402
    CandidateAwarePolicy,
    FrozenArrivalConsequenceScorer,
)
from gppo_world.m10_environment import (  # noqa: E402
    M10Config,
    M10Scenario,
    scenario_to_dict,
    weak_communication_tape,
)
from gppo_world.m10_training import (  # noqa: E402
    M10ActorCritic,
    PPOConfig,
    train_policy,
)
from tools.run_m10_arrival_gppo_fusion_pilot import (  # noqa: E402
    evaluate_group,
    train_candidate_policy,
)


RUN_ID = "m10-arrival-gppo-fusion-matrix-20260914-v2"
SEEDS = (1101, 2203, 3307)
LEARNING_GROUPS = ("GPPO", "GPPO-History", "GPPO-History-CandidateArrival")
ENV_STEPS = 8192
ROLLOUT_STEPS = 256
UPDATE_EPOCHS = 4
EXPECTED_UPDATES = (ENV_STEPS // ROLLOUT_STEPS) * UPDATE_EPOCHS
MAX_OPTIMIZER_UPDATES = EXPECTED_UPDATES
TOTAL_WALL_SECONDS = 8 * 60 * 60
TRAIN_BASE_SEED = 93001
TEST_BASE_SEED = 94001
TRAIN_TAPE_COUNT = 32
VALIDATION_TAPE_COUNT = 16
TEST_TAPE_COUNT = 64
TEST_TASKS_PER_TAPE = 6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def arrival_config() -> M10Config:
    return M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")


def git_value(*args: str) -> str:
    return subprocess.run(("git", *args), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def runtime_snapshot(device: str, threads: int) -> dict[str, Any]:
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "threads": threads,
        "device": device,
        "cuda": torch.version.cuda if device == "cuda" else None,
        "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "cuda_device_count": torch.cuda.device_count() if device == "cuda" else 0,
    }


def collect_model_data_ids(asset_root: Path) -> dict[str, set[str]]:
    collected: dict[str, set[str]] = {"parent_episode_id": set(), "prefix_id": set(), "tape_id": set()}
    data_root = asset_root / "data"
    for path in sorted(data_root.glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            for key in collected:
                value = row.get(key)
                if isinstance(value, str):
                    collected[key].add(value)
    return collected


def save_recovery_policy(path: Path, policy: torch.nn.Module, metadata: dict[str, Any], *, seed: int, tape_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    optimizer = getattr(policy, "optimizer", None)
    payload = {
        "state_dict": policy.state_dict(),
        "metadata": metadata,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "recovery_state": {
            "environment_steps": int(metadata.get("steps", metadata.get("environment_steps", 0))),
            "optimizer_updates": int(metadata.get("optimizer_updates", 0)),
            "seed": seed,
            "variant": metadata.get("variant"),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "data_order": {"train_tape_ids": tape_ids, "ordering": "frozen_input_order"},
            "early_stopping": {"enabled": False, "reason": "policy pilot uses fixed budget endpoint"},
        },
    }
    torch.save(payload, path)


def build_policy(config: M10Config, *, history: bool, fusion: bool, device: str) -> torch.nn.Module:
    base = M10ActorCritic(
        uav_count=config.uav_count,
        task_capacity=config.task_capacity,
        action_count=config.action_count,
        encoder="graph",
        type_count=5,
        history=history,
        context_dim=0,
        region_count=config.region_count,
        target_count=config.target_count,
        event_capacity=config.event_capacity,
        relation_width=config.relation_width,
    )
    if fusion:
        policy = CandidateAwarePolicy(base)
    else:
        policy = base
    return policy.to(torch.device(device))


def verify_saved_checkpoint(path: Path, config: M10Config, *, group: str, device: str) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("optimizer_state_dict") is None or payload.get("recovery_state") is None:
        raise RuntimeError(f"checkpoint missing optimizer/recovery state: {path}")
    policy = build_policy(config, history=group != "GPPO", fusion=group == "GPPO-History-CandidateArrival", device=device)
    policy.load_state_dict(payload["state_dict"], strict=True)
    policy.optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)  # type: ignore[attr-defined]
    policy.optimizer.load_state_dict(payload["optimizer_state_dict"])  # type: ignore[attr-defined]
    return {
        "status": "passed",
        "path": str(path),
        "sha256": sha256_file(path),
        "strict_state_dict": True,
        "optimizer_state_present": True,
        "recovery_state_present": True,
        "optimizer_updates": payload["recovery_state"].get("optimizer_updates"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arrival-model", type=Path, required=True)
    parser.add_argument("--arrival-model-sha256", required=True)
    parser.add_argument("--wm-asset-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; no silent fallback")
    if not 1 <= args.threads <= 4:
        raise SystemExit("threads must be 1..4")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing non-empty output: {args.out}")
    if not args.arrival_model.is_file() or not args.wm_asset_root.is_dir():
        raise SystemExit("arrival model or world-model asset root does not exist")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    config = arrival_config()
    model_sha = sha256_file(args.arrival_model)
    if model_sha != args.arrival_model_sha256.lower():
        raise SystemExit(f"arrival model SHA-256 mismatch: expected {args.arrival_model_sha256}, got {model_sha}")
    scorer = FrozenArrivalConsequenceScorer.from_checkpoint(args.arrival_model, device=args.device, expected_sha256=model_sha)
    tapes = {
        "train": list(weak_communication_tape("train", count=TRAIN_TAPE_COUNT, base_seed=TRAIN_BASE_SEED, level="composite")),
        "validation": list(weak_communication_tape("validation", count=VALIDATION_TAPE_COUNT, base_seed=TRAIN_BASE_SEED, level="composite")),
        "test": list(weak_communication_tape("test", count=TEST_TAPE_COUNT, base_seed=TEST_BASE_SEED, level="composite")),
    }
    model_ids = collect_model_data_ids(args.wm_asset_root)
    test_tape_ids = {item.tape_id for item in tapes["test"]}
    test_ids = {
        "tape_id": test_tape_ids,
        "parent_episode_id": {f"test:{value}" for value in test_tape_ids},
        "prefix_id": {f"test:{value}:prefix-{index}-public-hash-legal" for value in test_tape_ids for index in range(1, 10)},
    }
    overlap = {key: sorted(test_ids[key] & model_ids.get(key, set())) for key in test_ids}
    if any(overlap.values()):
        raise SystemExit(f"test tape overlaps frozen world-model data: {overlap}")
    dump(args.out / "tapes.json", {key: [scenario_to_dict(item) for item in values] for key, values in tapes.items()})
    identity = {
        "run_id": RUN_ID,
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status_at_start": git_value("status", "--short"),
        "protocol": "world-gppo-9.11-arrival/0.1.0",
        "arrival_config": asdict(config),
        "device": args.device,
        "runtime": runtime_snapshot(args.device, args.threads),
        "training": {
            "from_scratch": True,
            "seeds": list(SEEDS),
            "learning_groups": list(LEARNING_GROUPS),
            "environment_steps_per_group_seed": ENV_STEPS,
            "ppo": asdict(PPOConfig(rollout_steps=ROLLOUT_STEPS, update_epochs=UPDATE_EPOCHS)),
            "expected_rollout_updates": ENV_STEPS // ROLLOUT_STEPS,
            "expected_optimizer_steps": EXPECTED_UPDATES,
            "max_optimizer_updates": MAX_OPTIMIZER_UPDATES,
            "total_wall_seconds": TOTAL_WALL_SECONDS,
            "rotation": "group-seed jobs are executed in deterministic group order; no mid-run retuning",
        },
        "evaluation": {"test_parent_tapes": TEST_TAPE_COUNT, "tasks_per_tape": TEST_TASKS_PER_TAPE, "test_task_denominator": TEST_TAPE_COUNT * TEST_TASKS_PER_TAPE, "test_results_read_after_training": True},
        "tape_generation": {"train": {"count": TRAIN_TAPE_COUNT, "base_seed": TRAIN_BASE_SEED}, "validation": {"count": VALIDATION_TAPE_COUNT, "base_seed": TRAIN_BASE_SEED}, "test": {"count": TEST_TAPE_COUNT, "base_seed": TEST_BASE_SEED}, "communication": "composite weak communication", "test_tape_ids": [item.tape_id for item in tapes["test"]]},
        "model": {"path": str(args.arrival_model.resolve()), "sha256": model_sha, "format": scorer.checkpoint_format, "feature_contract": asdict(scorer.contract), "frozen": True, "requires_grad": False},
        "world_model_training_data": {"asset_root": str(args.wm_asset_root.resolve()), "data_file_sha256": {p.name: sha256_file(p) for p in sorted((args.wm_asset_root / "data").glob("*.jsonl"))}, "test_overlap": overlap},
        "source_sha256": {"matrix_runner": sha256_file(Path(__file__)), "pilot_runner": sha256_file(ROOT / "tools" / "run_m10_arrival_gppo_fusion_pilot.py"), "fusion": sha256_file(ROOT / "gppo_world" / "arrival_gppo_fusion.py"), "training": sha256_file(ROOT / "gppo_world" / "m10_training.py"), "protocol": sha256_file(ROOT / "configs" / "world-gppo-9.11-arrival-v0.1.0.json")},
    }
    dump(args.out / "run-identity.json", identity)
    dump(args.out / "run-status.json", {"status": "running", "run_id": RUN_ID, "started_at": time.time(), "groups_completed": []})
    results: dict[str, Any] = {"status": "running", "run_id": RUN_ID, "identity_sha256": sha256_file(args.out / "run-identity.json"), "groups": {}, "traditional": None}
    started = time.perf_counter()
    results["traditional"] = evaluate_group(None, tapes["test"], config, device=args.device, scorer=None, variant="legal-public-rule")  # type: ignore[arg-type]
    dump(args.out / "traditional-evaluation.json", results["traditional"])
    for seed in SEEDS:
        for group in ("GPPO", "GPPO-History", "GPPO-History-CandidateArrival"):
            if time.perf_counter() - started >= TOTAL_WALL_SECONDS:
                results["status"] = "stopped_wall_clock"
                dump(args.out / "matrix-results.json", results)
                dump(args.out / "run-status.json", {"status": results["status"], "run_id": RUN_ID, "groups_completed": list(results["groups"])})
                return 2
            print(json.dumps({"event": "group_start", "seed": seed, "group": group}, sort_keys=True), flush=True)
            group_start = time.perf_counter()
            if group == "GPPO-History-CandidateArrival":
                policy, metadata = train_candidate_policy(
                    env_config=config, scenarios=tapes["train"], scorer=scorer, seed=seed, device=args.device,
                    steps=ENV_STEPS, max_updates=MAX_OPTIMIZER_UPDATES, max_wall_seconds=TOTAL_WALL_SECONDS,
                )
                variant_name = group
                policy_path = args.out / "checkpoints" / group / f"seed-{seed}" / "last-recovery.pt"
                save_recovery_policy(policy_path, policy, metadata, seed=seed, tape_ids=[item.tape_id for item in tapes["train"]])
                evaluation = evaluate_group(policy, tapes["test"], config, device=args.device, scorer=scorer, variant=variant_name)
            else:
                history = group == "GPPO-History"
                policy, metadata = train_policy(
                    variant=group, encoder="graph", type_count=5, history=history, fusion="base", model=None,
                    seed=seed, steps=ENV_STEPS, env_config=config,
                    ppo_config=PPOConfig(rollout_steps=ROLLOUT_STEPS, update_epochs=UPDATE_EPOCHS),
                    device=args.device, scenarios=tapes["train"],
                )
                variant_name = group
                policy_path = args.out / "checkpoints" / group / f"seed-{seed}" / "last-recovery.pt"
                save_recovery_policy(policy_path, policy, metadata, seed=seed, tape_ids=[item.tape_id for item in tapes["train"]])
                evaluation = evaluate_group(policy, tapes["test"], config, device=args.device, scorer=None, variant=variant_name)
            metadata["wall_seconds_including_save_and_test_eval"] = time.perf_counter() - group_start
            metadata["expected_optimizer_steps"] = EXPECTED_UPDATES
            metadata["checkpoint_sha256"] = sha256_file(policy_path)
            metadata["test_evaluation_read_after_training"] = True
            verification = verify_saved_checkpoint(policy_path, config, group=group, device=args.device)
            entry = {"seed": seed, "group": group, "training": metadata, "evaluation": evaluation, "checkpoint": str(policy_path), "checkpoint_load_verification": verification}
            results["groups"][f"{group}__seed{seed}"] = entry
            dump(args.out / "matrix-results.json", results)
            dump(args.out / "run-status.json", {"status": "running", "run_id": RUN_ID, "groups_completed": list(results["groups"]), "elapsed_seconds": time.perf_counter() - started})
            print(json.dumps({"event": "group_complete", "seed": seed, "group": group, "steps": metadata.get("steps", metadata.get("environment_steps")), "optimizer_updates": metadata.get("optimizer_updates"), "elapsed_seconds": metadata.get("elapsed_seconds")}, sort_keys=True), flush=True)
    results["status"] = "complete"
    results["elapsed_seconds"] = time.perf_counter() - started
    dump(args.out / "matrix-results.json", results)
    dump(args.out / "run-status.json", {"status": "complete", "run_id": RUN_ID, "completed_at": time.time(), "groups_completed": list(results["groups"]), "elapsed_seconds": results["elapsed_seconds"]})
    print(json.dumps({"status": "complete", "output": str(args.out), "groups": len(results["groups"]), "elapsed_seconds": results["elapsed_seconds"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
