"""Paired CPU comparison: shared-gradient CS-R versus shared-gradient DS."""

from __future__ import annotations

from dataclasses import asdict
import copy
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

from gppo_world.joint_gppo import JOINT_PROTOCOL, JointTrainConfig, masked_normalized_preference
from gppo_world.joint_training import GROUPS, _atomic_json, run_group, verify_commit_pointer
from gppo_world.m10_environment import M10Config, M10Environment, scenario_from_dict
from tools.evaluate_joint_four_group_pilot import _decision, _episode, _load_policy, _summarize


RUN_ROOT = ROOT / "runs" / "joint-shared-gradient-paired-cpu-20260916-v2"
SOURCE_GRADIENT_RUN = ROOT / "runs" / "joint-gradient-isolation-cpu-20260916-v1"
SOURCE_TAPE_RUN = ROOT / "runs" / "joint-four-group-single-seed-pilot-cpu-20260916-v1"
GROUP_ORDER = ("CS", "DS")
PREFERENCES = ((0.2, 0.8), (0.5, 0.5), (0.8, 0.2))
SEED = 1101


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def frozen_inputs():
    tape_path = SOURCE_TAPE_RUN / "frozen-tapes.json"
    init_path = SOURCE_GRADIENT_RUN / "initialization.pt"
    if not tape_path.is_file() or not init_path.is_file():
        raise FileNotFoundError("registered frozen tapes or exact common initialization is missing")
    tapes = json.loads(tape_path.read_text(encoding="utf-8"))
    train = [scenario_from_dict(item) for item in tapes["train"]]
    validation = [scenario_from_dict(item) for item in tapes["validation"]]
    if len(train) != 32 or len(validation) != 16:
        raise ValueError("expected the registered 32/16 frozen tape sets")
    init = torch.load(init_path, map_location="cpu", weights_only=False)
    if set(init) != {"policy", "world", "action_rng"}:
        raise ValueError("common initialization payload identity mismatch")
    return train, validation, tape_path, init_path, init


def preference_key(value):
    return "p" + "_".join(str(x).replace(".", "", 1) for x in value)


def generate_b0_reference(validation, env_config, train_config):
    checkpoint = SOURCE_GRADIENT_RUN / "pilot" / "B0" / "last-recovery.pt"
    policy, _ = _load_policy("B0", checkpoint, env_config, torch.device("cpu"))
    records = []
    for scenario in validation:
        env = M10Environment(env_config, scenario)
        obs = env.reset()
        hidden = None
        steps = 0
        done = False
        while not done and steps < int(env_config.horizon / env_config.decision_interval) + 2:
            action, log_prob, value, next_hidden, _, _, _, _ = _decision(
                "B0", policy, None, obs, hidden, None, torch.tensor([0.5, 0.5]), torch.device("cpu"),
            )
            records.append({
                "tape_id": scenario.tape_id, "step": steps, "time": float(obs["time"]),
                "public_observation": {
                    "flat": np.asarray(obs["flat"]).tolist(), "uavs": np.asarray(obs["uavs"]).tolist(),
                    "tasks": np.asarray(obs["tasks"]).tolist(), "entity_ids": obs.get("public_entity_ids", {}),
                    "version": obs.get("version"), "mask": np.asarray(obs["mask"], dtype=int).tolist(),
                },
                "action": action, "log_prob": log_prob, "value": value,
            })
            obs, _, done, _ = env.step(action)
            hidden = next_hidden
            steps += 1
    return {
        "schema": "b0-public-reference-trajectory/1.0.0", "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint), "tape_ids": [x.tape_id for x in validation],
        "records": records, "purpose": "fixed public-observation reference only; never used for updates",
    }


def run_correctness(train, env_config, train_config, init_policy, init_world, action_rng):
    root = RUN_ROOT / "correctness-stage"
    root.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for group in GROUP_ORDER:
        summaries[group] = run_group(
            group=group, run_id=f"joint-shared-gradient-correctness-{group}-seed1101-v1",
            output_dir=root / group, device="cpu", env_config=env_config,
            train_config=train_config, max_steps=128, max_policy_updates=1,
            max_world_updates=4, wall_seconds=420, enforce_smoke_limits=True,
            verify_recovery_update=True, scenarios=train,
            initial_policy_state_dict=init_policy, initial_world_state_dict=init_world,
            initial_action_rng_state=action_rng,
        )
        verify_commit_pointer(root / group)
        if summaries[group].get("recovery_update_verification", {}).get("status") != "passed":
            raise RuntimeError(f"same-CPU recovery failed for {group}")
    _atomic_json(root / "summary.json", {"status": "passed", "groups": summaries})
    return summaries


def run_pair(train, env_config, train_config, init_policy, init_world, action_rng):
    root = RUN_ROOT / "pair-training"
    root.mkdir(parents=True, exist_ok=False)
    summaries = {}
    start = time.perf_counter()
    for group in GROUP_ORDER:
        summary = run_group(
            group=group, run_id=f"joint-shared-gradient-pair-{group}-seed1101-v1",
            output_dir=root / group, device="cpu", env_config=env_config,
            train_config=train_config, max_steps=4096, max_policy_updates=32,
            max_world_updates=256, wall_seconds=3600, enforce_smoke_limits=False,
            verify_recovery_update=False, scenarios=train,
            initial_policy_state_dict=init_policy, initial_world_state_dict=init_world,
            initial_action_rng_state=action_rng,
        )
        summary["transaction_commit"] = verify_commit_pointer(root / group)
        summaries[group] = summary
        _atomic_json(root / group / "pair-group-summary.json", summary)
        if (summary.get("environment_steps") != 4096 or summary.get("policy_optimizer_steps") != 32
                or summary.get("world_optimizer_steps") != 256
                or summary.get("stop_reason") != "environment_step_budget"):
            raise RuntimeError(f"{group} did not complete the registered pair budget")
    _atomic_json(root / "summary.json", {
        "status": "completed", "groups": summaries,
        "actual_training_wall_seconds": time.perf_counter() - start,
        "budget": {"environment_steps": 8192, "policy_optimizer_steps": 64, "world_optimizer_steps": 512},
    })
    return summaries


def evaluate(validation, env_config, train_config):
    result = {"status": "completed", "scope": "16 frozen validation parents; ideal communication; no updates", "groups": {}}
    for group in GROUP_ORDER:
        checkpoint = RUN_ROOT / "pair-training" / group / "last-recovery.pt"
        policy, world = _load_policy(group, checkpoint, env_config, torch.device("cpu"))
        result["groups"][group] = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "preferences": {}}
        for preference in PREFERENCES:
            rows = [_episode(group, policy, world, scenario, env_config, train_config, torch.device("cpu"), preference)
                    for scenario in validation]
            result["groups"][group]["preferences"][preference_key(preference)] = {
                "preference": preference, "summary": _summarize(rows), "episodes": rows,
            }
    _atomic_json(RUN_ROOT / "validation-results.json", result)
    return result


def main() -> int:
    if RUN_ROOT.exists():
        raise SystemExit(f"refusing to overwrite {RUN_ROOT}")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    env_config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    train_config = JointTrainConfig(seed=SEED, rollout_steps=128)
    train, validation, tape_path, init_path, init = frozen_inputs()
    RUN_ROOT.mkdir(parents=True, exist_ok=False)
    init_policy = copy.deepcopy(init["policy"])
    init_world = copy.deepcopy(init["world"])
    action_rng = init["action_rng"].clone()
    _atomic_json(RUN_ROOT / "pre-registration.json", {
        "protocol": JOINT_PROTOCOL, "groups": {key: asdict(GROUPS[key]) for key in GROUP_ORDER},
        "seed": SEED, "device": "server_cpu", "threads": {"torch_intra": 4, "torch_inter": 1},
        "environment": asdict(env_config), "training": asdict(train_config),
        "budgets": {"correctness_env_steps": 256, "correctness_policy_calls": 8, "correctness_world_calls": 16,
                     "pair_env_steps": 8192, "pair_policy_updates": 64, "pair_world_updates": 512},
        "tape_source": str(tape_path), "tape_source_sha256": sha256(tape_path),
        "common_initialization_source": str(init_path), "common_initialization_sha256": sha256(init_path),
        "policy_initialization_sha256": state_hash(init_policy), "world_initialization_sha256": state_hash(init_world),
        "action_rng_sha256": hashlib.sha256(action_rng.numpy().tobytes()).hexdigest(),
        "fixed_preferences": PREFERENCES, "no_new_tapes": True,
    })
    torch.save({"policy": init_policy, "world": init_world, "action_rng": action_rng}, RUN_ROOT / "initialization-copy.pt")
    _atomic_json(RUN_ROOT / "b0-public-reference.json", generate_b0_reference(validation, env_config, train_config))
    correctness = run_correctness(train, env_config, train_config, init_policy, init_world, action_rng)
    pair = run_pair(train, env_config, train_config, init_policy, init_world, action_rng)
    evaluation = evaluate(validation, env_config, train_config)
    _atomic_json(RUN_ROOT / "experiment-summary.json", {
        "status": "completed", "correctness": correctness, "pair": pair,
        "evaluation": {"path": "validation-results.json", "groups": list(evaluation["groups"])},
        "platform": platform.platform(), "python": sys.version,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda, "numpy": np.__version__,
    })
    print(json.dumps({"run_root": str(RUN_ROOT), "correctness": {g: v.get("recovery_update_verification", {}).get("status") for g,v in correctness.items()}, "pair": {g: {"steps": v.get("environment_steps"), "policy": v.get("policy_optimizer_steps"), "world": v.get("world_optimizer_steps"), "stop": v.get("stop_reason")} for g,v in pair.items()}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
