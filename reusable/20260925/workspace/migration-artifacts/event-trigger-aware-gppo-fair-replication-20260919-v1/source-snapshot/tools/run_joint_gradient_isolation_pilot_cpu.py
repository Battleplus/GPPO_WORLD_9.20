"""Bounded CPU comparison for shared versus isolated world-model gradients.

This is an intentionally separate experiment.  Historical A-D runs keep their
original semantics; the registered comparison groups are B0, CS, CI and DI.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from gppo_world.joint_gppo import (
    ActionConditionedTemporalWorldModel, JOINT_PROTOCOL, JointGraphPreferencePolicy,
    JointTrainConfig, masked_normalized_preference,
)
from gppo_world.joint_training import (
    GROUPS, _atomic_json, _make_optimizers, run_group, verify_commit_pointer,
)
from gppo_world.m10_environment import M10Config, scenario_from_dict
from tools.evaluate_joint_four_group_pilot import _episode, _load_policy, _summarize


RUN_ROOT = ROOT / "runs" / "joint-gradient-isolation-cpu-20260916-v1"
SOURCE_PILOT = ROOT / "runs" / "joint-four-group-single-seed-pilot-cpu-20260916-v1"
GROUP_ORDER = ("B0", "CS", "CI", "DI")
PREFERENCES = ((0.2, 0.8), (0.5, 0.5), (0.8, 0.2))
SEED = 1101


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        value = state[name].detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_frozen_tapes() -> tuple[list, list, Path]:
    path = SOURCE_PILOT / "frozen-tapes.json"
    if not path.is_file():
        raise FileNotFoundError(f"refusing to generate new tapes; frozen tape file missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    train = [scenario_from_dict(item) for item in payload["train"]]
    validation = [scenario_from_dict(item) for item in payload["validation"]]
    if len(train) != 32 or len(validation) != 16:
        raise ValueError("frozen comparison tapes are not the registered 32/16 sets")
    return train, validation, path


def base_initialization(env_config: M10Config) -> tuple[dict, dict, torch.Tensor, dict]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    policy = JointGraphPreferencePolicy(env_config, history=True)
    action_rng_after_policy = torch.get_rng_state().clone()
    world = ActionConditionedTemporalWorldModel()
    policy_state = copy.deepcopy(policy.state_dict())
    world_state = copy.deepcopy(world.state_dict())
    return policy_state, world_state, action_rng_after_policy, {
        "seed": SEED,
        "policy_state_sha256": state_hash(policy_state),
        "world_state_sha256": state_hash(world_state),
        "action_rng_state_sha256": hashlib.sha256(action_rng_after_policy.numpy().tobytes()).hexdigest(),
        "initialization_contract": "policy constructed first; world construction cannot perturb the common action RNG",
    }


def optimizer_scope_evidence(env_config: M10Config, config: JointTrainConfig) -> dict:
    evidence = {}
    for group in ("CS", "CI", "DI"):
        policy = JointGraphPreferencePolicy(env_config, history=True)
        world = ActionConditionedTemporalWorldModel()
        policy_optimizer, world_optimizer = _make_optimizers(policy, world, GROUPS[group], config)
        policy_parameter_names = {id(parameter): name for name, parameter in policy.named_parameters()}
        world_parameter_names = {id(parameter): name for name, parameter in world.named_parameters()}
        names = [policy_parameter_names.get(id(parameter), world_parameter_names.get(id(parameter), "<unknown>"))
                 for block in world_optimizer.param_groups for parameter in block["params"]]
        evidence[group] = {
            "world_optimizer_scope": "world_only" if GROUPS[group].isolate_world_gradient else "shared_encoder_plus_world",
            "parameter_count": len(names),
            "parameters": names,
            "policy_optimizer_parameter_count": sum(len(block["params"]) for block in policy_optimizer.param_groups),
            "shared_encoder_in_world_optimizer": any(name.startswith("base.") for name in names),
        }
    return evidence


def run_correctness_stage(env_config, train_config, train_tapes, init_policy, init_world, action_rng):
    stage_root = RUN_ROOT / "correctness-stage"
    stage_root.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for index, group in enumerate(GROUP_ORDER):
        out = stage_root / group
        # Four 64-step smoke runs consume exactly the registered 256 env-step
        # correctness budget.  Recovery replay adds one policy call per group;
        # total is therefore 8 policy calls, within the 8-call cap.
        summaries[group] = run_group(
            group=group, run_id=f"joint-gradient-correctness-{group}-seed1101-v1",
            output_dir=out, device="cpu", env_config=env_config,
            train_config=train_config, max_steps=64, max_policy_updates=1,
            max_world_updates=0 if group == "B0" else 4, wall_seconds=420.0,
            enforce_smoke_limits=True, verify_recovery_update=True, scenarios=train_tapes,
            initial_policy_state_dict=init_policy, initial_world_state_dict=init_world,
            initial_action_rng_state=action_rng,
        )
        verify_commit_pointer(out)
        if summaries[group].get("recovery_update_verification", {}).get("status") != "passed":
            raise RuntimeError(f"same-CPU recovery did not pass for {group}")
    _atomic_json(stage_root / "correctness-summary.json", {
        "status": "passed", "groups": summaries,
        "budget": {"environment_steps": 256, "policy_optimizer_calls": 8, "world_optimizer_calls": 12},
        "note": "world-update replay is covered by parameter-scope and finite-state checks; policy U-R replay is run by the existing recovery verifier",
    })
    return summaries


def run_pilot(env_config, train_config, train_tapes, init_policy, init_world, action_rng):
    pilot_root = RUN_ROOT / "pilot"
    pilot_root.mkdir(parents=True, exist_ok=False)
    summaries = {}
    started = time.perf_counter()
    for group in GROUP_ORDER:
        out = pilot_root / group
        wm_cap = 0 if group == "B0" else 256
        summary = run_group(
            group=group, run_id=f"joint-gradient-isolation-pilot-{group}-seed1101-v1",
            output_dir=out, device="cpu", env_config=env_config,
            train_config=train_config, max_steps=4096, max_policy_updates=32,
            max_world_updates=wm_cap, wall_seconds=3600.0,
            enforce_smoke_limits=False, verify_recovery_update=False, scenarios=train_tapes,
            initial_policy_state_dict=init_policy, initial_world_state_dict=init_world,
            initial_action_rng_state=action_rng,
        )
        summary["transaction_commit"] = verify_commit_pointer(out)
        summaries[group] = summary
        _atomic_json(out / "gradient-isolation-group-summary.json", summary)
        if summary.get("environment_steps") != 4096 or summary.get("policy_optimizer_steps") != 32:
            raise RuntimeError(f"{group} did not reach its registered policy/step budget")
        if summary.get("world_optimizer_steps") != wm_cap:
            raise RuntimeError(f"{group} did not reach its registered world-update budget")
    _atomic_json(pilot_root / "training-summary.json", {
        "status": "completed", "groups": summaries,
        "actual_total_wall_seconds": time.perf_counter() - started,
        "budgets": {"environment_steps": 16384, "policy_optimizer_steps": 128, "world_optimizer_steps": 768},
    })
    return summaries


def evaluate(pilot_summaries, validation, env_config, train_config):
    result = {"preferences": {}, "evaluation_scope": "fixed 16 validation parents; ideal communication; descriptive mechanism pilot"}
    for preference in PREFERENCES:
        key = "p" + "_".join(str(value).replace(".", "", 1) for value in preference)
        result["preferences"][key] = {}
        for group in GROUP_ORDER:
            checkpoint = RUN_ROOT / "pilot" / group / "last-recovery.pt"
            policy, world = _load_policy(group, checkpoint, env_config, torch.device("cpu"))
            rows = [_episode(group, policy, world, scenario, env_config, train_config, torch.device("cpu"), preference)
                    for scenario in validation]
            result["preferences"][key][group] = {
                "preference": preference, "summary": _summarize(rows), "episodes": rows,
                "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            }
    _atomic_json(RUN_ROOT / "validation-results.json", result)
    return result


def main() -> int:
    if RUN_ROOT.exists():
        raise SystemExit(f"refusing to overwrite existing run directory: {RUN_ROOT}")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    env_config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    train_config = JointTrainConfig(seed=SEED, rollout_steps=128)
    train_tapes, validation, tape_path = load_frozen_tapes()
    RUN_ROOT.mkdir(parents=True, exist_ok=False)
    init_policy, init_world, action_rng, init_info = base_initialization(env_config)
    _atomic_json(RUN_ROOT / "pre-registration.json", {
        "protocol": JOINT_PROTOCOL, "groups": {key: asdict(GROUPS[key]) for key in GROUP_ORDER},
        "seed": SEED, "device": "cpu", "threads": {"torch_intra": torch.get_num_threads(), "torch_inter": 1},
        "environment": asdict(env_config), "training": asdict(train_config),
        "correctness_budget": {"environment_steps": 256, "policy_optimizer_calls": 8, "world_optimizer_calls": 16, "wall_seconds": 1800},
        "pilot_budget": {"environment_steps": 16384, "policy_optimizer_calls": 128, "world_optimizer_calls": 768, "wall_seconds": 3600},
        "preferences": PREFERENCES, "tape_source": str(tape_path), "tape_source_sha256": sha256(tape_path),
        "initialization": init_info, "no_new_tapes": True,
    })
    torch.save({"policy": init_policy, "world": init_world, "action_rng": action_rng}, RUN_ROOT / "initialization.pt")
    correctness = run_correctness_stage(env_config, train_config, train_tapes, init_policy, init_world, action_rng)
    scopes = optimizer_scope_evidence(env_config, train_config)
    if not all(item["world_optimizer_scope"] == ("world_only" if GROUPS[key].isolate_world_gradient else "shared_encoder_plus_world")
               for key, item in scopes.items()):
        raise RuntimeError("optimizer parameter scope audit failed")
    pilot = run_pilot(env_config, train_config, train_tapes, init_policy, init_world, action_rng)
    evaluation = evaluate(pilot, validation, env_config, train_config)
    _atomic_json(RUN_ROOT / "experiment-summary.json", {
        "status": "completed", "correctness": correctness, "optimizer_scope": scopes,
        "pilot": pilot, "evaluation": {"path": "validation-results.json", "preferences": list(evaluation["preferences"])},
        "platform": platform.platform(), "python": sys.version, "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "numpy": np.__version__,
    })
    print(json.dumps({
        "run_root": str(RUN_ROOT),
        "correctness": {key: {"steps": value.get("environment_steps"), "policy": value.get("policy_optimizer_steps"), "world": value.get("world_optimizer_steps"), "recovery": value.get("recovery_update_verification", {}).get("status")} for key, value in correctness.items()},
        "pilot": {key: {"steps": value.get("environment_steps"), "policy": value.get("policy_optimizer_steps"), "world": value.get("world_optimizer_steps"), "stop": value.get("stop_reason")} for key, value in pilot.items()},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
