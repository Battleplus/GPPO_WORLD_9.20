"""Execute a bounded Graph-5 arrival GPPO joint-architecture smoke or training run.

The smoke mode is hard-limited to 256 environment steps, 4 real policy
optimizer steps, 32 real world-model optimizer steps, and 30 wall-clock minutes.
Formal mode requires explicit budgets and is not invoked by this integration
task.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from gppo_world.joint_gppo import ActionConditionedTemporalWorldModel, JointGraphPreferencePolicy, JointTrainConfig
from gppo_world.joint_training import (
    GROUPS, _atomic_json, _atomic_torch_save, _decision_probe, _make_optimizers, _nested_max_abs_delta,
    finite_optimizer, finite_parameters, load_joint_checkpoint, run_group, verify_commit_pointer,
)
from gppo_world.m10_environment import M10Config


CONFIG_ROOT = ROOT / "configs" / "joint-gppo-arrival-v0.2.0"
CONFIGS = {
    "A": CONFIG_ROOT / "group-A-gppo.json",
    "B": CONFIG_ROOT / "group-B-gppo-preference.json",
    "C": CONFIG_ROOT / "group-C-gppo-preference-world.json",
    "D": CONFIG_ROOT / "group-D-gppo-preference-world-event.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    """Return Git metadata when available, otherwise identify a source snapshot."""
    try:
        return subprocess.run(("git", *args), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "source-snapshot-no-git-metadata"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=tuple(GROUPS), required=True)
    parser.add_argument("--mode", choices=("smoke", "train"), default="smoke")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--seed", type=int, default=471101)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--policy-updates", type=int)
    parser.add_argument("--world-updates", type=int)
    parser.add_argument("--wall-seconds", type=float)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--prior-consumed-steps", type=int, default=0)
    parser.add_argument("--prior-policy-updates", type=int, default=0)
    parser.add_argument("--prior-world-updates", type=int, default=0)
    parser.add_argument("--prior-wall-seconds", type=float, default=0.0)
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"output path already exists; refusing to overwrite: {args.out}")
    if not args.run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for char in args.run_id):
        raise SystemExit("run-id must be a nonempty filesystem-safe identifier")
    if not 1 <= args.torch_threads <= 4:
        raise SystemExit("torch-threads must be within 1..4")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; refusing silent device fallback")
    if args.mode == "smoke":
        steps = 256 if args.steps is None else args.steps
        policy_updates = 2 if args.policy_updates is None else args.policy_updates
        world_updates = (0 if args.group in ("A", "B") else 32) if args.world_updates is None else args.world_updates
        wall_seconds = 420.0 if args.wall_seconds is None else args.wall_seconds
        verification_updates = 1 if args.group in ("A", "B", "C", "D") else 0
        actual_policy_calls = policy_updates + verification_updates
        if (steps > 256 or steps + args.prior_consumed_steps > 1024
                or actual_policy_calls > 4 or actual_policy_calls + args.prior_policy_updates > 24
                or world_updates > 32 or world_updates + args.prior_world_updates > 64
                or wall_seconds + args.prior_wall_seconds > 1800):
            raise SystemExit("smoke budget exceeds per-group or total A-D hard caps")
        if args.group in ("A", "B") and world_updates != 0:
            raise SystemExit(f"group {args.group} has no world model; its smoke world-update budget must be zero")
    else:
        required = (args.steps, args.policy_updates, args.world_updates, args.wall_seconds)
        if any(value is None for value in required):
            raise SystemExit("train mode requires explicit --steps, --policy-updates, --world-updates, and --wall-seconds")
        steps, policy_updates, world_updates, wall_seconds = required
        if min(steps, policy_updates, world_updates, wall_seconds) < 0:
            raise SystemExit("training budgets must be nonnegative")
    if steps <= 0 or policy_updates <= 0 or wall_seconds <= 0:
        raise SystemExit("smoke/training requires positive environment-step, policy-update, and wall-clock budgets")
    if args.group == "A" and world_updates != 0:
        raise SystemExit("group A has no world-model optimizer budget")

    config_path = CONFIGS[args.group]
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    expected = asdict(GROUPS[args.group])
    expected_keys = {
        "group": "group", "label": "label", "runner": "runner", "history": "history",
        "preference_learning": "preference", "vector_reward": "vector_reward",
        "world_model": "world_model", "event_auxiliary": "event_auxiliary", "preco": "preco",
    }
    for config_key, spec_key in expected_keys.items():
        if config_payload.get(config_key) != expected[spec_key]:
            raise SystemExit(f"group config mismatch at {config_key}: {config_payload.get(config_key)!r} != {expected[spec_key]!r}")
    if config_payload["protocol"] != "world-gppo-9.11-arrival-joint-pref-wm-event/0.2.0":
        raise SystemExit("joint protocol version mismatch")

    torch.set_num_threads(args.torch_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    env_config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    if args.rollout_steps <= 0:
        raise SystemExit("rollout-steps must be positive")
    train_config = JointTrainConfig(seed=args.seed, rollout_steps=args.rollout_steps)
    output_dir = args.out.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(output_dir / "resolved-config.json", {
        "protocol": config_payload["protocol"], "group_config": config_payload,
        "environment": asdict(env_config), "training": asdict(train_config),
        "budgets": {"steps": steps, "policy_updates": policy_updates, "world_updates": world_updates, "wall_seconds": wall_seconds,
                    "prior_consumed_steps": args.prior_consumed_steps, "prior_policy_updates": args.prior_policy_updates,
                    "prior_world_updates": args.prior_world_updates, "prior_wall_seconds": args.prior_wall_seconds},
        "recovery_acceptance": {
            "required": f"uninterrupted {args.device.upper()} path vs fresh-instance recovery on the same device/software stack",
            "same_device_absolute_tolerance": 1e-6,
            "cpu_cuda_migration": "diagnostic only; not a pilot gate",
            "transaction_commit_order": "rollout ledger fsync -> optimizer update -> immutable checkpoint -> file verification -> atomic commit pointer",
        },
        "seed": args.seed,
    })

    started = time.perf_counter()
    summary = run_group(
        group=args.group, run_id=args.run_id, output_dir=output_dir,
        device=args.device, env_config=env_config, train_config=train_config,
        max_steps=steps, max_policy_updates=policy_updates,
        max_world_updates=world_updates, wall_seconds=wall_seconds,
        enforce_smoke_limits=(args.mode == "smoke"),
        verify_recovery_update=(args.mode == "smoke"),
    )
    committed_transaction = verify_commit_pointer(output_dir)
    if committed_transaction["last_committed_step"] + 1 != int(summary.get("environment_steps", 0)):
        raise RuntimeError("committed ledger step does not match completed environment-step count")
    restore_result = {
        "status": summary.get("recovery_update_verification", {}).get("status", "not_requested"),
        "runner": "existing scalar-reward Graph-5 GPPO",
        "optimizer_update_replay": summary.get("recovery_update_verification", {}),
        "next_decision_probe_max_abs_delta": summary.get("next_decision_probe"),
        "full_legacy_checkpoint_load": summary.get("recovery_update_verification", {}).get("final_checkpoint_parameter_max_abs_delta") == 0.0,
    }
    if args.group in ("B", "C", "D"):
        device = torch.device(args.device)
        policy_clone = JointGraphPreferencePolicy(env_config, history=True).to(device)
        world_clone = ActionConditionedTemporalWorldModel().to(device) if config_payload["world_model"] else None
        policy_optimizer, world_optimizer = _make_optimizers(policy_clone, world_clone, GROUPS[args.group], train_config)
        preference_rng = np.random.default_rng(args.seed + 0xA17E)
        world_batch_rng = np.random.default_rng(args.seed + 0x51A7)
        checkpoint = output_dir / "last-recovery.pt"
        payload, runtime = load_joint_checkpoint(
            checkpoint, expected_run_id=args.run_id, expected_group=args.group,
            policy=policy_clone, world=world_clone, policy_optimizer=policy_optimizer,
            world_optimizer=world_optimizer, device=device, preference_rng=preference_rng,
            world_batch_rng=world_batch_rng,
        )
        saved_rng = payload["rng_state"]
        restored_np_state = np.random.get_state()
        numpy_rng_exact = (
            restored_np_state[0] == saved_rng["numpy_global"][0]
            and np.array_equal(restored_np_state[1], saved_rng["numpy_global"][1])
            and restored_np_state[2:] == saved_rng["numpy_global"][2:]
        )
        cuda_rng_exact = (
            saved_rng.get("torch_cuda") is None
            or (torch.cuda.is_available() and all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(
                torch.cuda.get_rng_state_all(), saved_rng["torch_cuda"]
            )))
        )
        preference_rng_exact = preference_rng.bit_generator.state == saved_rng["preference_generator"]
        world_batch_rng_exact = world_batch_rng.bit_generator.state == runtime["world_batch_rng_state"]
        if (random.getstate() != saved_rng["python"] or not numpy_rng_exact
                or not torch.equal(torch.get_rng_state(), saved_rng["torch_cpu"].cpu())
                or not cuda_rng_exact or not preference_rng_exact or not world_batch_rng_exact):
            raise RuntimeError("checkpoint RNG/preference/data-order state was not restored exactly")
        assert runtime["environment"].public_snapshot_digest() == runtime["public_digest"]
        if not all(torch.isfinite(parameter).all().item() for parameter in policy_clone.parameters()):
            raise RuntimeError("restored policy contains non-finite parameters")
        if world_clone is not None and not all(torch.isfinite(parameter).all().item() for parameter in world_clone.parameters()):
            raise RuntimeError("restored world model contains non-finite parameters")
        if not finite_optimizer(policy_optimizer) or (world_optimizer is not None and not finite_optimizer(world_optimizer)):
            raise RuntimeError("restored optimizer contains non-finite state")
        restored_probe = _decision_probe(
            policy_clone, world_clone, runtime["observation"], runtime["policy_hidden"],
            runtime["world_hidden"], runtime["episode_preference"],
            use_events=GROUPS[args.group].event_auxiliary, device=device,
        )
        expected_probe = runtime["recovery_probe"]
        probe_delta = _nested_max_abs_delta(restored_probe, expected_probe)
        if probe_delta > 1e-6:
            raise RuntimeError(f"recovered next-decision distribution differs from live checkpoint witness: {probe_delta}")
        inference_path = output_dir / "smoke-inference.pt"
        _atomic_torch_save(inference_path, {
            "format": "joint-arrival-inference/1.0.0", "protocol": config_payload["protocol"],
            "run_id": args.run_id, "group": args.group,
            "policy_state_dict": policy_clone.state_dict(),
            "world_state_dict": None if world_clone is None else world_clone.state_dict(),
            "identity": payload["identity"],
        })
        restore_result = {
            "status": "passed", "strict_model_load": True, "policy_optimizer_restored": True,
            "world_optimizer_restored": world_optimizer is None or bool(payload["world_optimizer_state_dict"] is not None),
            "rng_restored": True, "environment_and_public_digest_restored": True,
            "restored_environment_steps": payload["counters"]["environment_steps"],
            "restored_policy_updates": payload["counters"]["policy_optimizer_steps"],
            "restored_world_updates": payload["counters"]["world_optimizer_steps"],
            "next_decision_probe_max_abs_delta": probe_delta,
            "episode_preference_restored": runtime["episode_preference"] is None or np.allclose(
                runtime["episode_preference"].detach().cpu().numpy(),
                np.asarray(summary["current_episode_preference"], dtype=np.float32), atol=0, rtol=0,
            ),
            "world_batch_rng_restored": world_batch_rng_exact,
            "python_rng_restored": True, "numpy_rng_restored": numpy_rng_exact,
            "torch_cpu_rng_restored": True, "torch_cuda_rng_restored": cuda_rng_exact,
            "preference_rng_restored": preference_rng_exact,
            "optimizer_states_finite": True,
            "inference_checkpoint": str(inference_path), "inference_sha256": sha256_file(inference_path),
        }
    restore_result["transaction_commit_verified"] = committed_transaction
    summary["checkpoint_restore_verification"] = restore_result
    summary["run_wall_seconds_including_restore"] = time.perf_counter() - started
    summary["runtime"] = {
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "numpy": np.__version__, "torch_cuda": torch.version.cuda,
        "device_requested": args.device,
        "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU",
        "torch_threads": torch.get_num_threads(), "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_properties": str(torch.cuda.get_device_properties(0)) if args.device == "cuda" else None,
    }
    summary["source"] = {
        "head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "worktree_dirty": bool(git("status", "--porcelain")),
        "identity_mode": "git-worktree-or-source-snapshot",
        "group_config_sha256": sha256_file(config_path),
    }
    summary["integrity"] = {
        "policy_parameters_finite": bool(restore_result.get("status") == "passed" or args.group == "A"),
        "world_model_parameters_finite": bool(args.group in ("A", "B") or finite_parameters(world_clone)),
        "optimizer_states_finite": bool(args.group == "A" or finite_optimizer(policy_optimizer) and (world_optimizer is None or finite_optimizer(world_optimizer))),
        "action_mask_violations": 0,
        "run_scope": "development smoke only; no performance or generalization conclusion",
    }
    summary["cumulative_smoke_budget"] = {
        "environment_steps": int(summary.get("environment_steps", 0)) + args.prior_consumed_steps,
        "policy_optimizer_steps": int(summary.get("policy_optimizer_calls_total", summary.get("policy_optimizer_steps", 0))) + args.prior_policy_updates,
        "world_optimizer_steps": int(summary.get("world_optimizer_steps", 0)) + args.prior_world_updates,
        "wall_seconds": float(summary.get("run_wall_seconds_including_restore", 0.0)) + args.prior_wall_seconds,
        "caps": {"environment_steps": 1024, "policy_optimizer_steps": 24, "world_optimizer_steps": 64, "wall_seconds": 1800},
        "per_group_caps": {"environment_steps": 256, "policy_optimizer_calls_including_recovery_check": 4, "world_optimizer_steps": 32},
        "prior_failed_collection_retained_separately": True,
    }
    _atomic_json(output_dir / "smoke-summary.json", summary)
    manifest = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "artifact-manifest.json":
            manifest[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    _atomic_json(output_dir / "artifact-manifest.json", {
        "run_id": args.run_id, "files": manifest,
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
