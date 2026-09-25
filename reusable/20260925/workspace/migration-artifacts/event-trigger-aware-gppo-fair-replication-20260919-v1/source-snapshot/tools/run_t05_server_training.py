"""Run one frozen T-05 group x seed training job on a CUDA server."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def require_clean_commit(path: Path, expected: str, label: str) -> str:
    observed = git(path, "rev-parse", "HEAD")
    if observed != expected:
        raise SystemExit(f"{label} commit mismatch: expected {expected}, got {observed}")
    if git(path, "status", "--porcelain"):
        raise SystemExit(f"{label} worktree must be clean")
    return observed


def inventory(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=("GPPO", "WM-GPPO", "EA-noGES-GPPO", "EAWM-GPPO"))
    parser.add_argument("seed", type=int, choices=(1101, 2202, 3303))
    parser.add_argument("run_dir")
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--world-checkpoint-dir", required=True)
    parser.add_argument("--expected-target-commit", required=True)
    parser.add_argument("--config", default=str(ROOT / "nodes" / "T-05" / "server-training-config.json"))
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    frozen_config_path = (ROOT / "nodes" / "T-05" / "server-training-config.json").resolve()
    if config_path != frozen_config_path:
        raise SystemExit("formal T-05 only accepts the repository-frozen server config")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    baseline_root = Path(args.baseline_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    if f"{sys.version_info.major}.{sys.version_info.minor}" not in config["runtime"]["allowed_python"]:
        raise SystemExit("formal T-05 requires Python 3.10 or 3.11")
    if not torch.cuda.is_available():
        raise SystemExit("formal T-05 refuses to run without CUDA")
    target_commit = require_clean_commit(ROOT, args.expected_target_commit, "target")
    baseline_commit = require_clean_commit(
        baseline_root, config["baseline"]["commit"], "baseline"
    )
    for relative, expected in (
        ("configs/random_event_protocol.json", config["baseline"]["protocol_sha256"]),
        ("configs/seed_manifest.json", config["baseline"]["seed_manifest_sha256"]),
    ):
        observed = sha256(baseline_root / relative)
        if observed != expected:
            raise SystemExit(f"baseline {relative} SHA-256 mismatch")
    calibration_path = ROOT / config["shadow_calibration"]["path"]
    if sha256(calibration_path) != config["shadow_calibration"]["sha256"]:
        raise SystemExit("Shadow calibration SHA-256 mismatch")

    # Baseline imports are intentionally delayed until its frozen root is checked.
    sys.path.insert(0, str(baseline_root))
    from ppo_allocation.random_event.experiment import (  # noqa: PLC0415
        CyclingTrainingEnv,
        _load_frozen_train_episode_cap,
        _load_frozen_train_modes,
    )
    from ppo_allocation.random_event.models import GraphActorCritic  # noqa: PLC0415
    from ppo_allocation.random_event.trainer import PPOConfig, PPOTrainer  # noqa: PLC0415
    from gppo_world.calibration import ShadowCalibration  # noqa: PLC0415
    from gppo_world.gppo_adapter import (  # noqa: PLC0415
        LatentAdapterConfig,
        LatentAugmentedActorCritic,
        LatentContextStore,
        freeze_world_model,
    )
    from gppo_world.gppo_shadow_env import (  # noqa: PLC0415
        PostActionShadowEnv,
        ensure_noop_action_compatibility,
    )
    from gppo_world.gppo_trainer import LatentPPOTrainer  # noqa: PLC0415
    from gppo_world.model import EventAwareGraphWorldModel  # noqa: PLC0415
    from gppo_world.shadow import ShadowRuntime  # noqa: PLC0415

    training = config["training"]
    ppo_config = PPOConfig(
        rollout_steps=training["rollout_steps"],
        learning_rate=training["learning_rate"],
        gamma=training["gamma"],
        gae_lambda=training["gae_lambda"],
        clip_coef=training["clip_coef"],
        value_coef=training["value_coef"],
        entropy_coef=training["entropy_coef"],
        update_epochs=training["update_epochs"],
        minibatch_size=training["minibatch_size"],
        max_grad_norm=training["max_grad_norm"],
        seed=args.seed,
        device="cuda",
    )
    modes = tuple(_load_frozen_train_modes())
    if modes != tuple(training["mode_cycle"]):
        raise SystemExit("training mode cycle differs from the frozen protocol")
    raw_env = CyclingTrainingEnv(
        seed=args.seed,
        modes=modes,
        events_per_episode=training["events_per_tape"],
        max_resets=_load_frozen_train_episode_cap(),
    )
    noop_compatibility_installed = ensure_noop_action_compatibility(raw_env)

    world_path: Path | None = None
    world_sha: str | None = None
    shadow_runtime = None
    context_store = None
    env: Any = raw_env
    if args.group != "GPPO":
        world_seed = config["world_seed_by_training_seed"][str(args.seed)]
        pattern = config["groups"][args.group]["world_model_pattern"]
        world_path = Path(args.world_checkpoint_dir).resolve() / pattern.format(world_seed=world_seed)
        world_sha = sha256(world_path)
        expected_world_sha = config["world_checkpoint_sha256"][world_path.name]
        if world_sha != expected_world_sha:
            raise SystemExit(f"world checkpoint SHA-256 mismatch: {world_path.name}")
        world_model, _ = EventAwareGraphWorldModel.load(world_path, map_location="cpu")
        freeze_world_model(world_model)
        adapter_config = LatentAdapterConfig(
            latent_dim=world_model.config.hidden_dim + world_model.config.stochastic_dim
        )
        calibration = ShadowCalibration.from_dict(
            json.loads(calibration_path.read_text(encoding="utf-8"))
        )
        context_store = LatentContextStore(
            adapter_config,
            model_variant=args.group,
            model_version=f"{world_path.name}:{world_sha}",
        )
        shadow_runtime = ShadowRuntime(
            world_model,
            calibration,
            model_version=f"{world_path.name}:{world_sha}",
        )
        env = PostActionShadowEnv(raw_env, shadow_runtime, context_store, model_variant=args.group)

    # Consume the same frozen episode index 0 for every group, then bind the
    # returned graph as trainer current state so initialization never resets twice.
    graph, _ = env.reset()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.group == "GPPO":
        trainer: Any = PPOTrainer(env=env, variant="GPPO-Adaptive", config=ppo_config)
        trainer._current_graph = graph
        trainer.initialize(graph)
    else:
        base = GraphActorCritic.from_graph(graph)
        policy = LatentAugmentedActorCritic(
            base,
            adapter_config,
            context_store=context_store,
            enabled=True,
            model_variant=args.group,
        )
        trainer = LatentPPOTrainer(
            env=env,
            variant="GPPO-Adaptive",
            config=ppo_config,
            model=policy,
        )
        trainer._current_graph = graph
    # Re-anchor the policy/action/minibatch RNG after every group has finished
    # constructing its (different-sized) module tree.  This prevents adapter
    # initialization draws from shifting the common training seed stream.
    trainer._seed_everything(args.seed)

    parameter_counts = {
        "policy_total": sum(parameter.numel() for parameter in trainer.model.parameters()),
        "policy_trainable": sum(
            parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad
        ),
        "world_total": 0 if shadow_runtime is None else sum(
            parameter.numel() for parameter in shadow_runtime.model.parameters()
        ),
        "world_trainable": 0 if shadow_runtime is None else sum(
            parameter.numel() for parameter in shadow_runtime.model.parameters() if parameter.requires_grad
        ),
    }
    if parameter_counts["world_trainable"] != 0:
        raise SystemExit("world model is not completely frozen")
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "cudnn": torch.backends.cudnn.version(),
        "hostname": platform.node(),
    }
    manifest = {
        "format": config["format"],
        "group": args.group,
        "seed": args.seed,
        "target_commit": target_commit,
        "baseline_commit": baseline_commit,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "world_checkpoint": None if world_path is None else world_path.name,
        "world_checkpoint_sha256": world_sha,
        "calibration_sha256": sha256(calibration_path),
        "parameter_counts": parameter_counts,
        "baseline_noop_compatibility_installed": noop_compatibility_installed,
        "environment": environment,
        "pid": os.getpid(),
    }
    write_json(run_dir / "run-manifest.json", manifest)
    write_json(run_dir / "environment.json", environment)
    history_path = run_dir / "history.jsonl"
    progress_path = run_dir / "progress" / "live_progress.json"
    started = time.perf_counter()
    checkpoints = []

    def heartbeat(current: Any) -> None:
        elapsed = max(time.perf_counter() - started, 1e-9)
        write_json(
            progress_path,
            {
                "status": "running",
                "group": args.group,
                "seed": args.seed,
                "accepted_decision_steps": current.total_steps,
                "target": training["accepted_decision_steps_per_run"],
                "elapsed_seconds": elapsed,
                "steps_per_second": current.total_steps / elapsed,
            },
        )

    for checkpoint_step in training["checkpoint_steps"]:
        trainer.train(checkpoint_step - trainer.total_steps, progress_callback=heartbeat)
        torch.cuda.synchronize()
        checkpoint_path = run_dir / "models" / (
            f"{args.group.lower().replace('-', '_')}_seed{args.seed}_step{checkpoint_step}.pt"
        )
        trainer.save(
            checkpoint_path,
            extra={
                "t05_group": args.group,
                "training_seed": args.seed,
                "accepted_decision_steps": checkpoint_step,
                "target_commit": target_commit,
                "baseline_commit": baseline_commit,
                "world_checkpoint_sha256": world_sha,
                "t05_config_sha256": sha256(config_path),
            },
        )
        checkpoints.append(
            {
                "step": checkpoint_step,
                "path": checkpoint_path.relative_to(run_dir).as_posix(),
                "sha256": sha256(checkpoint_path),
            }
        )
    history_path.write_text(
        "".join(json.dumps(json_safe(row), sort_keys=True, allow_nan=False) + "\n" for row in trainer.history),
        encoding="utf-8",
    )
    elapsed = time.perf_counter() - started
    safety = (
        {
            **asdict(env.audit()),
            "world_parameters_frozen": parameter_counts["world_trainable"] == 0,
        }
        if isinstance(env, PostActionShadowEnv)
        else {
            "world_parameters_frozen": True,
            "belief_write_count": 0,
            "action_mask_write_count": 0,
            "graph_version_write_count": 0,
            "action_version_write_count": 0,
            "action_submission_count": 0,
            "note": "GPPO control has no world-model runtime",
        }
    )
    shadow_latencies = (
        []
        if shadow_runtime is None
        else [float(item.latency_ms) for item in shadow_runtime.records]
    )
    latency_tensor = (
        None if not shadow_latencies else torch.tensor(shadow_latencies, dtype=torch.float64)
    )
    shadow_latency = {
        "count": len(shadow_latencies),
        "p50_ms": None if latency_tensor is None else float(torch.quantile(latency_tensor, 0.50)),
        "p95_ms": None if latency_tensor is None else float(torch.quantile(latency_tensor, 0.95)),
        "p99_ms": None if latency_tensor is None else float(torch.quantile(latency_tensor, 0.99)),
        "max_ms": None if latency_tensor is None else float(latency_tensor.max()),
        "fallback_count": 0 if shadow_runtime is None else shadow_runtime.counters["fallback_count"],
        "timeout_count": 0 if shadow_runtime is None else shadow_runtime.counters["timeout_count"],
        "p95_budget_ms": None if shadow_runtime is None else shadow_runtime.calibration.latency_p95_budget_ms,
        "p99_budget_ms": None if shadow_runtime is None else shadow_runtime.calibration.latency_p99_budget_ms,
    }
    shadow_latency["p95_within_budget"] = (
        None
        if shadow_runtime is None
        else shadow_latency["p95_ms"] <= shadow_latency["p95_budget_ms"]
    )
    shadow_latency["p99_within_budget"] = (
        None
        if shadow_runtime is None
        else shadow_latency["p99_ms"] <= shadow_latency["p99_budget_ms"]
    )
    if sum(
        int(safety.get(key, 0))
        for key in (
            "belief_write_count",
            "action_mask_write_count",
            "graph_version_write_count",
            "action_version_write_count",
            "action_submission_count",
        )
    ) != 0:
        raise SystemExit("T-05 safety write counter is non-zero")
    write_json(run_dir / "safety-audit.json", safety)
    write_json(
        run_dir / "run-summary.json",
        {
            **manifest,
            "status": "done",
            "accepted_decision_steps": trainer.total_steps,
            "elapsed_seconds": elapsed,
            "checkpoints": checkpoints,
            "latent_context_counters": None if context_store is None else context_store.counters,
            "safety": safety,
            "shadow_latency": shadow_latency,
        },
    )
    write_json(
        progress_path,
        {
            "status": "done",
            "group": args.group,
            "seed": args.seed,
            "accepted_decision_steps": trainer.total_steps,
            "target": training["accepted_decision_steps_per_run"],
            "elapsed_seconds": elapsed,
        },
    )
    # The inventory intentionally excludes itself and is generated only after
    # every other file has reached its final content.
    write_json(run_dir / "sha256-inventory.json", {"files": inventory(run_dir)})
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
