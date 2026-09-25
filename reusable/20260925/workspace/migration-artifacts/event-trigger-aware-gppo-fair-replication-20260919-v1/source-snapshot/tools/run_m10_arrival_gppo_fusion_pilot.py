"""Run the bounded arrival-protocol GPPO fusion pilot.

This is the first runner that puts candidate consequence predictions inside
the PPO distribution used for both sampling and log-probability updates.  It
is intentionally a pilot: one seed, fixed train/validation tapes, 512
environment steps per learning group, and no model or notification training.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import random
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
    CandidateFeatureContract,
    FrozenArrivalConsequenceScorer,
    choose_candidate_action,
)
from gppo_world.m10_environment import (  # noqa: E402
    M10Config,
    M10Environment,
    M10Scenario,
    scenario_to_dict,
    weak_communication_tape,
)
from gppo_world.m10_training import (  # noqa: E402
    M10ActorCritic,
    PPOConfig,
    Transition,
    _gae,
    _act,
    masked_distribution,
    save_policy,
    seed_everything,
    train_policy,
)
from tools.run_m10_baseline_comparison import traditional_action  # noqa: E402


SEED = 1101
ENV_STEPS = 512
MAX_OPTIMIZER_UPDATES = 128
GROUP_WALL_SECONDS = 30 * 60


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


@dataclass
class CandidateTransition(Transition):
    candidate_features: np.ndarray | None = None


def finite_module(module: torch.nn.Module, label: str) -> None:
    for name, value in module.state_dict().items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"non-finite {label}: {name}")


def collect_candidate_rollout(policy: CandidateAwarePolicy, scorer: FrozenArrivalConsequenceScorer,
                              *, config: PPOConfig, env_config: M10Config, seed: int,
                              device: torch.device, scenarios: list[M10Scenario]) -> list[CandidateTransition]:
    if not scenarios:
        raise ValueError("candidate rollout requires a non-empty frozen tape")
    env = M10Environment(env_config, scenarios[0])
    obs = env.reset()
    hidden = None
    transitions: list[CandidateTransition] = []
    scenario_index = 0
    episode_start = True
    for step_index in range(config.rollout_steps):
        start = time.perf_counter()
        features = scorer.score_observation(obs)
        dense = features.dense().detach().cpu().numpy().astype(np.float32)
        public = torch.as_tensor(obs["flat"], dtype=torch.float32, device=device)[None, :]
        candidate_tensor = torch.as_tensor(dense, dtype=torch.float32, device=device)[None, :, :]
        logits, value_tensor, hidden = policy(public, candidate_tensor, hidden)
        mask = torch.as_tensor(obs["mask"], dtype=torch.bool, device=device)[None, :]
        distribution = masked_distribution(logits, mask)
        action_tensor = distribution.sample()
        action = int(action_tensor.item())
        log_prob = float(distribution.log_prob(action_tensor).item())
        value = float(value_tensor.item())
        next_obs, reward, done, info = env.step(action, submit_command=True)
        info = dict(info)
        info.update({
            "world_model_calls": 1,
            "candidate_prediction_consumed": True,
            "candidate_feature_rows": int(np.sum(np.asarray(obs["mask"], dtype=bool))),
            "decision_chain_wall_ms": (time.perf_counter() - start) * 1000.0,
        })
        terminated = bool(info.get("terminated", done))
        truncated = bool(info.get("truncated", False))
        if not done and step_index == config.rollout_steps - 1:
            truncated = True
        if terminated:
            next_value = 0.0
        else:
            next_public = torch.as_tensor(next_obs["flat"], dtype=torch.float32, device=device)[None, :]
            next_value_tensor, _ = policy.value_only(next_public, hidden)
            next_value = float(next_value_tensor.item())
        transitions.append(CandidateTransition(
            obs=np.asarray(obs["flat"], dtype=np.float32),
            context=np.zeros(0, dtype=np.float32),
            mask=np.asarray(obs["mask"], dtype=bool).copy(),
            action=action,
            log_prob=log_prob,
            value=value,
            reward=float(reward),
            done=bool(terminated or truncated),
            episode_start=episode_start,
            info=info,
            terminated=terminated,
            truncated=truncated,
            next_value=next_value,
            actor_decision=True,
            candidate_features=dense,
        ))
        if done:
            scenario_index = (scenario_index + 1) % len(scenarios)
            env = M10Environment(env_config, scenarios[scenario_index])
            obs = env.reset()
            hidden = None
            episode_start = True
        else:
            obs = next_obs
            episode_start = False
    return transitions


def evaluate_candidate_sequence(policy: CandidateAwarePolicy, transitions: list[CandidateTransition],
                                device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_probs, values, entropies = [], [], []
    hidden = None
    for transition in transitions:
        if transition.episode_start:
            hidden = None
        obs = torch.as_tensor(transition.obs, dtype=torch.float32, device=device)[None, :]
        features = torch.as_tensor(transition.candidate_features, dtype=torch.float32, device=device)[None, :, :]
        mask = torch.as_tensor(transition.mask, dtype=torch.bool, device=device)[None, :]
        logits, value, hidden = policy(obs, features, hidden)
        distribution = masked_distribution(logits, mask)
        action = torch.tensor([transition.action], dtype=torch.long, device=device)
        log_probs.append(distribution.log_prob(action)[0])
        values.append(value[0])
        entropies.append(distribution.entropy()[0])
    return torch.stack(log_probs), torch.stack(values), torch.stack(entropies)


def update_candidate_policy(policy: CandidateAwarePolicy, transitions: list[CandidateTransition], *, ppo: PPOConfig,
                            device: torch.device, max_updates: int) -> dict[str, Any]:
    advantages, returns = _gae(transitions, ppo.gamma, ppo.gae_lambda)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    old_log_probs = torch.as_tensor([item.log_prob for item in transitions], dtype=torch.float32, device=device)
    advantages, returns = advantages.to(device), returns.to(device)
    history: list[dict[str, float]] = []
    optimizer_steps = 0
    for _ in range(ppo.update_epochs):
        if optimizer_steps >= max_updates:
            break
        new_log_probs, values, entropy = evaluate_candidate_sequence(policy, transitions, device)
        ratio = (new_log_probs - old_log_probs).exp()
        clipped = torch.clamp(ratio, 1.0 - ppo.clip_epsilon, 1.0 + ppo.clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
        value_loss = torch.nn.functional.mse_loss(values, returns)
        entropy_mean = entropy.mean()
        loss = policy_loss + ppo.value_weight * value_loss - ppo.entropy_weight * entropy_mean
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite candidate PPO loss")
        policy.optimizer.zero_grad(set_to_none=True)  # type: ignore[attr-defined]
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), ppo.grad_clip))
        if not np.isfinite(grad_norm):
            raise FloatingPointError("non-finite candidate PPO gradient")
        policy.optimizer.step()  # type: ignore[attr-defined]
        finite_module(policy, "candidate policy after optimizer step")
        for state in policy.optimizer.state.values():  # type: ignore[attr-defined]
            for value in state.values():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise FloatingPointError("non-finite candidate optimizer state")
        optimizer_steps += 1
        history.append({
            "optimizer_step": float(optimizer_steps),
            "loss": float(loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "entropy": float(entropy_mean.detach().cpu()),
            "grad_norm": grad_norm,
        })
    return {"optimizer_steps": optimizer_steps, "updates": history}


def train_candidate_policy(*, env_config: M10Config, scenarios: list[M10Scenario], scorer: FrozenArrivalConsequenceScorer,
                           seed: int, device: str, steps: int, max_updates: int, max_wall_seconds: float) -> tuple[CandidateAwarePolicy, dict[str, Any]]:
    seed_everything(seed)
    device_obj = torch.device(device)
    base = M10ActorCritic(
        uav_count=env_config.uav_count, task_capacity=env_config.task_capacity,
        action_count=env_config.action_count, encoder="graph", type_count=5,
        history=True, context_dim=0, region_count=env_config.region_count,
        target_count=env_config.target_count, event_capacity=env_config.event_capacity,
        relation_width=env_config.relation_width,
    )
    policy = CandidateAwarePolicy(base).to(device_obj)
    policy.optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)  # type: ignore[attr-defined]
    started = time.perf_counter()
    total_steps = 0
    optimizer_steps = 0
    updates: list[dict[str, Any]] = []
    ppo = PPOConfig(rollout_steps=min(256, steps), update_epochs=4)
    while total_steps < steps:
        elapsed = time.perf_counter() - started
        if elapsed >= max_wall_seconds:
            break
        rollout = collect_candidate_rollout(
            policy, scorer, config=PPOConfig(**{**asdict(ppo), "rollout_steps": min(ppo.rollout_steps, steps - total_steps)}),
            env_config=env_config, seed=seed + total_steps, device=device_obj, scenarios=scenarios,
        )
        update = update_candidate_policy(policy, rollout, ppo=ppo, device=device_obj, max_updates=max_updates - optimizer_steps)
        updates.append({"environment_steps": len(rollout), **update})
        total_steps += len(rollout)
        optimizer_steps += int(update["optimizer_steps"])
        if optimizer_steps >= max_updates:
            break
    elapsed = time.perf_counter() - started
    metadata = {
        "variant": "GPPO-History-CandidateArrival",
        "seed": seed,
        "steps": total_steps,
        "optimizer_updates": optimizer_steps,
        "elapsed_seconds": elapsed,
        "steps_per_second": total_steps / max(elapsed, 1e-9),
        "world_model_calls": total_steps,
        "candidate_predictions_consumed": total_steps,
        "env_config": asdict(env_config),
        "ppo_config": asdict(ppo),
        "updates": updates,
        "stop_reason": "max_optimizer_updates" if optimizer_steps >= max_updates else ("max_wall_seconds" if elapsed >= max_wall_seconds else "environment_step_budget"),
        "device": str(device_obj),
    }
    return policy, metadata


@torch.no_grad()
def evaluate_group(policy: torch.nn.Module, scenarios: list[M10Scenario], config: M10Config, *, device: str,
                   scorer: FrozenArrivalConsequenceScorer | None, variant: str) -> dict[str, Any]:
    device_obj = torch.device(device)
    if policy is not None:
        policy.eval()
    records: list[dict[str, Any]] = []
    for scenario in scenarios:
        env = M10Environment(config, scenario)
        obs = env.reset()
        hidden = None
        done = False
        steps = 0
        total_reward = 0.0
        actions: list[dict[str, Any]] = []
        world_calls = 0
        actor_calls = 0
        wall_ms: list[float] = []
        info: dict[str, Any] = {"counts": {"completed": 0, "expired": 0, "rejected": 0}, "energy": {}}
        while not done and steps < int(config.horizon / config.decision_interval) + 2:
            start = time.perf_counter()
            if policy is None:
                action = traditional_action(obs)
                consumed = False
            elif scorer is None:
                vector = np.asarray(obs["flat"], dtype=np.float32)
                action, _, _, hidden = _act(policy, vector, obs["mask"], hidden, device_obj, deterministic=True)  # type: ignore[arg-type]
                consumed = False
            else:
                action, _, features, hidden = choose_candidate_action(policy, scorer, obs, device=device_obj, hidden=hidden, deterministic=True)  # type: ignore[arg-type]
                consumed = True
                world_calls += 1
            actor_calls += 1
            actions.append({"time": float(obs["time"]), "action": action, "candidate_prediction_consumed": consumed})
            obs, reward, done, info = env.step(action, submit_command=True)
            total_reward += float(reward)
            steps += 1
            if device_obj.type == "cuda":
                torch.cuda.synchronize(device_obj)
            wall_ms.append((time.perf_counter() - start) * 1000.0)
        comm = list(info.get("communication_log", []))
        execution = list(getattr(env.execution, "log", []))
        records.append({
            "tape_id": scenario.tape_id,
            "scenario_seed": scenario.seed,
            "steps": steps,
            "return": total_reward,
            "counts": info.get("counts", {}),
            "tasks": info.get("tasks", []),
            "completion_records": info.get("completion_records", {}),
            "actor_calls": actor_calls,
            "world_model_calls": world_calls,
            "actions": actions,
            "communication_count": len(comm),
            "communication_proxy_bytes": len(json.dumps(comm, sort_keys=True, default=str).encode("utf-8")),
            "safety": {
                "duplicate_accepts": sum(1 for x in execution if x.get("result") == "accepted") - len({x.get("command_id") for x in execution if x.get("result") == "accepted"}),
                "unauthorized_or_fenced": sum(x.get("result") in ("stale", "fenced", "expired", "unknown_command") for x in execution),
            },
            "decision_latency_ms": {"n": len(wall_ms), "mean": float(np.mean(wall_ms)) if wall_ms else None, "p95": float(np.percentile(wall_ms, 95)) if wall_ms else None, "raw": wall_ms},
        })
    def mean(key: str) -> float:
        return float(np.mean([record.get(key, 0.0) for record in records])) if records else 0.0
    return {
        "variant": variant,
        "protocol": "world-gppo-9.11-arrival/0.1.0",
        "episodes": records,
        "summary": {
            "episodes": len(records),
            "completed_mean": float(np.mean([record["counts"].get("completed", 0) for record in records])) if records else 0.0,
            "expired_mean": float(np.mean([record["counts"].get("expired", 0) for record in records])) if records else 0.0,
            "return_mean": mean("return"),
            "actor_calls": sum(record["actor_calls"] for record in records),
            "world_model_calls": sum(record["world_model_calls"] for record in records),
            "communication_proxy_bytes": sum(record["communication_proxy_bytes"] for record in records),
            "safety_violations": sum(record["safety"]["duplicate_accepts"] + record["safety"]["unauthorized_or_fenced"] for record in records),
            "decision_latency_ms": {
                "n": sum(record["decision_latency_ms"]["n"] for record in records),
                "mean": float(np.mean([v for r in records for v in r["decision_latency_ms"]["raw"]])) if records else None,
                "p95": float(np.percentile([v for r in records for v in r["decision_latency_ms"]["raw"]], 95)) if records else None,
                "p99": float(np.percentile([v for r in records for v in r["decision_latency_ms"]["raw"]], 99)) if records else None,
            },
            "task_denominator": sum(len(record.get("tasks", [])) for record in records),
            "physical_on_time": sum(record["counts"].get("completed", 0) for record in records),
            "host_on_time": sum(
                1 for record in records
                for task in record.get("completion_records", {}).values()
                if task.get("host_confirmation_before_deadline") is True
            ),
            "pending_tasks": sum(
                1 for record in records
                for task in record.get("tasks", [])
                if (task.get("status") if isinstance(task, dict) else task)
                in ("pending", "censored_window", "unknown")
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arrival-model", type=Path, required=True)
    parser.add_argument("--arrival-model-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; no silent fallback")
    if not 1 <= args.threads <= 4:
        raise SystemExit("threads must be 1..4")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing non-empty output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    config = arrival_config()
    model_sha = sha256_file(args.arrival_model)
    if model_sha != args.arrival_model_sha256.lower():
        raise SystemExit(f"arrival model SHA-256 mismatch: expected {args.arrival_model_sha256}, got {model_sha}")
    scorer = FrozenArrivalConsequenceScorer.from_checkpoint(args.arrival_model, device=args.device, expected_sha256=model_sha)
    tapes = {
        "train": list(weak_communication_tape("train", count=32, base_seed=93001, level="composite")),
        "validation": list(weak_communication_tape("validation", count=16, base_seed=93001, level="composite")),
    }
    dump(args.out / "tapes.json", {key: [scenario_to_dict(item) for item in values] for key, values in tapes.items()})
    dump(args.out / "run-identity.json", {
        "run_id": "m10-arrival-gppo-fusion-pilot-20260914-seed1101",
        "protocol": "world-gppo-9.11-arrival/0.1.0",
        "seed": SEED,
        "device": args.device,
        "threads": args.threads,
        "budget": {"environment_steps_per_learning_group": ENV_STEPS, "max_optimizer_updates": MAX_OPTIMIZER_UPDATES, "group_wall_seconds": GROUP_WALL_SECONDS},
        "model": {"path": str(args.arrival_model.resolve()), "sha256": model_sha, "format": scorer.checkpoint_format, "feature_contract": asdict(scorer.contract)},
        "source_sha256": {"runner": sha256_file(Path(__file__)), "fusion": sha256_file(ROOT / "gppo_world" / "arrival_gppo_fusion.py"), "training": sha256_file(ROOT / "gppo_world" / "m10_training.py"), "protocol": sha256_file(ROOT / "configs" / "world-gppo-9.11-arrival-v0.1.0.json")},
        "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(), "cuda": torch.cuda.get_device_name(0) if args.device == "cuda" else None, "cuda_version": torch.version.cuda if args.device == "cuda" else None},
        "tape_counts": {key: len(value) for key, value in tapes.items()},
    })
    results: dict[str, Any] = {"status": "running", "run_id": "m10-arrival-gppo-fusion-pilot-20260914-seed1101", "groups": {}, "training_performed": True}
    started = time.perf_counter()
    results["groups"]["traditional"] = {"training": None, "evaluation": evaluate_group(None, tapes["validation"], config, device=args.device, scorer=None, variant="legal-public-rule")}  # type: ignore[arg-type]
    for name, history in (("GPPO", False), ("GPPO-History", True)):
        group_start = time.perf_counter()
        policy, metadata = train_policy(
            variant=name, encoder="graph", type_count=5, history=history, fusion="base", model=None,
            seed=SEED, steps=ENV_STEPS, env_config=config, ppo_config=PPOConfig(rollout_steps=256, update_epochs=4),
            device=args.device, scenarios=tapes["train"],
        )
        if time.perf_counter() - group_start > GROUP_WALL_SECONDS:
            raise RuntimeError(f"{name} exceeded wall-clock bound")
        metadata["protocol"] = "world-gppo-9.11-arrival/0.1.0"
        policy_path = args.out / "checkpoints" / name / "policy.pt"
        save_policy(policy_path, policy, metadata)
        results["groups"][name] = {"training": metadata, "evaluation": evaluate_group(policy, tapes["validation"], config, device=args.device, scorer=None, variant=name), "checkpoint": str(policy_path)}
    fusion_start = time.perf_counter()
    fusion_policy, fusion_metadata = train_candidate_policy(
        env_config=config, scenarios=tapes["train"], scorer=scorer, seed=SEED, device=args.device,
        steps=ENV_STEPS, max_updates=MAX_OPTIMIZER_UPDATES, max_wall_seconds=GROUP_WALL_SECONDS,
    )
    if time.perf_counter() - fusion_start > GROUP_WALL_SECONDS:
        raise RuntimeError("fusion exceeded wall-clock bound")
    fusion_path = args.out / "checkpoints" / "GPPO-History-CandidateArrival" / "policy.pt"
    fusion_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": fusion_policy.state_dict(), "metadata": fusion_metadata, "optimizer_state_dict": fusion_policy.optimizer.state_dict(), "recovery_state": {"environment_steps": fusion_metadata["steps"], "optimizer_updates": fusion_metadata["optimizer_updates"], "seed": SEED}}, fusion_path)
    results["groups"]["GPPO-History-CandidateArrival"] = {"training": fusion_metadata, "evaluation": evaluate_group(fusion_policy, tapes["validation"], config, device=args.device, scorer=scorer, variant="GPPO-History-CandidateArrival"), "checkpoint": str(fusion_path)}
    results["status"] = "complete"
    results["elapsed_seconds"] = time.perf_counter() - started
    dump(args.out / "pilot-results.json", results)
    dump(args.out / "run-status.json", {"status": "complete", "run_id": results["run_id"], "completed_at": time.time(), "groups": list(results["groups"]), "training_steps": {key: value.get("training", {}).get("steps", 0) if isinstance(value, dict) else 0 for key, value in results["groups"].items()}})
    print(json.dumps({"status": "complete", "output": str(args.out), "groups": list(results["groups"]), "elapsed_seconds": results["elapsed_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
