"""M-10-R3 prediction, trigger-cost and learning-sufficiency experiments.

This runner keeps R2 artifacts immutable.  It evaluates the frozen R2 World
checkpoint on a newly frozen final tape, compares trigger schedules with the
same policy weights, and runs a budget ladder before any larger matrix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gppo_world.m10_environment import M10Config, M10Environment, scenario_tape, scenario_to_dict
from gppo_world.m10_training import (
    M10ActorCritic, M10WorldModel, PPOConfig, _act, _policy_input_bundle,
    _public_vector, _trigger_decision, collect_world_dataset,
    evaluate_policy, evaluate_world_model_rows, save_policy, train_policy,
)


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def load_payload(path: Path, device: str) -> dict[str, Any]:
    return torch.load(path, map_location=torch.device(device), weights_only=False)


def load_world(path: Path, obs_dim: int, action_count: int, device: str) -> M10WorldModel:
    payload = load_payload(path, device)
    state = payload.get("state_dict", payload)
    model = M10WorldModel(obs_dim, action_count).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_policy(path: Path, env_config: M10Config, device: str) -> tuple[M10ActorCritic, dict[str, Any]]:
    payload = load_payload(path, device)
    metadata = dict(payload.get("metadata", {}))
    policy = M10ActorCritic(
        uav_count=env_config.uav_count, task_capacity=env_config.task_capacity,
        action_count=env_config.action_count, encoder=str(metadata.get("encoder", "graph")),
        type_count=int(metadata.get("type_count", 5)), history=bool(metadata.get("history", False)),
        context_dim=8 if str(metadata.get("fusion", "base")) != "base" else 0,
        region_count=env_config.region_count, target_count=env_config.target_count,
        event_capacity=env_config.event_capacity, relation_width=env_config.relation_width,
    ).to(device)
    policy.load_state_dict(payload["state_dict"])
    policy.eval()
    return policy, metadata


def _latency_summary(samples: list[float]) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "samples": len(values),
        "mean_ms": float(values.mean()) if len(values) else float("nan"),
        "p95_ms": float(np.percentile(values, 95)) if len(values) else float("nan"),
        "p99_ms": float(np.percentile(values, 99)) if len(values) else float("nan"),
        "raw_ms": values.tolist(),
    }


@torch.no_grad()
def evaluate_trigger_mode(policy: M10ActorCritic, world: M10WorldModel, *, mode: str,
                          env_config: M10Config, scenarios: list[Any], device: str,
                          threshold: float, max_replan_interval: int) -> dict[str, Any]:
    """Evaluate periodic, rules and model-risk schedules on identical tapes.

    ``rules`` does not invoke the world model on continuation intervals.  It
    computes one fresh context only when a rule actually requests a decision.
    ``model-risk`` computes risk and context once per current snapshot and
    reuses that result if a semantic, safety, wait or risk trigger wins.
    """
    if mode not in ("periodic", "rules", "model-risk"):
        raise ValueError(mode)
    device_obj = torch.device(device)
    policy = policy.to(device_obj).eval()
    world = world.to(device_obj).eval()
    records: list[dict[str, Any]] = []
    for scenario in scenarios:
        env = M10Environment(env_config, scenario)
        obs = env.reset()
        hidden = None
        last_action: int | None = None
        since = max_replan_interval
        done = False
        steps = 0
        total_reward = 0.0
        actor_calls = 0
        continuation = 0
        world_calls = 0
        replan_reasons: dict[str, int] = {}
        trigger_condition_counts: dict[str, int] = {}
        latency: list[float] = []
        info: dict[str, Any] = {"counts": {"completed": 0, "expired": 0, "rejected": 0}, "energy": {}}
        while not done and steps < int(env_config.horizon / env_config.decision_interval) + 2:
            start = time.perf_counter()
            if mode == "periodic":
                _, _, risk, full_vector = _policy_input_bundle(env, obs, world, fusion="world", device=device_obj, trigger_threshold=threshold)
                world_calls += 1
                should_replan, reason, conditions = _trigger_decision(
                    obs, fusion="base", risk_active=False, last_action=last_action,
                    steps_since_replan=since, max_replan_interval=max_replan_interval,
                )
                decision_vector = full_vector
            elif mode == "rules":
                public_vector = np.concatenate((np.asarray(obs["flat"], dtype=np.float32), np.zeros(world.context_dim, dtype=np.float32)))
                should_replan, reason, conditions = _trigger_decision(
                    obs, fusion="triggered", risk_active=False, last_action=last_action,
                    steps_since_replan=since, max_replan_interval=max_replan_interval,
                )
                risk = 0.0
                decision_vector = public_vector
                if should_replan:
                    _, _, risk, decision_vector = _policy_input_bundle(
                        env, obs, world, fusion="world", device=device_obj, trigger_threshold=threshold,
                    )
                    world_calls += 1
            else:
                vector, risk_active, risk, full_vector = _policy_input_bundle(
                    env, obs, world, fusion="triggered", device=device_obj, trigger_threshold=threshold,
                )
                world_calls += 1
                should_replan, reason, conditions = _trigger_decision(
                    obs, fusion="triggered", risk_active=risk_active, last_action=last_action,
                    steps_since_replan=since, max_replan_interval=max_replan_interval,
                )
                decision_vector = full_vector if should_replan else vector
            for condition, active in conditions.items():
                if active:
                    trigger_condition_counts[condition] = trigger_condition_counts.get(condition, 0) + 1
            if should_replan:
                action, _, _, hidden = _act(policy, decision_vector, obs["mask"], hidden, device_obj, deterministic=True)
                actor_calls += 1
                replan_reasons[reason] = replan_reasons.get(reason, 0) + 1
            else:
                # The old action is the controller's continuation command.  A
                # zero context is explicit here: rules never cache context
                # across a changed observation/version.
                continuation_vector = np.concatenate((np.asarray(obs["flat"], dtype=np.float32), np.zeros(world.context_dim, dtype=np.float32)))
                _, hidden = policy.value_only(torch.tensor(continuation_vector, dtype=torch.float32, device=device_obj)[None, :], hidden)
                if last_action is None:
                    raise RuntimeError("continuation occurred before an initial action")
                action = int(last_action)
                continuation += 1
            last_action = action
            since = 0 if should_replan else since + 1
            obs, reward, done, info = env.step(action, submit_command=should_replan)
            total_reward += float(reward)
            steps += 1
            if device_obj.type == "cuda":
                torch.cuda.synchronize(device_obj)
            latency.append((time.perf_counter() - start) * 1000.0)
        records.append({
            "tape_id": scenario.tape_id, "scenario_seed": scenario.seed, "return": total_reward,
            "steps": steps, "completed": info["counts"]["completed"], "expired": info["counts"]["expired"],
            "rejected": info["counts"]["rejected"], "energy_remaining": float(sum(info["energy"].values())),
            "actor_calls": actor_calls, "continuation_steps": continuation,
            "world_model_calls": world_calls, "replan_reasons": replan_reasons,
            "trigger_condition_counts": trigger_condition_counts,
            "latency_ms": latency,
        })
    numeric = ("return", "completed", "expired", "rejected", "energy_remaining", "actor_calls", "continuation_steps", "world_model_calls")
    summary = {key: {"mean": float(np.mean([row[key] for row in records])), "std": float(np.std([row[key] for row in records]))} for key in numeric}
    summary["latency"] = _latency_summary([sample for row in records for sample in row["latency_ms"]])
    return {"mode": mode, "episodes": records, "summary": summary, "threshold": threshold, "max_replan_interval": max_replan_interval}


def prediction_audit(args: argparse.Namespace, env_config: M10Config, out: Path) -> dict[str, Any]:
    tapes = {
        "train": list(scenario_tape("train", count=32, base_seed=7001)),
        "validation": list(scenario_tape("validation", count=16, base_seed=7001)),
        "historical_test": list(scenario_tape("test", count=16, base_seed=7001)),
        "ood": list(scenario_tape("ood", count=16, base_seed=7001, name="energy_insufficient")),
        "final_test": list(scenario_tape("test", count=16, base_seed=91001)),
    }
    dump(out / "tapes.json", {key: [scenario_to_dict(item) for item in value] for key, value in tapes.items()})
    rows_by_split = {
        key: collect_world_dataset(episodes=len(value), seed=91001 + index * 101, config=env_config, scenarios=value, split=("test" if key == "historical_test" else "test" if key == "final_test" else key))
        for index, (key, value) in enumerate(tapes.items())
    }
    rows_train = rows_by_split["train"]
    rows_validation = rows_by_split["validation"]
    rows_historical = rows_by_split["historical_test"]
    rows_ood = rows_by_split["ood"]
    rows_final = rows_by_split["final_test"]
    world = load_world(Path(args.world_checkpoint), len(rows_train[0]["obs"]), env_config.action_count, args.device)
    metrics = {
        "train": evaluate_world_model_rows(world, rows_train, device=args.device, baseline_rows=rows_train),
        "validation": evaluate_world_model_rows(world, rows_validation, device=args.device, baseline_rows=rows_train),
        "historical_regression_test": evaluate_world_model_rows(world, rows_historical, device=args.device, baseline_rows=rows_train),
        "ood": evaluate_world_model_rows(world, rows_ood, device=args.device, baseline_rows=rows_train),
        "final_test": evaluate_world_model_rows(world, rows_final, device=args.device, baseline_rows=rows_train),
    }
    metadata = {
        "protocol": "frozen_R3_prediction_audit",
        "world_checkpoint": str(args.world_checkpoint),
        "label_horizon": "one environment decision interval, after env.step",
        "event_labels": {"damage": "new ServiceClock damage event", "disconnect": "new ServiceClock disconnect event", "reconnect": "new ServiceClock reconnect event", "reserved_event_3": "always zero in current simulator"},
        "label_limit": "event labels indicate simulator events during the next interval; they are not a validated label for future replanning necessity",
        "split_policy": "train/validation used for baseline description; historical test is regression only; final_test tape was frozen before evaluation and not used for selection",
        "tape_counts": {key: len(value) for key, value in tapes.items()},
    }
    dump(out / "prediction-audit.json", {"metadata": metadata, "metrics": metrics})
    return {"tapes": tapes, "world": world, "metrics": metrics}


def trigger_audit(args: argparse.Namespace, env_config: M10Config, out: Path) -> None:
    tape = list(scenario_tape("test", count=16, base_seed=91001))
    policy, policy_meta = load_policy(Path(args.policy_checkpoint), env_config, args.device)
    world = load_world(Path(args.world_checkpoint), policy.base_obs_dim, env_config.action_count, args.device)
    results = [evaluate_trigger_mode(policy, world, mode=mode, env_config=env_config, scenarios=tape, device=args.device, threshold=args.threshold, max_replan_interval=args.max_replan_interval) for mode in ("periodic", "rules", "model-risk")]
    dump(out / "trigger-comparison.json", {
        "protocol": "same_world_policy_checkpoint_same_final_test_tape",
        "policy_checkpoint": str(args.policy_checkpoint), "policy_metadata": policy_meta,
        "world_checkpoint": str(args.world_checkpoint), "threshold": args.threshold,
        "modes": results,
        "interpretation": "rules and model-risk differ in trigger source; model-risk inference is computed once per snapshot and reused for context/risk",
    })


def ladder(args: argparse.Namespace, env_config: M10Config, out: Path) -> None:
    train_tape = list(scenario_tape("train", count=32, base_seed=7001))
    validation_tape = {item.seed: item for item in scenario_tape("validation", count=16, base_seed=7001)}
    world_payload = load_payload(Path(args.world_checkpoint), args.device)
    world_state = world_payload.get("state_dict", world_payload)
    world_obs_dim = int(world_state["backbone.0.weight"].shape[1] - env_config.action_count)
    world = load_world(Path(args.world_checkpoint), world_obs_dim, env_config.action_count, args.device)
    specs = [
        ("graph-5-base", "graph", 5, False, "base"),
        ("graph-history-5-base", "graph", 5, True, "base"),
        ("graph-5-world", "graph", 5, False, "world"),
    ]
    budgets = [int(item) for item in args.budgets.split(",")]
    results = []
    for name, encoder, type_count, history, fusion in specs:
        for budget in budgets:
            model = world if fusion == "world" else None
            policy, metadata = train_policy(
                variant=name, encoder=encoder, type_count=type_count, history=history,
                fusion=fusion, model=model, seed=args.seed, steps=budget,
                env_config=env_config, ppo_config=PPOConfig(rollout_steps=256, update_epochs=4),
                device=args.device, trigger_threshold=args.threshold, scenarios=train_tape,
            )
            evaluation = evaluate_policy(policy, model=model, fusion=fusion, env_config=env_config,
                                         seeds=[item.seed for item in validation_tape.values()], device=args.device,
                                         trigger_threshold=args.threshold, scenarios=validation_tape)
            checkpoint = out / "checkpoints" / name / f"seed-{args.seed}-steps-{budget}.pt"
            save_policy(checkpoint, policy, metadata)
            results.append({"variant": name, "budget_steps": budget, "metadata": metadata, "validation": evaluation, "checkpoint": str(checkpoint)})
            dump(out / "records" / name / f"seed-{args.seed}-steps-{budget}.json", results[-1])
    dump(out / "learning-budget-ladder.json", {"budgets": budgets, "seed": args.seed, "results": results, "world_pretraining_cost": "reused frozen R2 checkpoint; no new world-model training in ladder"})


def formal_matrix(args: argparse.Namespace, env_config: M10Config, out: Path) -> None:
    train_tape = list(scenario_tape("train", count=32, base_seed=7001))
    final_tape = {item.seed: item for item in scenario_tape("test", count=16, base_seed=91001)}
    world_payload = load_payload(Path(args.world_checkpoint), args.device)
    world_state = world_payload.get("state_dict", world_payload)
    world_obs_dim = int(world_state["backbone.0.weight"].shape[1] - env_config.action_count)
    world = load_world(Path(args.world_checkpoint), world_obs_dim, env_config.action_count, args.device)
    specs = [
        ("mlp-2-base", "mlp", 2, False, "base"),
        ("graph-2-base", "graph", 2, False, "base"),
        ("graph-5-base", "graph", 5, False, "base"),
        ("graph-history-5-base", "graph", 5, True, "base"),
        ("graph-5-world", "graph", 5, False, "world"),
    ]
    seeds = [int(item) for item in args.seeds.split(",")]
    results = []
    for name, encoder, type_count, history, fusion in specs:
        for seed in seeds:
            model = world if fusion == "world" else None
            policy, metadata = train_policy(
                variant=name, encoder=encoder, type_count=type_count, history=history,
                fusion=fusion, model=model, seed=seed, steps=args.formal_steps,
                env_config=env_config, ppo_config=PPOConfig(rollout_steps=256, update_epochs=4),
                device=args.device, trigger_threshold=args.threshold, scenarios=train_tape,
            )
            evaluation = evaluate_policy(
                policy, model=model, fusion=fusion, env_config=env_config,
                seeds=[item.seed for item in final_tape.values()], device=args.device,
                trigger_threshold=args.threshold, scenarios=final_tape,
            )
            checkpoint = out / "checkpoints" / name / f"seed-{seed}.pt"
            save_policy(checkpoint, policy, metadata)
            record = {"variant": name, "seed": seed, "metadata": metadata, "evaluation": evaluation, "checkpoint": str(checkpoint)}
            results.append(record)
            dump(out / "records" / name / f"seed-{seed}.json", record)
    dump(out / "formal-matrix.json", {
        "protocol": "R3_frozen_8192_env_steps_same_train_and_final_test_tape",
        "formal_steps": args.formal_steps, "seeds": seeds,
        "variants": [item[0] for item in specs], "world_pretraining_cost": "reused frozen R2 checkpoint and reported separately",
        "results": results,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prediction", "trigger", "ladder", "formal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-checkpoint", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--max-replan-interval", type=int, default=3)
    parser.add_argument("--budgets", default="2048,4096,8192")
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--seeds", default="1101,2203,3307")
    parser.add_argument("--formal-steps", type=int, default=8192)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    env_config = M10Config()
    if args.mode == "prediction":
        prediction_audit(args, env_config, out)
    elif args.mode == "trigger":
        if args.policy_checkpoint is None:
            raise SystemExit("--policy-checkpoint is required for trigger mode")
        trigger_audit(args, env_config, out)
    elif args.mode == "ladder":
        ladder(args, env_config, out)
    else:
        formal_matrix(args, env_config, out)
    dump(out / "run-complete.json", {"mode": args.mode, "device": args.device, "completed_at": time.time()})


if __name__ == "__main__":
    main()
