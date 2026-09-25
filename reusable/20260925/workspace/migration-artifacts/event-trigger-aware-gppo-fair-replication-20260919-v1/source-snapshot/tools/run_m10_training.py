"""Run the authorized M-10 pilot or frozen-budget comparison matrix.

The script never downloads or starts a training process implicitly.  It writes
all checkpoints, configs, logs and evaluation summaries below the requested
output directory.
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

from gppo_world.m10_environment import M10Config, scenario_tape, scenario_to_dict
from gppo_world.m10_training import (
    M10ActorCritic,
    M10WorldModel,
    PPOConfig,
    calibrate_trigger_threshold,
    collect_world_dataset,
    evaluate_policy,
    save_policy,
    train_policy,
    train_world_model,
)


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def variant_specs(mode: str) -> list[dict[str, Any]]:
    if mode == "pilot":
        return [
            {"name": "mlp-2-base", "encoder": "mlp", "type_count": 2, "history": False, "fusion": "base"},
            {"name": "graph-5-base", "encoder": "graph", "type_count": 5, "history": False, "fusion": "base"},
            {"name": "graph-history-5-base", "encoder": "graph", "type_count": 5, "history": True, "fusion": "base"},
            {"name": "graph-5-world", "encoder": "graph", "type_count": 5, "history": False, "fusion": "world"},
            {"name": "graph-5-triggered-replan", "encoder": "graph", "type_count": 5, "history": False, "fusion": "triggered"},
        ]
    return [
        {"name": "mlp-2-base", "encoder": "mlp", "type_count": 2, "history": False, "fusion": "base"},
        {"name": "graph-2-base", "encoder": "graph", "type_count": 2, "history": False, "fusion": "base"},
        {"name": "graph-5-base", "encoder": "graph", "type_count": 5, "history": False, "fusion": "base"},
        {"name": "graph-history-5-base", "encoder": "graph", "type_count": 5, "history": True, "fusion": "base"},
        {"name": "graph-5-world", "encoder": "graph", "type_count": 5, "history": False, "fusion": "world"},
        {"name": "graph-5-triggered-replan", "encoder": "graph", "type_count": 5, "history": False, "fusion": "triggered"},
    ]


def benchmark(policy: M10ActorCritic, world: M10WorldModel | None, env_config: M10Config,
              fusion: str, device_name: str, trigger_threshold: float, max_replan_interval: int = 3) -> dict[str, float]:
    from gppo_world.m10_environment import M10Environment, scenario_tape
    from gppo_world.m10_training import _act, _policy_input_bundle, _trigger_decision

    device = torch.device(device_name)
    policy = policy.to(device).eval()
    if world is not None:
        world = world.to(device).eval()
    env = M10Environment(env_config, scenario_tape("test", count=1, base_seed=9901)[0])
    obs = env.reset()
    vector, _, _, _ = _policy_input_bundle(env, obs, world, fusion=fusion, device=device, trigger_threshold=trigger_threshold)
    for _ in range(20):
        _act(policy, vector, obs["mask"], None, device, deterministic=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    policy_times, chain_times, env_times = [], [], []
    last_action = None
    hidden = None
    steps_since_replan = max_replan_interval
    actor_calls = 0
    continuation_steps = 0
    world_model_calls = 0
    replan_reasons: dict[str, int] = {}
    for _ in range(100):
        start = time.perf_counter()
        obs = env._observation()
        vector, active, risk, full_vector = _policy_input_bundle(
            env, obs, world, fusion=fusion, device=device, trigger_threshold=trigger_threshold,
        )
        world_model_calls += int(world is not None and fusion != "base")
        should_replan, reason, _ = _trigger_decision(
            obs, fusion=fusion, risk_active=active, last_action=last_action,
            steps_since_replan=steps_since_replan, max_replan_interval=max_replan_interval,
        )
        if should_replan:
            actor_start = time.perf_counter()
            action_vector = full_vector if fusion == "triggered" else vector
            action, _, _, hidden = _act(policy, action_vector, obs["mask"], hidden, device, deterministic=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            policy_times.append((time.perf_counter() - actor_start) * 1000.0)
            actor_calls += 1
            replan_reasons[reason] = replan_reasons.get(reason, 0) + 1
        else:
            _, hidden = policy.value_only(torch.tensor(vector, dtype=torch.float32, device=device)[None, :], hidden)
            action = int(last_action)
            continuation_steps += 1
        last_action = action
        steps_since_replan = 0 if should_replan else steps_since_replan + 1
        chain_policy_done = time.perf_counter()
        obs, _, done, _ = env.step(action, submit_command=should_replan)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        chain_times.append((time.perf_counter() - start) * 1000.0)
        env_times.append((time.perf_counter() - chain_policy_done) * 1000.0)
        if done:
            env = M10Environment(env_config, scenario_tape("test", count=1, base_seed=9901)[0])
            obs = env.reset()
            last_action = None
            hidden = None
            steps_since_replan = max_replan_interval
    return {
        "device": device_name,
        "policy_mean_ms": float(np.mean(policy_times)),
        "policy_p95_ms": float(np.percentile(policy_times, 95)),
        "policy_p99_ms": float(np.percentile(policy_times, 99)),
        "decision_chain_mean_ms": float(np.mean(chain_times)),
        "decision_chain_p95_ms": float(np.percentile(chain_times, 95)),
        "decision_chain_p99_ms": float(np.percentile(chain_times, 99)),
        "env_step_mean_ms": float(np.mean(env_times)),
        "env_step_p95_ms": float(np.percentile(env_times, 95)),
        "actor_calls": actor_calls,
        "continuation_steps": continuation_steps,
        "world_model_calls": world_model_calls,
        "replan_reasons": replan_reasons,
        "samples": len(chain_times),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "matrix"), default="pilot")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seeds", default=None, help="comma-separated policy seeds")
    parser.add_argument("--world-episodes", type=int, default=32)
    parser.add_argument("--world-epochs", type=int, default=25)
    parser.add_argument("--trigger-threshold", type=float, default=None)
    parser.add_argument("--trigger-max-interval", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env_config = M10Config()
    ppo_config = PPOConfig(rollout_steps=128 if args.mode == "pilot" else 256)
    steps = args.steps or (256 if args.mode == "pilot" else 2048)
    seeds = [int(item) for item in (args.seeds.split(",") if args.seeds else ([1101] if args.mode == "pilot" else [1101, 2203, 3307]))]
    specs = variant_specs(args.mode)
    train_tape = scenario_tape("train", count=max(args.world_episodes, 32), base_seed=7001)
    validation_tape = scenario_tape("validation", count=16, base_seed=7001)
    test_tape = scenario_tape("test", count=16, base_seed=7001)
    ood_tape = scenario_tape("ood", count=16, base_seed=7001)
    for split, tape in (("train", train_tape), ("validation", validation_tape), ("test", test_tape), ("ood", ood_tape)):
        dump(args.output / "tapes" / f"{split}.json", [scenario_to_dict(scenario) for scenario in tape])
    train_rows = collect_world_dataset(episodes=args.world_episodes, seed=7001, config=env_config, scenarios=train_tape[:args.world_episodes], split="train")
    validation_rows = collect_world_dataset(episodes=len(validation_tape), seed=7001, config=env_config, scenarios=validation_tape, split="validation")
    test_rows = collect_world_dataset(episodes=len(test_tape), seed=7001, config=env_config, scenarios=test_tape, split="test")
    ood_rows = collect_world_dataset(episodes=len(ood_tape), seed=7001, config=env_config, scenarios=ood_tape, split="ood")
    world_rows = train_rows + validation_rows + test_rows
    dump(args.output / "protocol.json", {
        "format": "m10-training-protocol/0.1.0", "mode": args.mode,
        "environment": {"config": env_config.__dict__, "task_semantics": ["arrival", "service_progress", "deadline", "damage", "energy", "communication"], "fixed_regression_seed_0": True},
        "shared_action_count": env_config.action_count, "shared_visible_observation": True, "node_types": ["UAV", "Region", "Target", "Task", "Event"], "relation_encoding": "explicit_uav_task_edges_with_distance_visibility_mask_region",
        "variants": specs, "policy_seeds": seeds, "steps_per_seed": steps,
        "ppo": ppo_config.__dict__, "world_model": {"episodes": args.world_episodes, "epochs": args.world_epochs, "split": "frozen_train_validation_test_tapes", "ood_episodes": len(ood_tape)},
        "trigger": {"threshold_source": "validation_f1" if args.trigger_threshold is None else "cli_override", "requested_threshold": args.trigger_threshold, "max_replan_interval": args.trigger_max_interval, "forced_events": ["task_arrival", "confirmed_fault", "link_recovery", "completion_or_invalidation", "safety"], "optional_conditions": ["risk_above_threshold", "max_wait"], "priority": ["initial", "safety", "confirmed_fault", "link_recovery", "task_arrival", "completion_or_invalidation", "risk", "max_wait"], "continuation": "renew_existing_acked_lease_without_resubmission", "no_trigger_policy": "continue_existing_task_or_hold_control"},
        "charging_return_scope": "not included pending explicit meeting acceptance; no silent exclusion claim",
    })
    world, world_meta = train_world_model(world_rows, seed=7001, action_count=env_config.action_count, epochs=args.world_epochs, device=args.device, ood_rows=ood_rows)
    calibration = calibrate_trigger_threshold(world, validation_rows, device=args.device)
    trigger_threshold = float(args.trigger_threshold if args.trigger_threshold is not None else calibration["selected_threshold"])
    world_meta["trigger_calibration"] = calibration
    world_meta["threshold_used_for_new_matrix"] = trigger_threshold
    torch.save({"state_dict": world.state_dict(), "metadata": world_meta,
                "optimizer_state_dict": world.optimizer.state_dict() if hasattr(world, "optimizer") else None,
                "recovery_state": {"epochs": args.world_epochs, "seed": 7001}}, args.output / "world-model.pt")
    dump(args.output / "world-model.json", world_meta)
    all_results = []
    for spec in specs:
        for seed in seeds:
            model_for_variant = world if spec["fusion"] != "base" else None
            policy, metadata = train_policy(
                variant=spec["name"], encoder=spec["encoder"], type_count=spec["type_count"], history=spec["history"],
                fusion=spec["fusion"], model=model_for_variant, seed=seed, steps=steps, env_config=env_config,
                ppo_config=ppo_config, device=args.device, trigger_threshold=trigger_threshold,
                scenarios=train_tape, max_replan_interval=args.trigger_max_interval,
            )
            eval_result = evaluate_policy(policy, model=model_for_variant, fusion=spec["fusion"], env_config=env_config, seeds=[4401, 4402, 4403, 4404, 4405], device=args.device, trigger_threshold=trigger_threshold, scenarios={4401 + i: test_tape[i] for i in range(5)}, max_replan_interval=args.trigger_max_interval)
            safe_name = spec["name"].replace("/", "_")
            checkpoint = args.output / "checkpoints" / safe_name / f"seed-{seed}.pt"
            save_policy(checkpoint, policy, metadata)
            record = {"spec": spec, "seed": seed, "training": metadata, "evaluation": eval_result}
            dump(args.output / "records" / safe_name / f"seed-{seed}.json", record)
            all_results.append(record)
    benchmark_results = []
    for representative_name in ("graph-5-world", "graph-5-triggered-replan"):
        representative = next(item for item in all_results if item["spec"]["name"] == representative_name)
        rep_spec = representative["spec"]
        rep_model = world if rep_spec["fusion"] != "base" else None
        rep_policy, _ = train_policy(
            variant=rep_spec["name"], encoder=rep_spec["encoder"], type_count=rep_spec["type_count"], history=rep_spec["history"],
            fusion=rep_spec["fusion"], model=rep_model, seed=seeds[0], steps=max(64, min(steps, 256)), env_config=env_config,
            ppo_config=ppo_config, device=args.device, trigger_threshold=trigger_threshold,
            scenarios=train_tape, max_replan_interval=args.trigger_max_interval,
        )
        for device_name in ("cpu", args.device) if args.device != "cpu" else ("cpu",):
            result = benchmark(rep_policy, rep_model, env_config, rep_spec["fusion"], device_name, trigger_threshold, args.trigger_max_interval)
            result["variant"] = representative_name
            benchmark_results.append(result)
    dump(args.output / "matrix-results.json", {"mode": args.mode, "records": all_results, "latency": benchmark_results, "world_model": world_meta, "world_model_test_metrics": world_meta.get("test_metrics"), "world_model_baseline_metrics": world_meta.get("test_baseline_metrics"), "trigger_calibration": calibration, "tape_counts": {"train": len(train_tape), "validation": len(validation_tape), "test": len(test_tape), "ood": len(ood_tape)}})
    dump(args.output / "run-complete.json", {"mode": args.mode, "variants": len(specs), "seeds": seeds, "steps_per_seed": steps, "world_model_rows": len(world_rows), "world_model_ood_rows": len(ood_rows), "trigger_threshold": trigger_threshold, "status": "completed"})


if __name__ == "__main__":
    main()
