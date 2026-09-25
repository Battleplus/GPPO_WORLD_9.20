"""Reproduce the frozen M-10 meeting candidates without training.

This is intentionally a small, auditable execution trace rather than another
benchmark or a checkpoint-selection loop.  It runs the public observation,
optional frozen world-model context, deterministic policy, TaskDecisionBridge
and ServiceClock on one serialized demo tape.  Credentials and server details
are outside this file by design.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gppo_world.m10_environment import M10Config, M10Environment, default_scenario, scenario_to_dict
from gppo_world.m10_training import _act, _policy_input_bundle
from tools.run_m10_r3 import load_policy, load_world


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def decode_action(action: int, config: M10Config) -> dict[str, Any]:
    if action == config.action_count - 1:
        return {"kind": "noop", "uav_id": None, "task_index": None}
    uav_index, task_index = divmod(action, config.task_capacity)
    return {"kind": "uav_task", "uav_id": f"uav-{uav_index}", "task_index": task_index}


def run_episode(policy: torch.nn.Module, world: torch.nn.Module | None, *, scenario: Any,
                config: M10Config, fusion: str, device: str) -> dict[str, Any]:
    env = M10Environment(config, scenario=scenario)
    obs = env.reset()
    hidden = None
    done = False
    total_reward = 0.0
    steps = 0
    actor_calls = 0
    world_calls = 0
    trajectory: list[dict[str, Any]] = []
    device_obj = torch.device(device)
    while not done and steps < int(config.horizon / config.decision_interval) + 2:
        before = dict(obs)
        if fusion == "world":
            vector, context_active, risk, full_vector = _policy_input_bundle(
                env, obs, world, fusion="world", device=device_obj, trigger_threshold=0.5,
            )
            world_calls += 1
            context_norm = float(np.linalg.norm(full_vector[len(obs["flat"]):]))
        else:
            vector = np.asarray(obs["flat"], dtype=np.float32)
            context_active, risk, context_norm = False, 0.0, 0.0
        action, log_prob, value, hidden = _act(
            policy, vector, obs["mask"], hidden, device_obj, deterministic=True,
        )
        actor_calls += 1
        next_obs, reward, done, info = env.step(action, submit_command=True)
        total_reward += float(reward)
        decoded = decode_action(action, config)
        trajectory.append({
            "step": steps,
            "time_before": float(before["time"]),
            "time_after": float(info["time"]),
            "policy_version_before": int(before["version"]),
            "policy_version_after": int(info["policy_version"]),
            "visible_mask_count": int(np.asarray(before["mask"]).sum()),
            "action": int(action),
            "decoded_action": decoded,
            "actor_called": True,
            "world_model_called": fusion == "world",
            "world_risk": float(risk),
            "world_context_active": bool(context_active),
            "world_context_l2": context_norm,
            "log_prob": float(log_prob),
            "value": float(value),
            "command_submitted": bool(info["command_submitted"]),
            "feedback": info["feedback"],
            "lease_renewal": info["lease_renewal"],
            "new_events": info["new_events"],
            "trigger_flags": info["trigger_flags"],
            "reward": float(reward),
            "counts": info["counts"],
            "energy": info["energy"],
            "tasks": info["tasks"],
            "terminated": bool(info["terminated"]),
            "truncated": bool(info["truncated"]),
        })
        obs = next_obs
        steps += 1
        if done:
            break
    final_info = trajectory[-1] if trajectory else {}
    return {
        "scenario": scenario_to_dict(scenario),
        "return": total_reward,
        "steps": steps,
        "actor_calls": actor_calls,
        "world_model_calls": world_calls,
        "completion": final_info.get("counts", {}),
        "final_energy": final_info.get("energy", {}),
        "trajectory": trajectory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--world", type=Path)
    parser.add_argument("--fusion", choices=("base", "world"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1101)
    args = parser.parse_args()
    if args.fusion == "world" and args.world is None:
        parser.error("--world is required for world fusion")
    config = M10Config(seed=args.seed)
    device = torch.device(args.device)
    # The tape is frozen in the output before execution.  These are explicit
    # semantic paths, not a search over episodes or a post-hoc test set.
    scenario_specs = (
        ("normal", 1101),
        ("mixed", 1102),
        ("uav_damage", 1103),
        ("energy_insufficient", 1104),
        ("communication_interrupt", 1105),
    )
    scenarios = [default_scenario(name, seed=seed, split="validation") for name, seed in scenario_specs]
    policy, policy_metadata = load_policy(args.policy, config, str(device))
    world = load_world(args.world, len(config_for_probe(config)["flat"]), config.action_count, str(device)) if args.world else None
    records = [run_episode(policy, world, scenario=scenario, config=config, fusion=args.fusion, device=str(device)) for scenario in scenarios]
    summary = {
        "episodes": len(records),
        "environment_steps": sum(item["steps"] for item in records),
        "actor_calls": sum(item["actor_calls"] for item in records),
        "world_model_calls": sum(item["world_model_calls"] for item in records),
        "returns": [item["return"] for item in records],
        "completed": [item["completion"].get("completed", 0) for item in records],
        "expired": [item["completion"].get("expired", 0) for item in records],
        "rejected": [item["completion"].get("rejected", 0) for item in records],
    }
    dump(args.output, {
        "protocol": "m10-final-acceptance-reproduction-v1",
        "training_performed": False,
        "fusion": args.fusion,
        "device": str(device),
        "policy_checkpoint": str(args.policy),
        "world_checkpoint": str(args.world) if args.world else None,
        "policy_metadata": policy_metadata,
        "config": config.__dict__,
        "tape": [scenario_to_dict(scenario) for scenario in scenarios],
        "summary": summary,
        "episodes": records,
        "limitations": [
            "This is a deterministic simulator reproduction, not a production or flight validation.",
            "The candidate was selected for engineering demonstration from the frozen formal matrix; it is not an unbiased post-selection claim.",
        ],
    })


def config_for_probe(config: M10Config) -> dict[str, Any]:
    env = M10Environment(config, scenario=default_scenario("normal", seed=config.seed, split="validation"))
    return env.reset()


if __name__ == "__main__":
    main()
