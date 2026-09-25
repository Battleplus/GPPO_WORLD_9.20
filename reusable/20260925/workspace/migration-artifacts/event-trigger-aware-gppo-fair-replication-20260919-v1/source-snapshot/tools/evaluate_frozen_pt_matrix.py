"""Evaluate the repaired P/T checkpoints on the frozen validation tapes only."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.joint_training import _mask_safe, _vector_reward  # noqa: E402
from gppo_world.m10_environment import M10Environment  # noqa: E402
from tools.run_event_trigger_aware_gppo import (  # noqa: E402
    CONDITIONS,
    MAX_REPLAN_INTERVAL,
    JointGraphPreferencePolicy,
    ActionConditionedTemporalWorldModel,
    load_config,
    load_tapes,
    masked_normalized_preference,
    probe,
    trigger_decision,
    sha256,
)


PREFERENCE = (0.8, 0.2)
MATRIX = {
    "P_train_periodic": ("P_train", "periodic"),
    "P_train_triggered": ("P_train", "triggered"),
    "T_train_triggered": ("T_train", "triggered"),
}


def percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean_ms": None, "p95_ms": None, "p99_ms": None}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean_ms": float(arr.mean()), "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99))}


def load_model(checkpoint: Path, config, device: torch.device):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    policy = JointGraphPreferencePolicy(config, history=True).to(device)
    world = ActionConditionedTemporalWorldModel().to(device)
    policy.load_state_dict(payload["policy_state_dict"], strict=True)
    world.load_state_dict(payload["world_state_dict"], strict=True)
    policy.eval(); world.eval()
    for parameter in world.parameters():
        parameter.requires_grad_(False)
    return policy, world, payload


def evaluate_episode(policy, world, config, scenario, mode: str, seed: int, group: str,
                     model_name: str, preference: tuple[float, float]) -> dict:
    device = torch.device("cpu")
    env = M10Environment(config, scenario)
    obs = env.reset()
    previous_obs = None
    last_action = None
    policy_hidden = None
    world_hidden = None
    steps = 0
    actor_calls = 0
    continuation_steps = 0
    world_calls = 0
    candidate_calls = 0
    event_calls = 0
    command_submitted = 0
    command_accepted = 0
    noop_opportunities = 0
    noop_selected = 0
    illegal_new_actions = 0
    illegal_continuations = 0
    security_violations = 0
    communications = 0
    communication_bytes = 0
    communication_bytes_observed = True
    trigger_counts: Counter[str] = Counter()
    feedback_counts: Counter[str] = Counter()
    chain_ms: list[float] = []
    model_ms: list[float] = []
    previous_counts = {"completed": 0, "expired": 0}
    previous_energy = float(config.uav_count * config.initial_energy)
    total_reward = np.zeros(2, dtype=np.float64)
    info = {}
    steps_since_replan = MAX_REPLAN_INTERVAL

    while not bool(info.get("terminated", False) or info.get("truncated", False)) and steps < 128:
        chain_start = time.perf_counter()
        model_start = time.perf_counter()
        result = probe(policy, world, obs, policy_hidden, world_hidden,
                       masked_normalized_preference(preference, device=device), device)
        model_ms.append((time.perf_counter() - model_start) * 1000.0)
        world_calls += 1; candidate_calls += 1; event_calls += 1
        if mode == "periodic":
            actor = True
            reasons = ["periodic_policy"]
        else:
            actor, reasons, _ = trigger_decision(obs, previous_obs, last_action,
                                                 steps_since_replan)
        actor = bool(actor)
        if actor:
            action = int(torch.argmax(result["evaluation"]["distribution"].logits, dim=-1).item())
            actor_calls += 1
            steps_since_replan = 0
            for reason in reasons:
                trigger_counts[reason] += 1
            submit = True
        else:
            action = int(last_action)
            continuation_steps += 1
            steps_since_replan += 1
            submit = False
            if not (action in {int(value) for value in obs.get("continuation_actions", ())}):
                illegal_continuations += 1

        mask = _mask_safe(obs["mask"])
        has_non_noop = bool(mask[:-1].any())
        noop_opportunities += int(has_non_noop)
        noop_selected += int(action == config.action_count - 1 and has_non_noop)
        if submit and not (0 <= action < len(mask) and bool(mask[action])):
            illegal_new_actions += 1
        if submit:
            command_submitted += 1
        next_obs, _, done, info = env.step(action, submit_command=submit)
        info = dict(info)
        feedback = str(info.get("feedback", "unknown"))
        feedback_counts[feedback] += 1
        if feedback in ("accepted", "awaiting_ack", "ack_lost_after_accept"):
            command_accepted += 1
        command_submitted += 0
        communications += len(info.get("communication_delta", []))
        for message in info.get("communication_delta", []):
            if isinstance(message, dict) and "bytes" in message:
                communication_bytes += int(message["bytes"])
            else:
                communication_bytes_observed = False
        violations = info.get("security_violations", [])
        if isinstance(violations, (list, tuple)):
            security_violations += len(violations)
        else:
            security_violations += int(bool(info.get("security_violation", False)))
        reward_vector, _, previous_counts, previous_energy = _vector_reward(
            info, previous_counts, previous_energy, config
        )
        total_reward += reward_vector.astype(np.float64)
        chain_ms.append((time.perf_counter() - chain_start) * 1000.0)
        previous_obs, obs = obs, next_obs
        policy_hidden = result["next_policy_hidden"]
        world_hidden = result["by_action"][action]["hidden"]
        last_action = action
        steps += 1

    counts = info.get("counts", {})
    completed = int(counts.get("completed", 0))
    expired = int(counts.get("expired", 0))
    total_tasks = len(scenario.tasks)
    completion_records = info.get("completion_records", {}) if info else {}
    host_on_time = sum(record.get("host_confirmation_before_deadline") is True
                       for record in completion_records.values())
    host_confirmed = sum(record.get("host_confirmation_time") is not None
                         for record in completion_records.values())
    final_energy = float(sum(info.get("energy", {}).values())) if info else previous_energy
    return {
        "unique_key": f"{model_name}|{seed}|{scenario.tape_id}|{mode}",
        "model": model_name, "group": group, "seed": seed, "mode": mode,
        "tape_id": scenario.tape_id, "condition": getattr(scenario.communication, "__dict__", {}),
        "total_tasks": total_tasks, "completed": completed, "expired": expired,
        "undecided": max(0, total_tasks - completed - expired), "steps": steps,
        "host_confirmed": int(host_confirmed), "host_on_time": int(host_on_time),
        "energy_used": float(config.uav_count * config.initial_energy - final_energy),
        "reward_vector": total_reward.tolist(), "actor_calls": actor_calls,
        "continuation_steps": continuation_steps, "world_calls": world_calls,
        "candidate_calls": candidate_calls, "event_calls": event_calls,
        "command_submitted": command_submitted, "command_accepted": command_accepted,
        "feedback_counts": dict(feedback_counts), "trigger_counts": dict(trigger_counts),
        "noop_opportunities": noop_opportunities, "noop_selected": noop_selected,
        "illegal_new_actions": illegal_new_actions, "illegal_continuations": illegal_continuations,
        "security_violations": security_violations, "communication_messages": communications,
        "communication_proxy_bytes": communication_bytes if communication_bytes_observed else None,
        "communication_bytes_observed": communication_bytes_observed,
        "decision_chain_timing": percentiles(chain_ms),
        "model_forward_timing": percentiles(model_ms),
        "feedback_final": info.get("feedback"), "episode_end_reason": info.get("episode_end_reason"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    base = args.artifact_root.resolve()
    source = Path(__file__).resolve().parents[1]
    device = torch.device("cpu")
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    config = load_config()
    tapes = load_tapes()
    checkpoints = {
        ("P_train", 1101): base / "formal-v3-continuation-v4-run-20260918/training/seed-1101/P_train/last-recovery.pt",
        ("T_train", 1101): base / "formal-v5-history-fix-run-20260918b/training/seed-1101/T_train/last-recovery.pt",
        ("P_train", 2203): base / "formal-v5-history-fix-run-20260918b/training/seed-2203/P_train/last-recovery.pt",
        ("T_train", 2203): base / "formal-v5-history-fix-run-20260918b/training/seed-2203/T_train/last-recovery.pt",
        ("P_train", 3307): base / "formal-v5-history-fix-run-20260918b/training/seed-3307/P_train/last-recovery.pt",
        ("T_train", 3307): base / "formal-v5-history-fix-run-20260918c/training/seed-3307/T_train/last-recovery.pt",
    }
    model_info = {}
    models = {}
    for (group, seed), path in checkpoints.items():
        policy, world, payload = load_model(path, config, device)
        model_name = f"{group}/seed-{seed}"
        models[(group, seed)] = (policy, world)
        model_info[model_name] = {
            "path": str(path), "sha256": sha256(path),
            "protocol": payload.get("protocol"), "checkpoint_group": payload.get("group"),
            "counters": payload.get("counters"),
            "identity": payload.get("identity", {}),
            "policy_finite": all(torch.isfinite(v).all().item() for v in payload["policy_state_dict"].values() if torch.is_tensor(v)),
            "world_finite": all(torch.isfinite(v).all().item() for v in payload["world_state_dict"].values() if torch.is_tensor(v)),
        }

    args.out.mkdir(parents=True)
    protocol = {
        "schema": "frozen-pt-validation/1.0.0", "source_root": str(source),
        "source_hashes": {str(p.relative_to(source)): sha256(p) for p in source.rglob("*.py")},
        "artifact_root": str(base), "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__, "threads": {"intra": 4, "inter": 1},
        "validation_conditions": CONDITIONS, "validation_parent_count_per_condition": 16,
        "preference": PREFERENCE, "configurations": MATRIX,
        "episode_limit": 128, "max_episodes": 432, "max_environment_steps": 12000,
        "independent_test_read": False, "checkpoint_info": model_info,
    }
    (args.out / "evaluation-protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True, default=str), encoding="utf-8")
    rows = []
    for (group, seed), (policy, world) in models.items():
        for model_name, (eval_group, mode) in MATRIX.items():
            if eval_group != group:
                continue
            for condition in CONDITIONS:
                for scenario in tapes["validation"][condition][:16]:
                    rows.append(evaluate_episode(policy, world, config, scenario, mode, seed, group, model_name, PREFERENCE))
                    with (args.out / "episodes.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(rows[-1], sort_keys=True, default=str) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
    (args.out / "evaluation-summary.json").write_text(json.dumps({
        "status": "completed", "episode_count": len(rows),
        "environment_steps": int(sum(row["steps"] for row in rows)),
        "independent_test_read": False, "models": model_info,
    }, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({"status": "completed", "episodes": len(rows),
                      "environment_steps": int(sum(row["steps"] for row in rows)), "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
