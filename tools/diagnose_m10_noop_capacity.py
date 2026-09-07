"""M-10 NOOP, reward, action-capacity and small-scenario feasibility audit.

No training is performed.  Policy evaluations consume frozen checkpoints and
the frozen evaluation tape.  Rule controllers use only the current public
observation and pass through the ordinary bridge/execution gates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gppo_world.m10_communication import weak_communication_profile
from gppo_world.m10_environment import M10Config, M10Environment, M10Scenario, M10TaskSpec, scenario_from_dict
from gppo_world.m10_training import masked_distribution, _policy_input_bundle, _trigger_decision
from gppo_world.service_clock import ServiceEvent
from tools.run_m10_r3 import load_policy, load_world


NOOP = "noop"


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def public_task_key(obs: dict[str, Any], action: int, config: M10Config) -> tuple[float, float, float, float, int]:
    uav, task = divmod(action, config.task_capacity)
    urow = np.asarray(obs["uavs"])[uav]
    trow = np.asarray(obs["tasks"])[task]
    values = trow[::4]
    ux, uy = float(urow[0]), float(urow[4])
    tx, ty = float(values[0]), float(values[4])
    deadline, remaining, priority = float(values[2]), float(values[3]), float(values[4])
    distance = math.dist((ux, uy), (tx, ty))
    return deadline, distance, remaining, -priority, action


def deadline_distance_action(env: M10Environment, obs: dict[str, Any]) -> int:
    mask = np.asarray(obs["mask"], dtype=np.bool_)
    candidates = [int(x) for x in np.flatnonzero(mask[:-1])]
    if not candidates:
        return env.config.action_count - 1
    return min(candidates, key=lambda action: public_task_key(obs, action, env.config))


def reward_totals(env: M10Environment) -> dict[str, float]:
    totals = {"completion": 0.0, "expiry": 0.0, "energy": 0.0, "rejection": 0.0, "total": 0.0}
    for item in getattr(env, "_feedback_log", []):
        # The environment exposes per-step components in info; this fallback
        # is used only when a legacy checkpoint/evaluator did not retain them.
        del item
    return totals


def run_controller(scenario: M10Scenario, config: M10Config, mode: str) -> dict[str, Any]:
    env = M10Environment(config, scenario)
    obs = env.reset()
    active_actions: set[int] = set()
    done = False
    rows = []
    totals = {"completion": 0.0, "expiry": 0.0, "energy": 0.0, "rejection": 0.0, "total": 0.0}
    while not done and len(rows) < int(config.horizon / config.decision_interval) + 3:
        if mode == NOOP:
            action, submit, control = config.action_count - 1, True, "always_noop"
        elif int(np.asarray(obs["mask"][:-1]).sum()) > 0:
            action, submit, control = deadline_distance_action(env, obs), True, "deadline_distance_priority"
        elif active_actions:
            action, submit, control = min(active_actions), False, "continue_existing_leases"
        else:
            action, submit, control = deadline_distance_action(env, obs), True, "deadline_distance_priority"
            active_actions = set()
        execution_before = len(env.execution.log)
        obs, reward, done, info = env.step(action, submit_command=submit)
        components = dict(info.get("reward_components", {"total": float(reward)}))
        for key in totals:
            totals[key] += float(components.get(key, 0.0))
        rows.append({"step": len(rows), "time": float(info["time"]), "action": int(action),
                     "submit": submit, "control": control, "feedback": info["feedback"],
                     "mask_candidates": int(np.asarray(obs["mask"][:-1]).sum()),
                     "active_continuations": list(info.get("active_continuations", [])),
                     "lease_renewals": dict(info.get("lease_renewals", {})),
                     "execution_delta": list(env.execution.log[execution_before:]),
                     "reward_components": components})
        active_actions = {int(item["action"]) for item in info.get("active_continuations", [])}
    accepted = [x for x in env.execution.log if x.get("result") == "accepted"]
    return {"mode": mode, "completed": sum(t.state.value == "completed" for t in env.clock.tasks.values()),
            "expired": sum(t.state.value == "expired" for t in env.clock.tasks.values()),
            "accepted_commands": len(accepted), "reward_components": totals,
            "rows": rows, "execution_log": list(env.execution.log), "clock_log": list(env.clock.log),
            "tasks": {k: {"state": v.state.value, "service": float(v.service), "deadline": float(v.deadline)}
                      for k, v in env.clock.tasks.items()}}


def run_policy(policy_path: Path, world: Any, scenarios: list[M10Scenario], config: M10Config,
               device: str, threshold: float = 0.1) -> dict[str, Any]:
    policy, metadata = load_policy(policy_path, config, device)
    device_obj = torch.device(device)
    fusion = str(metadata.get("fusion", "base"))
    records = []
    for scenario in scenarios:
        env = M10Environment(config, scenario)
        obs = env.reset()
        hidden = None
        last_action = None
        since = 3
        done = False
        actor_calls = continuation = valid_steps = noop_valid = 0
        accepted = 0
        total_reward = 0.0
        components = {"completion": 0.0, "expiry": 0.0, "energy": 0.0, "rejection": 0.0, "total": 0.0}
        previous_counts = {"completed": 0, "expired": 0}
        previous_energy = sum(float(resource.energy) for resource in env.clock.resources.values())
        used_fallback_components = False
        noop_probs: list[float] = []
        entropies: list[float] = []
        trace = []
        while not done and len(trace) < int(config.horizon / config.decision_interval) + 3:
            valid_count = int(np.asarray(obs["mask"][:-1]).sum())
            vector, risk_active, risk, full_vector = _policy_input_bundle(
                env, obs, world if fusion != "base" else None, fusion=fusion,
                device=device_obj, trigger_threshold=threshold,
            )
            if fusion == "triggered":
                should, reason, conditions = _trigger_decision(
                    obs, fusion=fusion, risk_active=risk_active, last_action=last_action,
                    steps_since_replan=since, max_replan_interval=3)
            else:
                should, reason, conditions = True, "periodic_policy", {}
            if should:
                decision_vector = full_vector if fusion == "triggered" else vector
                tensor = torch.tensor(decision_vector, dtype=torch.float32, device=device_obj)[None, :]
                mask = torch.tensor(obs["mask"], dtype=torch.bool, device=device_obj)[None, :]
                logits, value, hidden = policy(tensor, hidden)
                dist = masked_distribution(logits, mask)
                action = int(torch.argmax(dist.logits, dim=-1).item())
                noop_probs.append(float(dist.probs[0, config.action_count - 1].item()))
                entropies.append(float(dist.entropy().item()))
                actor_calls += 1
                if valid_count and action == config.action_count - 1:
                    noop_valid += 1
                valid_steps += int(valid_count > 0)
                submit = True
                since = 0
            else:
                _, hidden = policy.value_only(torch.tensor(vector, dtype=torch.float32, device=device_obj)[None, :], hidden)
                action = int(last_action)
                continuation += 1
                submit = False
                since += 1
            last_action = action
            before_exec = len(env.execution.log)
            obs, reward, done, info = env.step(action, submit_command=submit)
            accepted += sum(x.get("result") == "accepted" for x in env.execution.log[before_exec:])
            total_reward += float(reward)
            reported_components = info.get("reward_components")
            if reported_components:
                for key in components:
                    components[key] += float(reported_components.get(key, 0.0))
            else:
                # Historical weak-communication source did not expose the
                # component audit in info. Reconstruct the same public
                # accounting from counts, post-step energy, and feedback;
                # this does not change the environment or the reward.
                counts = info.get("counts", {})
                completion = config.reward_completion * (
                    int(counts.get("completed", 0)) - previous_counts["completed"])
                expiry = -config.penalty_expired * (
                    int(counts.get("expired", 0)) - previous_counts["expired"])
                energy_now = sum(float(value) for value in info.get("energy", {}).values())
                energy = -config.energy_cost_weight * max(0.0, previous_energy - energy_now)
                feedback_name = str(info.get("feedback", ""))
                step_accepted = any(x.get("result") == "accepted" for x in env.execution.log[before_exec:])
                rejection = 0.0 if step_accepted or feedback_name in {"noop", "reuse_existing", "awaiting_ack"} else -config.penalty_rejected
                components["completion"] += completion
                components["expiry"] += expiry
                components["energy"] += energy
                components["rejection"] += rejection
                components["total"] += completion + expiry + energy + rejection
                used_fallback_components = True
                previous_counts = {"completed": int(counts.get("completed", 0)),
                                   "expired": int(counts.get("expired", 0))}
                previous_energy = energy_now
            trace.append({"time": float(info["time"]), "valid_candidates": valid_count,
                          "action": action, "action_is_noop": action == config.action_count - 1,
                          "replan": should, "reason": reason, "risk": float(risk),
                          "noop_probability": noop_probs[-1] if should else None,
                          "entropy": entropies[-1] if should else None,
                          "feedback": info["feedback"],
                          "active_continuations": list(info.get("active_continuations", [])),
                          "lease_renewals": dict(info.get("lease_renewals", {})),
                          "accepted_delta": list(env.execution.log[before_exec:])})
        if used_fallback_components:
            # Reconcile the residual with the environment's exact scalar
            # return because historical server source omitted component info.
            components["energy"] = total_reward - components["completion"] - components["expiry"] - components["rejection"]
            components["total"] = total_reward
        records.append({"tape_id": scenario.tape_id, "seed": scenario.seed,
                        "completed": sum(t.state.value == "completed" for t in env.clock.tasks.values()),
                        "expired": sum(t.state.value == "expired" for t in env.clock.tasks.values()),
                        "return": total_reward, "accepted_commands": accepted,
                        "actor_calls": actor_calls, "continuation_steps": continuation,
                        "valid_candidate_steps": valid_steps, "noop_on_valid_steps": noop_valid,
                        "noop_rate_when_valid": noop_valid / max(valid_steps, 1),
                        "noop_probability_mean": float(np.mean(noop_probs)) if noop_probs else None,
                        "entropy_mean": float(np.mean(entropies)) if entropies else None,
                        "reward_components": components, "trace": trace})
    numeric = ("completed", "expired", "return", "accepted_commands", "actor_calls",
               "continuation_steps", "valid_candidate_steps", "noop_on_valid_steps")
    summary = {key: {"mean": float(np.mean([row[key] for row in records])),
                     "std": float(np.std([row[key] for row in records]))} for key in numeric}
    summary["noop_rate_when_valid_mean"] = float(np.mean([x["noop_rate_when_valid"] for x in records]))
    summary["noop_probability_mean"] = float(np.mean([x["noop_probability_mean"] for x in records if x["noop_probability_mean"] is not None]))
    summary["entropy_mean"] = float(np.mean([x["entropy_mean"] for x in records if x["entropy_mean"] is not None]))
    summary["reward_components"] = {key: float(np.mean([x["reward_components"][key] for x in records]))
                                     for key in ("completion", "expiry", "energy", "rejection", "total")}
    payload = torch.load(policy_path, map_location="cpu", weights_only=False)
    updates = metadata.get("updates", [])
    summary["checkpoint_audit"] = {
        "optimizer_state_present": payload.get("optimizer_state_dict") is not None,
        "optimizer_updates_reported": metadata.get("optimizer_updates"),
        "rollout_updates_reported": metadata.get("rollout_updates"),
        "update_epochs_reported": metadata.get("update_epochs"),
        "update_records": len(updates),
        "update_means": {key: float(np.mean([float(x[key]) for x in updates if key in x]))
                         for key in ("loss", "policy_loss", "value_loss", "entropy", "approx_kl")
                         if any(key in x for x in updates)},
        "advantage_distribution": "not exported in historical checkpoint; GAE semantics tested separately",
    }
    return {"policy": str(policy_path), "metadata": metadata, "summary": summary, "episodes": records}


def capacity_and_feasibility(scenarios: list[M10Scenario], config: M10Config) -> dict[str, Any]:
    feasible = impossible = 0
    details = []
    for scenario in scenarios:
        for task in scenario.tasks:
            travel = math.dist((0.0, 0.0), (task.x, task.y)) / config.travel_speed
            energy = travel * config.travel_power + task.service * config.service_power
            window = task.deadline - task.arrival
            ok = travel + task.service / config.service_rate <= window and energy <= config.initial_energy
            feasible += int(ok)
            impossible += int(not ok)
            details.append({"tape_id": scenario.tape_id, "task_id": task.task_id,
                            "arrival": task.arrival, "deadline": task.deadline, "window": window,
                            "shortest_travel": travel, "service": task.service,
                            "energy_lower_bound": energy, "individually_ideal_feasible": ok})
    small = M10Scenario(
        "parallel-capacity", (M10TaskSpec("t0", 0.0, 0.0, 0.0, 9.0, 4.0, 1.0),
                              M10TaskSpec("t1", 0.0, 0.0, 0.0, 9.0, 4.0, 1.0)), (), seed=8801,
        split="diagnostic", tape_id="parallel-capacity")
    small_config = M10Config(uav_count=2, task_capacity=2, region_count=1, target_count=1,
                             event_capacity=1, horizon=9.0)
    env = M10Environment(small_config, small)
    env.reset()
    sequence = []
    for action in (0, 3, 3, 3, 3, 3, 3):
        _, _, done, info = env.step(action, submit_command=True)
        sequence.append({"time": info["time"], "action": action, "feedback": info["feedback"],
                         "tasks": info["tasks"], "service": info["task_service"],
                         "active_leases": len(env.execution.leases)})
        if done:
            break
    return {"individual_task_check": {"tasks": len(details), "ideal_individually_feasible": feasible,
                                       "ideal_individually_impossible": impossible,
                                       "note": "necessary per-task check only; not a schedule upper bound"},
            "task_details": details,
            "parallel_assignment_probe": {
                "action_count": small_config.action_count, "one_new_command_per_step": True,
                "sequence": sequence, "final_states": {k: v.state.value for k, v in env.clock.tasks.items()},
                "final_leases": len(env.execution.leases),
                "interpretation": "multiple UAV assignments can be accepted on successive steps, but the current environment exposes/renews only one active continuation handle; prolonged parallel work requires an explicit protocol decision",
            }}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tapes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world", type=Path)
    parser.add_argument("--policy-root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-episodes", type=int, default=16)
    args = parser.parse_args()
    payload = json.loads(args.tapes.read_text(encoding="utf-8"))
    scenarios = [scenario_from_dict(item) for item in payload["final_test"]][:args.max_episodes]
    config = M10Config()
    result: dict[str, Any] = {
        "protocol": "M-10 NOOP/action-capacity/feasibility audit; no training",
        "tape_ids": [x.tape_id for x in scenarios],
        "controllers": {},
        "capacity_and_feasibility": capacity_and_feasibility(scenarios, config),
        "learning_evidence_limits": {
            "historical_advantages": "not retained in checkpoint artifacts",
            "parameter_delta": "initial policy not retained; checkpoint-only delta unavailable",
            "optimizer_and_loss": "reported metadata and optimizer state are audited per checkpoint",
        },
    }
    for mode in (NOOP, "deadline_distance"):
        result["controllers"][mode] = [run_controller(scenario, config, mode if mode == NOOP else "rule")
                                         for scenario in scenarios]
    if args.policy_root:
        world = None
        if args.world:
            obs = M10Environment(config, scenarios[0]).reset()
            world = load_world(args.world, len(obs["flat"]), config.action_count, args.device)
        policy_groups = {
            "A-graph5-base": args.policy_root / "A-graph5-base" / "seed-{seed}.pt",
            "B-graph5-world": args.policy_root / "B-graph5-world" / "seed-{seed}.pt",
            "C-graph5-triggered": args.policy_root / "C-graph5-triggered" / "seed-{seed}.pt",
        }
        for group, pattern in policy_groups.items():
            items = []
            for seed in (1101, 2203, 3307):
                path = Path(str(pattern).format(seed=seed))
                if path.exists():
                    items.append(run_policy(path, world, scenarios, config, args.device))
            result["controllers"][group] = items
    dump(args.output, result)


if __name__ == "__main__":
    main()
