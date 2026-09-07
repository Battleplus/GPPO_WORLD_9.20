"""Continue Graph-5 Base from phase one through a bounded weak-link course.

The phase-one checkpoint is required to pass its gates before this runner
starts.  This is A-only continuation training; B/C are deliberately not
started here.  The final test tape is never read by this script.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gppo_world.m10_environment import M10Config, scenario_to_dict, weak_communication_tape
from gppo_world.m10_training import (
    M10ActorCritic,
    PPOConfig,
    collect_rollout,
    save_policy,
    seed_everything,
    update_policy,
)
from tools.run_m10_bounded_course import SEEDS, evaluate_gate, legal_completion


NEW_STEPS = 16384
EVAL_INTERVAL = 2048
LEVEL_BOUNDS = (
    ("telemetry-delay", 2048),
    ("random-loss", 4096),
    ("burst-loss", 6144),
    ("reorder", 8192),
    ("recovery", 12288),
    ("composite", 16384),
)


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def make_policy(config: M10Config, ppo: PPOConfig, seed: int, device: str) -> M10ActorCritic:
    seed_everything(seed)
    policy = M10ActorCritic(
        uav_count=config.uav_count, task_capacity=config.task_capacity,
        action_count=config.action_count, encoder="graph", type_count=5,
        history=False, context_dim=0, region_count=config.region_count,
        target_count=config.target_count, event_capacity=config.event_capacity,
        relation_width=config.relation_width,
    ).to(torch.device(device))
    policy.optimizer = torch.optim.Adam(policy.parameters(), lr=ppo.learning_rate)  # type: ignore[attr-defined]
    return policy


def load_phase_one(path: Path, config: M10Config, ppo: PPOConfig, seed: int, device: str) -> M10ActorCritic:
    payload = torch.load(path, map_location=device, weights_only=False)
    metadata = payload.get("metadata", {})
    if int(metadata.get("seed", -1)) != seed or int(metadata.get("steps", -1)) != 8192:
        raise ValueError(f"phase-one checkpoint mismatch: {path}")
    if metadata.get("variant") != "graph-5-base-course":
        raise ValueError(f"not a phase-one Graph-5 checkpoint: {path}")
    policy = make_policy(config, ppo, seed, device)
    policy.load_state_dict(payload["state_dict"])
    if payload.get("optimizer_state_dict") is None:
        raise ValueError("phase-one checkpoint lacks optimizer recovery state")
    policy.optimizer.load_state_dict(payload["optimizer_state_dict"])  # type: ignore[attr-defined]
    return policy


def recovery_pairs(scenario: Any, episode: dict[str, Any]) -> tuple[int, int, list[dict[str, Any]]]:
    """Return event-task pairs where the reference-defined handoff is auditable.

    A pair is in the denominator only when an event interrupted a task on the
    affected UAV and the same task was later accepted by another UAV.  A
    completed task then counts as recovered.  Events with no interrupted task
    are reported separately and do not inflate the denominator.
    """
    clock_log = episode.get("clock_log", [])
    commands = episode.get("commands", {})
    execution_log = episode.get("execution_log", [])
    pairs = []
    for event in scenario.events:
        if event.kind not in ("damage", "disconnect"):
            continue
        interrupted = set()
        for row in clock_log:
            if row.get("kind") not in ("travel", "service") or row.get("resource") != event.resource:
                continue
            if abs(float(row.get("end", -1.0)) - float(event.time)) <= 1e-7:
                interrupted.add(str(row["task"]))
        for task_id in sorted(interrupted):
            handoffs = []
            for item in execution_log:
                if item.get("result") != "accepted":
                    continue
                command = commands.get(item.get("command_id"), {})
                if command.get("task_id") == task_id and command.get("uav_id") != event.resource:
                    handoffs.append({"time": float(item.get("time", 0.0)), "uav_id": command.get("uav_id")})
            if handoffs:
                final = episode.get("tasks", {}).get(task_id, {})
                pairs.append({"event": event.kind, "resource": event.resource, "task_id": task_id,
                              "handoff": handoffs, "recovered": final.get("state") == "completed"})
    recovered = sum(bool(item["recovered"]) for item in pairs)
    return recovered, len(pairs), pairs


def evaluate_level(policy: M10ActorCritic, scenarios: tuple[Any, ...], config: M10Config,
                   device: str) -> dict[str, Any]:
    policy_eval = evaluate_gate(policy, scenarios, config, device)
    reference = [run_reference(scenario, config) for scenario in scenarios]
    policy_recovered = policy_total = reference_recovered = reference_total = 0
    recovery_detail = []
    for scenario, episode, reference_episode in zip(scenarios, policy_eval["episodes"], reference):
        good, total, detail = recovery_pairs(scenario, episode)
        ref_good, ref_total, ref_detail = recovery_pairs(scenario, reference_episode)
        policy_recovered += good
        policy_total += total
        reference_recovered += ref_good
        reference_total += ref_total
        recovery_detail.append({"tape_id": scenario.tape_id, "policy": detail, "reference": ref_detail})
    summary = policy_eval["summary"]
    reference_summary = {
        "arrived_tasks": sum(len(s.tasks) for s in scenarios),
        "completed": sum(item["completed"] for item in reference),
        "expired": sum(item["expired"] for item in reference),
    }
    reference_summary["completion_rate"] = reference_summary["completed"] / max(reference_summary["arrived_tasks"], 1)
    reference_summary["deadline_violation_rate"] = reference_summary["expired"] / max(reference_summary["arrived_tasks"], 1)
    return {
        "policy": policy_eval, "legal_scheduler": reference_summary,
        "policy_deadline_violation_rate": summary["expired"] / max(summary["arrived_tasks"], 1),
        "reference_deadline_violation_rate": reference_summary["deadline_violation_rate"],
        "completion_gap": summary["completion_rate"] - reference_summary["completion_rate"],
        "recovery": {
            "definition": "event-interrupted task later accepted by another UAV; completion is recovery",
            "policy_recovered": policy_recovered, "policy_recoverable": policy_total,
            "policy_rate": policy_recovered / max(policy_total, 1),
            "reference_recovered": reference_recovered, "reference_recoverable": reference_total,
            "reference_rate": reference_recovered / max(reference_total, 1),
            "details": recovery_detail,
        },
    }


def run_reference(scenario: Any, config: M10Config) -> dict[str, Any]:
    from tools.diagnose_m10_noop_capacity import run_controller
    result = run_controller(scenario, config, "rule")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-one", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    phase_one_status = json.loads((args.phase_one / "run-complete.json").read_text(encoding="utf-8"))
    if not phase_one_status.get("phase_one_gates", {}).get("all_seeds_passed", False):
        raise SystemExit("phase-one gates did not pass; weak course not started")
    args.output.mkdir(parents=True, exist_ok=True)
    config = M10Config()
    ppo = PPOConfig(rollout_steps=256)
    train_tapes = {level: weak_communication_tape("train", count=32, base_seed=10001 + 100 * i, level=level)
                   for i, (level, _) in enumerate(LEVEL_BOUNDS)}
    validation_tapes = {level: weak_communication_tape("validation", count=64, base_seed=11001 + 100 * i, level=level)
                        for i, (level, _) in enumerate(LEVEL_BOUNDS)}
    dump(args.output / "protocol.json", {
        "format": "m10-weak-course/0.1.0", "variant": "graph-5-base",
        "initialized_from_phase_one": True, "phase_one_budget_per_seed": 8192,
        "new_budget_per_seed": NEW_STEPS, "cumulative_budget_per_seed": 24576,
        "levels": [{"name": level, "end_new_steps": end} for level, end in LEVEL_BOUNDS],
        "seeds": list(SEEDS), "validation_tape_count_per_level": 64,
        "final_test_used": False, "action_reward_contract_unchanged": True,
        "recoverable_event_definition": "event interrupts task on affected UAV, later accepted by another UAV; completed task is recovered",
        "gate": "composite completion gap >= -5pp; composite deadline violation <= reference +5pp; recoverable-event rate >=80%; safety errors zero",
        "long_disconnected_autonomy_return_swap_charge": "out of scope",
    })
    for group, tapes in (("train", train_tapes), ("validation", validation_tapes)):
        for level, tape in tapes.items():
            dump(args.output / "tapes" / group / f"{level}.json", [scenario_to_dict(item) for item in tape])
    seed_results = []
    for seed in SEEDS:
        policy = load_phase_one(args.phase_one / "checkpoints" / f"seed-{seed}" / "step-8192.pt",
                                config, ppo, seed, args.device)
        new_steps = 0
        records = []
        started = time.perf_counter()
        for level, end_step in LEVEL_BOUNDS:
            while new_steps < end_step:
                chunk = min(ppo.rollout_steps, end_step - new_steps)
                rollout = replace(ppo, rollout_steps=chunk)
                transitions = collect_rollout(policy, config=rollout, env_config=config,
                                              seed=seed + 8192 + new_steps, device=torch.device(args.device),
                                              model=None, fusion="base", trigger_threshold=0.5,
                                              scenarios=train_tapes[level])
                update = update_policy(policy, transitions, ppo=rollout, device=torch.device(args.device))
                new_steps += len(transitions)
                if new_steps % EVAL_INTERVAL == 0:
                    ckpt = args.output / "checkpoints" / f"seed-{seed}" / f"new-step-{new_steps}.pt"
                    metadata = {"variant": "graph-5-base-weak-course", "encoder": "graph", "type_count": 5,
                                "history": False, "fusion": "base", "seed": seed, "steps": 8192 + new_steps,
                                "new_steps": new_steps, "level": level, "optimizer_updates": update.get("optimizer_steps", 0),
                                "ppo_config": asdict(rollout), "env_config": asdict(config), "update": update}
                    save_policy(ckpt, policy, metadata)
                    metrics = evaluate_level(policy, validation_tapes[level], config, args.device)
                    records.append({"level": level, "new_steps": new_steps, "checkpoint": str(ckpt),
                                    "update": update, "metrics": metrics})
        final = evaluate_level(policy, validation_tapes["composite"], config, args.device)
        seed_result = {"seed": seed, "new_environment_steps": new_steps, "records": records,
                       "final_composite": final, "elapsed_seconds": time.perf_counter() - started,
                       "stopping_reason": "weak_course_budget_exhausted"}
        dump(args.output / "per-seed" / f"seed-{seed}.json", seed_result)
        seed_results.append(seed_result)
    checks = []
    for result in seed_results:
        m = result["final_composite"]
        recovery = m["recovery"]
        policy = m["policy"]["summary"]
        reference = m["legal_scheduler"]
        checks.append({"seed": result["seed"],
                       "completion_gap": m["completion_gap"],
                       "deadline_gap": m["policy_deadline_violation_rate"] - reference["deadline_violation_rate"],
                       "recovery_rate": recovery["policy_rate"],
                       "recoverable_events": recovery["policy_recoverable"],
                       "security": policy["security"],
                       "completion_gate": m["completion_gap"] >= -0.05,
                       "deadline_gate": m["policy_deadline_violation_rate"] <= reference["deadline_violation_rate"] + 0.05,
                       "recovery_gate": recovery["policy_recoverable"] > 0 and recovery["policy_rate"] >= 0.80,
                       "security_gate": all(value == 0 for value in policy["security"].values())})
    gates = {"all_seeds_passed": all(all(row[key] for key in ("completion_gate", "deadline_gate", "recovery_gate", "security_gate")) for row in checks),
             "per_seed": checks,
             "decision": "stop_after_weak_course_preserve_results"}
    dump(args.output / "course-results.json", {"protocol": "m10-weak-course/0.1.0", "results": seed_results, "gates": gates})
    dump(args.output / "run-complete.json", {"status": "completed", "training_started": True,
                                               "new_steps_per_seed": NEW_STEPS, "cumulative_steps_per_seed": 24576,
                                               "gates": gates})
    print(json.dumps(gates, ensure_ascii=False, indent=2))
    raise SystemExit(0 if gates["all_seeds_passed"] else 2)


if __name__ == "__main__":
    main()
