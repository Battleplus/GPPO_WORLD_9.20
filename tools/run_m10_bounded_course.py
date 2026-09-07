"""Run the bounded post-lease-fix Graph-5 Base learning course.

This runner intentionally stops after the phase-one budget.  It trains only
Graph-5 Base from scratch, evaluates every 2048 environment steps on frozen
validation tapes, and writes all checkpoints, optimizer state, tapes, curves,
and gate decisions.  It does not inspect or use a final test tape.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gppo_world.m10_environment import (
    M10Config,
    M10Environment,
    M10Scenario,
    M10TaskSpec,
    default_scenario,
    scenario_to_dict,
)
from gppo_world.m10_training import (
    M10ActorCritic,
    PPOConfig,
    _act,
    _policy_input_bundle,
    collect_rollout,
    evaluate_policy,
    save_policy,
    seed_everything,
    update_policy,
)
from tools.diagnose_m10_noop_capacity import run_controller


SEEDS = (1101, 2203, 3307)
TOTAL_STEPS = 8192
EVAL_INTERVAL = 2048
STAGE_BOUNDS = (("single_task", 2048), ("parallel_tasks", 4096), ("normal_no_fault", 8192))


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def single_task_tape(split: str, count: int, base_seed: int) -> tuple[M10Scenario, ...]:
    rows = []
    for index in range(count):
        seed = base_seed + index
        task = M10TaskSpec(f"task-{index}", 0.0, 0.5, 0.5, 12.0, 1.0, 1.0, 0, 0)
        rows.append(M10Scenario("single-task", (task,), (), seed, split, f"{split}-single-{seed}"))
    return tuple(rows)


def parallel_task_tape(split: str, count: int, base_seed: int) -> tuple[M10Scenario, ...]:
    rows = []
    for index in range(count):
        seed = base_seed + index
        tasks = (
            M10TaskSpec("task-0", 0.0, 0.5, 0.5, 14.0, 1.0, 1.0, 0, 0),
            M10TaskSpec("task-1", 0.0, 1.0, 1.0, 14.0, 1.0, 1.0, 0, 0),
        )
        rows.append(M10Scenario("parallel-tasks", tasks, (), seed, split, f"{split}-parallel-{seed}"))
    return tuple(rows)


def normal_no_fault_tape(split: str, count: int, base_seed: int) -> tuple[M10Scenario, ...]:
    return tuple(
        replace(default_scenario("normal", seed=base_seed + index, split="validation"),
                split=split, tape_id=f"{split}-normal-{base_seed + index}")
        for index in range(count)
    )


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


def audit_execution(env: M10Environment) -> dict[str, int]:
    log = list(env.execution.log)
    return {
        "duplicate_or_empty_id": sum(item.get("result") == "duplicate_or_empty_id" for item in log),
        "ack_identity": sum(item.get("result") == "ack_identity" for item in log),
        "fenced": sum(item.get("result") == "fenced" for item in log),
        "unauthorized": sum(item.get("result") in {"masked", "stale", "unknown_identity"} for item in log),
    }


@torch.no_grad()
def evaluate_gate(policy: M10ActorCritic, scenarios: Iterable[M10Scenario], config: M10Config,
                  device: str) -> dict[str, Any]:
    """Evaluate the exact base-policy execution path and retain security audit."""
    device_obj = torch.device(device)
    policy.eval()
    episodes = []
    for scenario in scenarios:
        env = M10Environment(config, scenario)
        obs = env.reset()
        hidden = None
        last_action = None
        done = False
        steps = 0
        total_reward = 0.0
        actor_calls = 0
        while not done and steps < int(config.horizon / config.decision_interval) + 3:
            vector, _, _, _ = _policy_input_bundle(env, obs, None, fusion="base", device=device_obj,
                                                   trigger_threshold=0.5)
            action, _, _, hidden = _act(policy, vector, obs["mask"], hidden, device_obj, deterministic=True)
            last_action = action
            obs, reward, done, info = env.step(action, submit_command=True)
            total_reward += float(reward)
            actor_calls += 1
            steps += 1
        audit = audit_execution(env)
        counts = info["counts"]
        episodes.append({
            "tape_id": scenario.tape_id, "seed": scenario.seed,
            "arrived_tasks": len(scenario.tasks), "completed": int(counts["completed"]),
            "expired": int(counts["expired"]), "return": total_reward,
            "steps": steps, "actor_calls": actor_calls, "audit": audit,
            "active_leases": len(env.execution.leases),
            "valid_candidate_steps": sum(int(np.asarray(row["mask"][:-1]).sum() > 0)
                                          for row in getattr(env, "_step_records", [])),
        })
    total_tasks = sum(row["arrived_tasks"] for row in episodes)
    completed = sum(row["completed"] for row in episodes)
    security = {key: sum(row["audit"][key] for row in episodes)
                for key in ("duplicate_or_empty_id", "ack_identity", "fenced", "unauthorized")}
    return {"episodes": episodes, "summary": {
        "episodes": len(episodes), "arrived_tasks": total_tasks, "completed": completed,
        "completion_rate": completed / max(total_tasks, 1),
        "expired": sum(row["expired"] for row in episodes),
        "return_mean": float(np.mean([row["return"] for row in episodes])) if episodes else 0.0,
        "actor_calls": sum(row["actor_calls"] for row in episodes),
        "security": security,
    }}


def legal_completion(scenarios: Iterable[M10Scenario], config: M10Config) -> dict[str, Any]:
    episodes = [run_controller(scenario, config, "rule") for scenario in scenarios]
    tasks = sum(len(scenario.tasks) for scenario in scenarios)
    completed = sum(item["completed"] for item in episodes)
    return {"arrived_tasks": tasks, "completed": completed,
            "completion_rate": completed / max(tasks, 1),
            "episodes": len(episodes)}


def evaluate_bundle(policy: M10ActorCritic, tapes: dict[str, tuple[M10Scenario, ...]], config: M10Config,
                    device: str) -> dict[str, Any]:
    result = {}
    for name, scenarios in tapes.items():
        policy_eval = evaluate_gate(policy, scenarios, config, device)
        rule_eval = legal_completion(scenarios, config)
        result[name] = {"policy": policy_eval, "legal_scheduler": rule_eval,
                        "completion_gap": policy_eval["summary"]["completion_rate"] - rule_eval["completion_rate"]}
    return result


def train_seed(seed: int, output: Path, config: M10Config, ppo: PPOConfig,
               train_tapes: dict[str, tuple[M10Scenario, ...]], validation_tapes: dict[str, tuple[M10Scenario, ...]],
               device: str) -> dict[str, Any]:
    policy = make_policy(config, ppo, seed, device)
    records = []
    total_steps = 0
    started = time.perf_counter()
    for stage_name, stage_end in STAGE_BOUNDS:
        stage_tape = train_tapes[stage_name]
        while total_steps < stage_end:
            chunk = min(EVAL_INTERVAL, stage_end - total_steps)
            rollout_config = replace(ppo, rollout_steps=min(ppo.rollout_steps, chunk))
            transitions = collect_rollout(
                policy, config=rollout_config, env_config=config, seed=seed + total_steps,
                device=torch.device(device), model=None, fusion="base", trigger_threshold=0.5,
                scenarios=stage_tape,
            )
            update = update_policy(policy, transitions, ppo=rollout_config, device=torch.device(device))
            total_steps += len(transitions)
            checkpoint = output / "checkpoints" / f"seed-{seed}" / f"step-{total_steps}.pt"
            metadata = {
                "variant": "graph-5-base-course", "encoder": "graph", "type_count": 5,
                "history": False, "fusion": "base", "seed": seed, "steps": total_steps,
                "stage": stage_name, "rollout_steps": len(transitions),
                "optimizer_updates": int(update.get("optimizer_steps", 0)),
                "actor_decisions": int(sum(t.actor_decision for t in transitions)),
                "continuation_steps": int(sum(not t.actor_decision for t in transitions)),
                "environment_steps": len(transitions), "ppo_config": asdict(rollout_config),
                "env_config": asdict(config), "update": update,
            }
            save_policy(checkpoint, policy, metadata)
            metrics = evaluate_bundle(policy, validation_tapes, config, device)
            records.append({"stage": stage_name, "steps": total_steps, "checkpoint": str(checkpoint),
                            "update": update, "metrics": metrics})
    final = evaluate_bundle(policy, validation_tapes, config, device)
    return {"seed": seed, "records": records, "final": final,
            "elapsed_seconds": time.perf_counter() - started,
            "total_environment_steps": total_steps,
            "stopping_reason": "phase_one_budget_exhausted"}


def gate_summary(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    passed = True
    for result in seed_results:
        single = result["final"]["single_task"]["policy"]["summary"]
        parallel = result["final"]["parallel_tasks"]["policy"]["summary"]
        rule = result["final"]["parallel_tasks"]["legal_scheduler"]["completion_rate"]
        security = parallel["security"] | single["security"]
        row = {
            "seed": result["seed"], "single_completion_rate": single["completion_rate"],
            "parallel_completion_rate": parallel["completion_rate"],
            "parallel_legal_scheduler_rate": rule,
            "parallel_gap_vs_legal": parallel["completion_rate"] - rule,
            "security": security,
            "single_gate": single["completion_rate"] >= 0.95,
            "parallel_gate": parallel["completion_rate"] >= 0.90 and parallel["completion_rate"] >= rule - 0.05,
            "security_gate": all(value == 0 for value in security.values()),
        }
        row["passed"] = bool(row["single_gate"] and row["parallel_gate"] and row["security_gate"])
        passed = passed and row["passed"]
        rows.append(row)
    return {"all_seeds_passed": passed, "per_seed": rows,
            "decision": "proceed_to_weak_communication_course" if passed else "stop_after_phase_one_preserve_negative_results"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = M10Config()
    ppo = PPOConfig(rollout_steps=256)
    train_tapes = {
        "single_task": single_task_tape("course-train", 32, 8101),
        "parallel_tasks": parallel_task_tape("course-train", 32, 8201),
        "normal_no_fault": normal_no_fault_tape("course-train", 64, 8301),
    }
    validation_tapes = {
        "single_task": single_task_tape("course-validation", 64, 9101),
        "parallel_tasks": parallel_task_tape("course-validation", 64, 9201),
        "normal_no_fault": normal_no_fault_tape("course-validation", 64, 9301),
    }
    dump(args.output / "protocol.json", {
        "format": "m10-bounded-course/0.1.0", "variant": "graph-5-base",
        "training_from_scratch": True, "seeds": list(SEEDS),
        "budget_per_seed": TOTAL_STEPS, "evaluation_interval": EVAL_INTERVAL,
        "stage_bounds": [{"name": name, "end_step": end} for name, end in STAGE_BOUNDS],
        "environment": asdict(config), "ppo": asdict(ppo),
        "validation_minimum": 64, "final_test_used": False,
        "gate_definition": "single >=95%; parallel >=90% and no more than 5pp below legal scheduler; security errors zero",
        "execution_contract": "parallel acknowledged leases with networked renewal and ACK confirmation",
        "no_reward_or_noop_change": True,
    })
    for group, tape_map in (("train", train_tapes), ("validation", validation_tapes)):
        for name, tape in tape_map.items():
            dump(args.output / "tapes" / group / f"{name}.json", [scenario_to_dict(item) for item in tape])
    results = []
    for seed in SEEDS:
        result = train_seed(seed, args.output, config, ppo, train_tapes, validation_tapes, args.device)
        results.append(result)
        dump(args.output / "per-seed" / f"seed-{seed}.json", result)
    gates = gate_summary(results)
    dump(args.output / "course-results.json", {"protocol": "m10-bounded-course/0.1.0", "results": results, "gates": gates})
    dump(args.output / "run-complete.json", {"status": "completed", "training_started": True,
                                               "seeds": list(SEEDS), "steps_per_seed": TOTAL_STEPS,
                                               "phase_one_gates": gates})
    print(json.dumps(gates, ensure_ascii=False, indent=2))
    raise SystemExit(0 if gates["all_seeds_passed"] else 2)


if __name__ == "__main__":
    main()
