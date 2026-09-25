"""One authorized P/T micro validation for post-update history and U/R work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import torch

from gppo_world.budget_executor import PersistentBudget
from run_event_trigger_aware_gppo import interleave, load_config, load_tapes, run_training
from gppo_world.joint_gppo import JointTrainConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"refusing to reuse micro output: {out}")
    out.mkdir(parents=True)
    budget_path = out / "persistent-budget.json"
    budget = PersistentBudget(
        budget_path,
        limits={"environment_steps": 256, "policy_optimizer_calls": 4, "world_optimizer_calls": 0},
        attempt_id="micro-controller",
    )
    config = load_config()
    tapes = load_tapes()
    train_scenarios = interleave([tapes["train"][condition] for condition in ("I", "W1", "W2")])
    protocol = {
        "schema": "window-budget-fix-micro/2.0.0",
        "window_protocol": "event-trigger-semi-markov-window-boundary-v3-history-replay",
        "groups": ["P_train", "T_train"], "seed": 1101,
        "environment_steps_each_path": 64, "environment_steps_total": 256,
        "policy_optimizer_calls_each_path": 1, "policy_optimizer_calls_total": 4,
        "paths": ["U_continuous", "R_reload"], "rollout_steps": 32,
        "world_optimizer_calls": 0, "new_tape": False, "final_test_read": False,
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "started_at": time.time(),
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    results = {}
    started = time.perf_counter()
    for group in ("P_train", "T_train"):
        u_out = out / f"{group}_U"
        u_result = run_training(
            group=group, seed=1101, output_dir=u_out, env_config=config,
            train_config=JointTrainConfig(seed=1101, rollout_steps=32),
            scenarios=train_scenarios, max_steps=64, max_policy_updates=1,
            wall_seconds=1200, smoke=True, budget=budget,
            budget_attempt_id=f"micro-v2-{group}-U",
        )
        pre_out = out / f"{group}_R_pre"
        pre_result = run_training(
            group=group, seed=1101, output_dir=pre_out, env_config=config,
            train_config=JointTrainConfig(seed=1101, rollout_steps=32),
            scenarios=train_scenarios, max_steps=33, max_policy_updates=1,
            wall_seconds=1200, smoke=True, budget=budget,
            budget_attempt_id=f"micro-v2-{group}-R-pre",
        )
        post_out = out / f"{group}_R_post"
        post_result = run_training(
            group=group, seed=1101, output_dir=post_out, env_config=config,
            train_config=JointTrainConfig(seed=1101, rollout_steps=32),
            scenarios=train_scenarios, max_steps=64, max_policy_updates=1,
            wall_seconds=1200, smoke=True, budget=budget,
            budget_attempt_id=f"micro-v2-{group}-R-post",
            resume_from=pre_out / "last-recovery.pt",
        )
        results[group] = {"U": u_result, "R_pre": pre_result, "R_post": post_result}
    summary = {"status": "passed", "elapsed_seconds": time.perf_counter() - started,
               "results": results, "budget": budget.snapshot()}
    (out / "micro-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": "passed", "out": str(out), "budget": summary["budget"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
