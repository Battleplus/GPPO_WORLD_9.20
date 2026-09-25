"""Finish only the uncompleted post-update U/R micro paths.

This script reuses the first attempt's persistent budget ledger.  It never
restarts the already verified P/U or P/R-pre segments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from gppo_world.budget_executor import PersistentBudget
from gppo_world.joint_gppo import JointTrainConfig
from run_event_trigger_aware_gppo import interleave, load_config, load_tapes, run_training


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"refusing to reuse retry output: {out}")
    out.mkdir(parents=True)
    budget = PersistentBudget(
        args.budget.resolve(),
        limits={"environment_steps": 256, "policy_optimizer_calls": 4, "world_optimizer_calls": 0},
        attempt_id="micro-v2-retry",
    )
    config = load_config()
    tapes = load_tapes()
    train_scenarios = interleave([tapes["train"][condition] for condition in ("I", "W1", "W2")])
    started = time.perf_counter()
    p_pre = args.previous.resolve() / "P_train_R_pre" / "last-recovery.pt"
    p_post = run_training(
        group="P_train", seed=1101, output_dir=out / "P_train_R_post",
        env_config=config, train_config=JointTrainConfig(seed=1101, rollout_steps=32),
        scenarios=train_scenarios, max_steps=64, max_policy_updates=1,
        wall_seconds=1200, smoke=True, budget=budget,
        budget_attempt_id="micro-v2-retry-P-R-post", resume_from=p_pre,
    )
    results = {"P_train": {"R_post": p_post}}
    for group in ("T_train",):
        u_result = run_training(
            group=group, seed=1101, output_dir=out / f"{group}_U",
            env_config=config, train_config=JointTrainConfig(seed=1101, rollout_steps=32),
            scenarios=train_scenarios, max_steps=64, max_policy_updates=1,
            wall_seconds=1200, smoke=True, budget=budget,
            budget_attempt_id=f"micro-v2-retry-{group}-U",
        )
        pre_result = run_training(
            group=group, seed=1101, output_dir=out / f"{group}_R_pre",
            env_config=config, train_config=JointTrainConfig(seed=1101, rollout_steps=32),
            scenarios=train_scenarios, max_steps=33, max_policy_updates=1,
            wall_seconds=1200, smoke=True, budget=budget,
            budget_attempt_id=f"micro-v2-retry-{group}-R-pre",
        )
        post_result = run_training(
            group=group, seed=1101, output_dir=out / f"{group}_R_post",
            env_config=config, train_config=JointTrainConfig(seed=1101, rollout_steps=32),
            scenarios=train_scenarios, max_steps=64, max_policy_updates=1,
            wall_seconds=1200, smoke=True, budget=budget,
            budget_attempt_id=f"micro-v2-retry-{group}-R-post",
            resume_from=out / f"{group}_R_pre" / "last-recovery.pt",
        )
        results[group] = {"U": u_result, "R_pre": pre_result, "R_post": post_result}
    summary = {"status": "completed", "results": results,
               "budget": budget.snapshot(),
               "elapsed_seconds": time.perf_counter() - started,
               "previous_attempt": str(args.previous.resolve())}
    (out / "retry-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "budget": summary["budget"],
                      "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
