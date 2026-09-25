"""Final authorized P/T U-vs-R micro validation on the repaired source."""

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
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"refusing to reuse final U/R output: {out}")
    out.mkdir(parents=True)
    budget = PersistentBudget(
        out / "persistent-budget.json",
        limits={"environment_steps": 256, "policy_optimizer_calls": 4, "world_optimizer_calls": 0},
        attempt_id="micro-v3-final-ur",
    )
    config = load_config()
    tapes = load_tapes()
    scenarios = interleave([tapes["train"][condition] for condition in ("I", "W1", "W2")])
    results = {}
    started = time.perf_counter()
    for group in ("P_train", "T_train"):
        common = dict(group=group, seed=1101, env_config=config,
                      train_config=JointTrainConfig(seed=1101, rollout_steps=32),
                      scenarios=scenarios, wall_seconds=1200, smoke=True, budget=budget)
        u = run_training(output_dir=out / f"{group}_U", max_steps=64, max_policy_updates=1,
                         budget_attempt_id=f"micro-v3-{group}-U",
                         capture_boundary_checkpoint=True, **common)
        r_pre = run_training(output_dir=out / f"{group}_R_pre", max_steps=64, max_policy_updates=1,
                             budget_attempt_id=f"micro-v3-{group}-R-pre",
                             stop_after_update_boundary=True, **common)
        r_post = run_training(output_dir=out / f"{group}_R_post", max_steps=64, max_policy_updates=1,
                              budget_attempt_id=f"micro-v3-{group}-R-post",
                              resume_from=out / f"{group}_R_pre" / "last-recovery.pt", **common)
        results[group] = {"U": u, "R_pre": r_pre, "R_post": r_post}
    summary = {"status": "completed", "results": results,
               "budget": budget.snapshot(), "elapsed_seconds": time.perf_counter() - started,
               "protocol": {"rollout_steps": 32, "path_steps_max": 64,
                             "policy_updates_max": 1, "world_updates": 0,
                             "no_new_tape": True, "no_test_read": True}}
    (out / "ur-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "budget": summary["budget"],
                      "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
