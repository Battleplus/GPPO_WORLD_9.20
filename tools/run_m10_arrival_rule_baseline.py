"""Run the legal public-information rule under the arrival contract only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.m10_environment import M10Config, M10Environment, weak_communication_tape  # noqa: E402
from tools.run_m10_baseline_comparison import traditional_action  # noqa: E402


def run_basis(basis: str, count: int, base_seed: int) -> dict[str, object]:
    config = M10Config(task_completion_mode="arrival_to_region", deadline_basis=basis)
    episodes = []
    for scenario in weak_communication_tape("test", count=count, base_seed=base_seed, level="composite"):
        env = M10Environment(config=config, scenario=scenario)
        obs = env.reset()
        done = False
        actions = []
        total_reward = 0.0
        while not done and len(actions) < int(config.horizon / config.decision_interval) + 2:
            action = traditional_action(obs)
            obs, reward, done, info = env.step(action, submit_command=True)
            actions.append(int(action))
            total_reward += float(reward)
        episodes.append({
            "tape_id": scenario.tape_id,
            "steps": len(actions),
            "actions": actions,
            "return": total_reward,
            "counts": info["counts"],
            "deadline_basis": basis,
            "task_completion_mode": config.task_completion_mode,
            "completion_records": info["completion_records"],
            "communication": info["communication_log"],
            "execution_log": list(env.execution.log),
            "safety": {
                "duplicate_accepts": len([x for x in env.execution.log if x.get("result") == "accepted"]) - len({x.get("command_id") for x in env.execution.log if x.get("result") == "accepted"}),
                "unauthorized_or_fenced": sum(x.get("result") in ("fenced", "stale", "expired", "unknown_command") for x in env.execution.log),
            },
        })
    return {
        "basis": basis,
        "episodes": episodes,
        "summary": {
            "episodes": len(episodes),
            "completed_mean": float(np.mean([x["counts"]["completed"] for x in episodes])),
            "expired_mean": float(np.mean([x["counts"]["expired"] for x in episodes])),
            "return_mean": float(np.mean([x["return"] for x in episodes])),
            "arrival_events": sum(len(x["completion_records"]) for x in episodes),
            "host_confirmed": sum(sum(record.get("host_confirmation_time") is not None for record in x["completion_records"].values()) for x in episodes),
            "safety_violations": sum(x["safety"]["duplicate_accepts"] + x["safety"]["unauthorized_or_fenced"] for x in episodes),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--base-seed", type=int, default=93001)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing non-empty output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    result = {"protocol": "world-gppo-9.11-arrival/0.1.0", "count": args.count, "base_seed": args.base_seed,
              "generated_at": time.time(), "training_performed": False,
              "bases": {basis: run_basis(basis, args.count, args.base_seed) for basis in ("physical_arrival", "host_confirmation")}}
    (args.out / "rule-baseline.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({basis: result["bases"][basis]["summary"] for basis in result["bases"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
