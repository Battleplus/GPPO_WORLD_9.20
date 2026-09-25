"""Frozen validation for the event-trigger-aware training comparison."""

from __future__ import annotations

import json
from pathlib import Path
import time

from tools.run_event_trigger_aware_gppo import (
    CONDITIONS, PREFERENCE, SEEDS, evaluate_checkpoint, load_config, load_tapes, sha256, write_json,
)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    args.out.mkdir(parents=True)
    config = load_config(); tapes = load_tapes()
    validation = {condition: tapes["validation"][condition][:16] for condition in CONDITIONS}
    plan = []
    results = []
    for seed in SEEDS:
        p_ck = args.run_root / "training" / f"seed-{seed}" / "P_train" / "last-recovery.pt"
        t_ck = args.run_root / "training" / f"seed-{seed}" / "T_train" / "last-recovery.pt"
        if not p_ck.is_file() or not t_ck.is_file():
            raise FileNotFoundError(f"missing final checkpoint for seed {seed}")
        for condition in CONDITIONS:
            scenarios = validation[condition]
            plan.extend({"seed": seed, "condition": condition, "mode": mode, "group": group,
                         "tape_id": scenario.tape_id, "checkpoint_sha256": sha256(ck)}
                        for mode, group, ck in (("periodic", "P_train", p_ck), ("triggered", "P_train", p_ck), ("triggered", "T_train", t_ck))
                        for scenario in scenarios)
            for mode, group, ck in (("periodic", "P_train", p_ck), ("triggered", "P_train", p_ck), ("triggered", "T_train", t_ck)):
                row = evaluate_checkpoint(ck, group, seed, scenarios, config, PREFERENCE, mode)
                row["condition"] = condition
                results.append(row)
    write_json(args.out / "evaluation-plan.json", {"schema": "event-trigger-validation-plan/1.0.0", "parent_count_per_condition": 16,
        "conditions": CONDITIONS, "seed": SEEDS, "preference": PREFERENCE, "configurations": [
            "P_train+periodic", "P_train+T_dispatch", "T_train+T_dispatch"], "plan": plan, "final_test_read": False})
    write_json(args.out / "validation-results.json", {"status": "completed", "episode_count": sum(len(item["episodes"]) for item in results), "results": results, "final_test_read": False})
    write_json(args.out / "evaluation-summary.json", {"status": "completed", "episode_count": sum(len(item["episodes"]) for item in results), "result_rows": len(results), "wall_seconds": time.time(), "results": results})
    print(json.dumps({"status": "completed", "episode_count": sum(len(item["episodes"]) for item in results), "result_rows": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
