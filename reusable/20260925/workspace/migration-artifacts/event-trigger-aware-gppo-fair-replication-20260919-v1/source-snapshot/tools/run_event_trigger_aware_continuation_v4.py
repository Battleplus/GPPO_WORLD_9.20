"""Continue the fixed P/T matrix from the last committed transaction.

This runner is deliberately separate from the historical v3 controller.  It
requires the already-created SQLite budget database and uses stable run IDs so
an attempt retry cannot obtain a fresh per-run quota.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path
import os
import platform
import sys
import threading
import time
import traceback

import numpy as np
import torch

from gppo_world.budget_executor import PersistentBudget
from gppo_world.joint_gppo import JointTrainConfig
from run_event_trigger_aware_gppo import (
    CHECKPOINT_HASHES, CONDITIONS, PREFERENCE, SEEDS, TAPE_HASH,
    interleave, load_config, load_tapes, run_training, sha256, write_json,
)


GLOBAL_LIMITS = {
    "environment_steps": 49152,
    "policy_optimizer_calls": 384,
    "world_optimizer_calls": 0,
}
RUN_LIMITS = {
    "environment_steps": 8192,
    "policy_optimizer_calls": 64,
    "world_optimizer_calls": 0,
}
LEGACY_RUN_MAP = {
    "formal-v3-controller": "P_train/seed-1101",
    "formal-v3-resume-controller": "P_train/seed-1101",
}


def budget_for(db: Path, attempt_id: str) -> PersistentBudget:
    return PersistentBudget(
        db,
        limits=GLOBAL_LIMITS,
        run_limits=RUN_LIMITS,
        attempt_id=attempt_id,
        require_existing=True,
        legacy_run_map=LEGACY_RUN_MAP,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, required=True)
    parser.add_argument("--resume-ledger", type=Path, default=None)
    parser.add_argument("--resume-group", choices=("P_train", "T_train"), default="P_train")
    parser.add_argument("--resume-seed", type=int, default=1101)
    parser.add_argument("--skip-run", action="append", default=[])
    args = parser.parse_args()
    out = args.out.resolve()
    db = args.budget.resolve()
    resume_checkpoint = args.resume_checkpoint.resolve()
    if out.exists():
        raise SystemExit(f"refusing to reuse continuation output: {out}")
    if not db.is_file():
        raise SystemExit(f"required existing budget database missing: {db}")
    if not resume_checkpoint.is_file():
        raise SystemExit(f"resume checkpoint missing: {resume_checkpoint}")
    if args.resume_ledger is not None and not args.resume_ledger.is_file():
        raise SystemExit(f"resume ledger missing: {args.resume_ledger}")

    out.mkdir(parents=True)
    source_root = Path(__file__).resolve().parents[1]
    config = load_config()
    tapes = load_tapes()
    scenarios = interleave([tapes["train"][condition] for condition in CONDITIONS])
    initial_snapshot = budget_for(db, "formal-v4-controller").snapshot()
    protocol = {
        "schema": "event-trigger-aware-gppo-continuation-v4/1.0.0",
        "source_root": str(source_root),
        "source_runner_sha256": sha256(source_root / "tools" / "run_event_trigger_aware_gppo.py"),
        "continuation_runner_sha256": sha256(Path(__file__).resolve()),
        "budget_db": str(db), "budget_db_sha256_before": sha256(db),
        "resume_checkpoint": str(resume_checkpoint), "resume_checkpoint_sha256": sha256(resume_checkpoint),
        "legacy_run_map": LEGACY_RUN_MAP, "run_limits": RUN_LIMITS, "global_limits": GLOBAL_LIMITS,
        "seeds": SEEDS, "groups": {"P_train": "periodic policy decisions", "T_train": "frozen public T_dispatch decisions"},
        "conditions": CONDITIONS, "preference": PREFERENCE, "environment_steps_each": 8192,
        "policy_optimizer_calls_each": 64, "world_optimizer_calls": 0, "rollout_steps": 128,
        "new_tapes": False, "final_test_read": False, "environment": vars(config),
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
        "started_at": time.time(), "previous_start_at": 1789723787.6418252,
        "previous_time_window_seconds": 8 * 3600, "additional_time_window_seconds": 0,
    }
    write_json(out / "protocol.json", protocol)
    write_json(out / "source-identity.json", {
        "source_runner_sha256": protocol["source_runner_sha256"],
        "continuation_runner_sha256": protocol["continuation_runner_sha256"],
        "checkpoint_hashes": CHECKPOINT_HASHES, "formal_tapes_sha256": TAPE_HASH,
        "budget_db_sha256_before": protocol["budget_db_sha256_before"],
    })
    progress = {"status": "running", "runs": {}, "budget": initial_snapshot}
    write_json(out / "training-progress.json", progress)
    active = {"run": None, "pid": os.getpid(), "updated_at": time.time()}
    stop_event = threading.Event()

    def heartbeat() -> None:
        while not stop_event.wait(5.0):
            active["updated_at"] = time.time()
            write_json(out / "heartbeat.json", {**active, "budget": budget_for(db, "formal-v4-heartbeat").snapshot()})

    thread = threading.Thread(target=heartbeat, name="formal-v4-heartbeat", daemon=True)
    thread.start()
    started = time.perf_counter()
    jobs = [(args.resume_group, args.resume_seed, resume_checkpoint)]
    jobs.extend((group, seed, None) for seed in SEEDS for group in ("P_train", "T_train")
                 if (group, seed) != (args.resume_group, args.resume_seed)
                 and f"{group}/seed-{seed}" not in set(args.skip_run))
    try:
        for group, seed, resume_from in jobs:
            run_key = f"{group}/seed-{seed}"
            run_dir = out / "training" / f"seed-{seed}" / group
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            active.update({"run": run_key, "updated_at": time.time()})
            write_json(out / "heartbeat.json", {**active, "budget": budget_for(db, f"formal-v4-{group}-{seed}").snapshot()})
            write_json(run_dir.parent / f"{group}.start.json", {
                "run": run_key, "run_id": run_key, "attempt_id": f"formal-v4-{group}-{seed}",
                "started_at": time.time(), "pid": os.getpid(),
            })
            try:
                budget = budget_for(db, f"formal-v4-{group}-{seed}")
                run_totals = budget.run_totals(run_key)
                if resume_from is None:
                    resume_counters = {"environment_steps": 0, "policy_optimizer_steps": 0}
                else:
                    resume_payload = torch.load(resume_from, map_location="cpu", weights_only=False)
                    resume_counters = dict(resume_payload.get("counters", {}))
                prior_steps = int(resume_counters.get("environment_steps", 0))
                prior_updates = int(resume_counters.get("policy_optimizer_steps", 0))
                prior_reserved_steps = int(run_totals.get("environment_steps", {}).get("reserved", 0))
                prior_reserved_updates = int(run_totals.get("policy_optimizer_calls", {}).get("reserved", 0))
                target_steps = prior_steps + max(0, RUN_LIMITS["environment_steps"] - prior_reserved_steps)
                target_updates = prior_updates + max(0, RUN_LIMITS["policy_optimizer_calls"] - prior_reserved_updates)
                write_json(run_dir.parent / f"{group}.budget-start.json", {
                    "run_id": run_key, "attempt_id": f"formal-v4-{group}-{seed}",
                    "prior_checkpoint_counters": resume_counters,
                    "prior_run_reserved": run_totals,
                    "target_environment_steps": target_steps,
                    "target_policy_optimizer_calls": target_updates,
                })
                with (run_dir.parent / f"{group}.stdout.log").open("w", encoding="utf-8") as stdout, \
                     (run_dir.parent / f"{group}.stderr.log").open("w", encoding="utf-8") as stderr:
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        result = run_training(
                            group=group, seed=seed, output_dir=run_dir,
                            env_config=config, train_config=JointTrainConfig(seed=seed, rollout_steps=128),
                            scenarios=scenarios, max_steps=target_steps, max_policy_updates=target_updates,
                            wall_seconds=40 * 60, smoke=False, budget=budget,
                            budget_attempt_id=f"formal-v4-{group}-seed-{seed}", resume_from=resume_from,
                            resume_ledger=args.resume_ledger if resume_from is not None else None,
                        )
                progress["runs"][run_key] = result
                write_json(run_dir.parent / f"{group}.exit.json", {
                    "run": run_key, "run_id": run_key, "status": "completed", "exit_code": 0,
                    "ended_at": time.time(),
                })
            except BaseException as exc:
                trace = traceback.format_exc()
                failure = {"run": run_key, "run_id": run_key, "status": "failed", "exit_code": 1,
                           "ended_at": time.time(), "error_type": type(exc).__name__, "error": repr(exc),
                           "traceback": trace, "commit_pointer": str(run_dir / "ledger-commit.json")}
                write_json(run_dir.parent / f"{group}.exit.json", failure)
                (run_dir.parent / f"{group}.traceback.log").write_text(trace, encoding="utf-8")
                progress["status"] = "stopped_on_error"; progress["failure"] = failure
                progress["budget"] = budget_for(db, "formal-v4-failure").snapshot()
                write_json(out / "training-progress.json", progress)
                raise
            progress["budget"] = budget_for(db, "formal-v4-controller").snapshot()
            write_json(out / "training-progress.json", progress)
        progress["status"] = "completed"
        progress["budget"] = budget_for(db, "formal-v4-controller").snapshot()
        write_json(out / "formal-summary.json", {"status": "completed", "runs": progress["runs"],
                                                   "budget": progress["budget"],
                                                   "elapsed_seconds": time.perf_counter() - started})
        write_json(out / "run-status.json", {"status": "completed", "final_test_read": False,
                                              "budget": progress["budget"]})
        return 0
    finally:
        stop_event.set(); thread.join(timeout=2.0)
        active.update({"run": None, "updated_at": time.time()})
        write_json(out / "heartbeat.json", {**active, "budget": budget_for(db, "formal-v4-finalizer").snapshot()})


if __name__ == "__main__":
    raise SystemExit(main())
