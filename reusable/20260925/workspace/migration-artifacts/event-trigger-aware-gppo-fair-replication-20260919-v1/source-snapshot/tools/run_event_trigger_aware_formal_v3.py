"""Execute the fixed six-run P/T event-trigger comparison under one budget."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import copy
import json
from pathlib import Path
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"refusing to reuse formal output: {out}")
    out.mkdir(parents=True)
    source_root = Path(__file__).resolve().parents[1]
    source_hash = sha256(source_root / "tools" / "run_event_trigger_aware_gppo.py")
    config = load_config()
    tapes = load_tapes()
    scenarios = interleave([tapes["train"][condition] for condition in CONDITIONS])
    budget = PersistentBudget(
        out / "persistent-budget.json",
        limits={"environment_steps": 49152, "policy_optimizer_calls": 384, "world_optimizer_calls": 0},
        attempt_id="formal-v3-controller",
    )
    protocol = {
        "schema": "event-trigger-aware-gppo-formal-v3/1.0.0",
        "source_root": str(source_root), "source_runner_sha256": source_hash,
        "checkpoint_hashes": CHECKPOINT_HASHES, "formal_tapes_sha256": TAPE_HASH,
        "seeds": SEEDS, "groups": {"P_train": "periodic policy decisions", "T_train": "frozen public T_dispatch decisions"},
        "conditions": CONDITIONS, "preference": PREFERENCE,
        "environment_steps_each": 8192, "policy_optimizer_calls_each": 64,
        "total_environment_steps": 49152, "total_policy_optimizer_calls": 384,
        "world_optimizer_calls": 0, "rollout_steps": 128,
        "world_parameters_frozen": True, "new_tapes": False, "final_test_read": False,
        "environment": vars(config), "python": sys.version,
        "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
        "started_at": time.time(),
    }
    write_json(out / "protocol.json", protocol)
    write_json(out / "source-identity.json", {"source_runner_sha256": source_hash,
                                                  "checkpoint_hashes": CHECKPOINT_HASHES,
                                                  "formal_tapes_sha256": TAPE_HASH})
    progress = {"status": "running", "runs": {}, "budget": budget.snapshot()}
    write_json(out / "training-progress.json", progress)
    active = {"run": None, "pid": None, "updated_at": time.time()}
    stop_event = threading.Event()

    def heartbeat() -> None:
        while not stop_event.wait(5.0):
            active["updated_at"] = time.time()
            write_json(out / "heartbeat.json", {**active, "budget": budget.snapshot()})

    heartbeat_thread = threading.Thread(target=heartbeat, name="formal-v3-heartbeat", daemon=True)
    heartbeat_thread.start()
    started = time.perf_counter()
    try:
        for seed in SEEDS:
            for group in ("P_train", "T_train"):
                run_key = f"{group}/seed-{seed}"
                run_dir = out / "training" / f"seed-{seed}" / group
                run_dir.parent.mkdir(parents=True, exist_ok=True)
                active.update({"run": run_key, "pid": __import__("os").getpid(), "updated_at": time.time()})
                write_json(out / "heartbeat.json", {**active, "budget": budget.snapshot()})
                write_json(run_dir.parent / f"{group}.start.json", {"run": run_key, "started_at": time.time(), "pid": active["pid"]})
                try:
                    with (run_dir.parent / f"{group}.stdout.log").open("w", encoding="utf-8") as stdout, \
                         (run_dir.parent / f"{group}.stderr.log").open("w", encoding="utf-8") as stderr:
                        with redirect_stdout(stdout), redirect_stderr(stderr):
                            result = run_training(
                                group=group, seed=seed, output_dir=run_dir,
                                env_config=config, train_config=JointTrainConfig(seed=seed, rollout_steps=128),
                                scenarios=scenarios, max_steps=8192, max_policy_updates=64,
                                wall_seconds=40 * 60, smoke=False, budget=budget,
                                budget_attempt_id=f"formal-v3-{group}-seed-{seed}",
                            )
                    progress["runs"][run_key] = result
                    write_json(run_dir.parent / f"{group}.exit.json", {"run": run_key, "status": "completed", "exit_code": 0, "ended_at": time.time()})
                except BaseException as exc:
                    trace = traceback.format_exc()
                    failure = {
                        "run": run_key,
                        "status": "failed",
                        "exit_code": 1,
                        "ended_at": time.time(),
                        "error_type": type(exc).__name__,
                        "error": repr(exc),
                        "traceback": trace,
                        "last_progress": run_dir / "run-status.json" if (run_dir / "run-status.json").is_file() else None,
                        "commit_pointer": run_dir / "ledger-commit.json" if (run_dir / "ledger-commit.json").is_file() else None,
                    }
                    write_json(run_dir.parent / f"{group}.exit.json", failure)
                    (run_dir.parent / f"{group}.traceback.log").write_text(trace, encoding="utf-8")
                    progress["status"] = "stopped_on_error"
                    progress["failure"] = failure
                    progress["budget"] = budget.snapshot()
                    write_json(out / "training-progress.json", progress)
                    raise
                progress["budget"] = budget.snapshot()
                write_json(out / "training-progress.json", progress)
        progress["status"] = "completed"
        write_json(out / "formal-summary.json", {"status": "completed", "runs": progress["runs"],
                                                   "budget": budget.snapshot(),
                                                   "elapsed_seconds": time.perf_counter() - started})
        write_json(out / "run-status.json", {"status": "completed", "final_test_read": False,
                                              "budget": budget.snapshot()})
        print(json.dumps({"status": "completed", "runs": len(progress["runs"]),
                          "budget": budget.snapshot()}, indent=2))
        return 0
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=2.0)
        active.update({"run": None, "updated_at": time.time()})
        write_json(out / "heartbeat.json", {**active, "budget": budget.snapshot()})


if __name__ == "__main__":
    raise SystemExit(main())
