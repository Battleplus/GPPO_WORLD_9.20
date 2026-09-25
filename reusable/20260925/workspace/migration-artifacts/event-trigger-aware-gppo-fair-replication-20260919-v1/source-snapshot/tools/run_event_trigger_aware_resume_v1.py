"""One controlled resume of formal-v3, using the original persistent budget."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
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
    CHECKPOINT_HASHES, CONDITIONS, PREFERENCE, SEEDS,
    interleave, load_config, load_tapes, run_training, sha256, write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--resume-target-steps", type=int, default=8177)
    parser.add_argument("--resume-target-updates", type=int, default=63)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"refusing to reuse resume output: {out}")
    out.mkdir(parents=True)
    source_root = Path(__file__).resolve().parents[1]
    config = load_config()
    tapes = load_tapes()
    scenarios = interleave([tapes["train"][condition] for condition in CONDITIONS])
    budget = PersistentBudget(
        args.budget.resolve(),
        limits={"environment_steps": 49152, "policy_optimizer_calls": 384, "world_optimizer_calls": 0},
        attempt_id="formal-v3-resume-controller",
    )
    previous = args.resume.resolve()
    resume_payload = torch.load(previous, map_location="cpu", weights_only=False)
    resume_counters = dict(resume_payload.get("counters", {}))
    resume_commit_path = previous.parent / "ledger-commit.json"
    resume_commit = json.loads(resume_commit_path.read_text(encoding="utf-8")) if resume_commit_path.is_file() else {}
    resume_ledger_path = previous.parent / "training-ledger.jsonl"
    resume_committed_bytes = int(resume_commit.get("ledger", {}).get("ledger_bytes", 0))
    resume_total_bytes = resume_ledger_path.stat().st_size if resume_ledger_path.is_file() else resume_committed_bytes
    protocol = {
        "schema": "event-trigger-aware-gppo-formal-v3-resume/1.0.0",
        "source_root": str(source_root),
        "source_runner_sha256": sha256(source_root / "tools" / "run_event_trigger_aware_gppo.py"),
        "controller_sha256": sha256(Path(__file__).resolve()),
        "checkpoint_hashes": CHECKPOINT_HASHES,
        "formal_tapes_sha256": "5d1cfaed3683e3e1c512f4ff67d1f93566dae2b418942b67df3f1c81cb5eaea7",
        "seeds": SEEDS,
        "groups": {"P_train": "periodic policy decisions", "T_train": "frozen public T_dispatch decisions"},
        "conditions": CONDITIONS,
        "preference": PREFERENCE,
        "original_budget_limits": {"environment_steps": 49152, "policy_optimizer_calls": 384, "world_optimizer_calls": 0},
        "resume_source": str(previous),
        "resume_source_commit": {"counters": resume_counters,
                                 "transaction_id": resume_commit.get("transaction_id"),
                                 "checkpoint_sha256": sha256(previous),
                                 "committed_ledger_bytes": resume_committed_bytes,
                                 "ledger_total_bytes": resume_total_bytes,
                                 "uncommitted_tail_bytes": max(0, resume_total_bytes - resume_committed_bytes),
                                 "uncommitted_tail_promoted": False},
        "first_resume_logical_target": {"environment_steps": args.resume_target_steps,
                                         "policy_optimizer_steps": args.resume_target_updates,
                                         "additional_budget_steps": max(0, args.resume_target_steps - int(resume_counters.get("environment_steps", 0))),
                                         "additional_budget_updates": max(0, args.resume_target_updates - int(resume_counters.get("policy_optimizer_steps", 0)))},
        "new_runs_target": {"environment_steps": 8192, "policy_optimizer_steps": 64},
        "world_parameters_frozen": True, "new_tapes": False, "final_test_read": False,
        "environment": vars(config), "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__, "started_at": time.time(),
    }
    write_json(out / "protocol.json", protocol)
    write_json(out / "source-identity.json", {"source_runner_sha256": protocol["source_runner_sha256"],
                                               "controller_sha256": protocol["controller_sha256"],
                                               "checkpoint_hashes": CHECKPOINT_HASHES,
                                               "formal_tapes_sha256": protocol["formal_tapes_sha256"],
                                               "shared_budget": str(args.budget.resolve())})
    progress = {"status": "running", "runs": {}, "budget": budget.snapshot()}
    write_json(out / "training-progress.json", progress)
    active = {"run": None, "pid": None, "updated_at": time.time()}
    stop_event = threading.Event()

    def heartbeat() -> None:
        while not stop_event.wait(5.0):
            active["updated_at"] = time.time()
            write_json(out / "heartbeat.json", {**active, "budget": budget.snapshot()})

    hb = threading.Thread(target=heartbeat, name="formal-v3-resume-heartbeat", daemon=True)
    hb.start()
    run_specs = [("P_train", 1101, previous, args.resume_target_steps, args.resume_target_updates)]
    for seed in SEEDS:
        for group in ("P_train", "T_train"):
            if group == "P_train" and seed == 1101:
                continue
            run_specs.append((group, seed, None, 8192, 64))

    started = time.perf_counter()
    try:
        for group, seed, resume_from, max_steps, max_updates in run_specs:
            run_key = f"{group}/seed-{seed}"
            run_dir = out / "training" / f"seed-{seed}" / group
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            active.update({"run": run_key, "pid": __import__("os").getpid(), "updated_at": time.time()})
            write_json(out / "heartbeat.json", {**active, "budget": budget.snapshot()})
            write_json(run_dir.parent / f"{group}.start.json", {
                "run": run_key, "started_at": time.time(), "pid": active["pid"],
                "resume_from": None if resume_from is None else str(resume_from),
                "max_steps": max_steps, "max_policy_updates": max_updates,
            })
            try:
                with (run_dir.parent / f"{group}.stdout.log").open("w", encoding="utf-8") as stdout, \
                     (run_dir.parent / f"{group}.stderr.log").open("w", encoding="utf-8") as stderr:
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        result = run_training(
                            group=group, seed=seed, output_dir=run_dir,
                            env_config=config, train_config=JointTrainConfig(seed=seed, rollout_steps=128),
                            scenarios=scenarios, max_steps=max_steps, max_policy_updates=max_updates,
                            wall_seconds=40 * 60, smoke=False, budget=budget,
                            budget_attempt_id=f"formal-v3-resume-{group}-seed-{seed}",
                            resume_from=resume_from,
                        )
                progress["runs"][run_key] = result
                write_json(run_dir.parent / f"{group}.exit.json", {
                    "run": run_key, "status": "completed", "exit_code": 0,
                    "ended_at": time.time(), "result": result,
                })
            except BaseException as exc:
                trace = traceback.format_exc()
                failure = {
                    "run": run_key, "status": "failed", "exit_code": 1,
                    "ended_at": time.time(), "error_type": type(exc).__name__,
                    "error": repr(exc), "traceback": trace,
                    "last_progress": str(run_dir / "run-status.json") if (run_dir / "run-status.json").is_file() else None,
                    "commit_pointer": str(run_dir / "ledger-commit.json") if (run_dir / "ledger-commit.json").is_file() else None,
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
        progress["budget"] = budget.snapshot()
        write_json(out / "training-progress.json", progress)
        write_json(out / "formal-resume-summary.json", {
            "status": "completed", "runs": progress["runs"],
            "budget": budget.snapshot(), "elapsed_seconds": time.perf_counter() - started,
        })
        return 0
    finally:
        stop_event.set()
        hb.join(timeout=2.0)


if __name__ == "__main__":
    raise SystemExit(main())
