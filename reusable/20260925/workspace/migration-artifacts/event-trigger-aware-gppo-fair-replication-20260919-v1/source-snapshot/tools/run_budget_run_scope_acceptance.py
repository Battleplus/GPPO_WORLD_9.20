"""No-training acceptance for stable run-scoped SQLite budgeting."""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
import tempfile
import threading

from gppo_world.budget_executor import BudgetExhausted, PersistentBudget


LIMITS = {"environment_steps": 32, "policy_optimizer_calls": 16, "world_optimizer_calls": 0}
RUN_LIMITS = {"environment_steps": 8, "policy_optimizer_calls": 4, "world_optimizer_calls": 0}


def worker(db_path: str, worker_id: int) -> int:
    budget = PersistentBudget(db_path, limits=LIMITS, run_limits=RUN_LIMITS,
                              attempt_id=f"attempt-{worker_id}", require_existing=True)
    accepted = 0
    for index in range(8):
        try:
            token = budget.reserve("environment_steps", run_id="run-A")
        except BudgetExhausted:
            break
        budget.complete(token)
        accepted += 1
    return accepted


def main(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=False)
    db = out / "scope.sqlite3"
    PersistentBudget(db, limits=LIMITS, run_limits=RUN_LIMITS, attempt_id="bootstrap")
    results: dict[str, object] = {}

    # Attempt changes must not reset a stable run's quota.
    first = PersistentBudget(db, limits=LIMITS, run_limits=RUN_LIMITS, attempt_id="attempt-1", require_existing=True)
    for _ in range(4):
        token = first.reserve("policy_optimizer_calls", run_id="run-B")
        first.complete(token)
    second = PersistentBudget(db, limits=LIMITS, run_limits=RUN_LIMITS, attempt_id="attempt-2", require_existing=True)
    try:
        second.reserve("policy_optimizer_calls", run_id="run-B")
        results["attempt_switch_capped"] = False
    except BudgetExhausted:
        results["attempt_switch_capped"] = True

    # Concurrent writers for one run cannot exceed its per-run cap.
    ctx = mp.get_context("spawn")
    with ctx.Pool(4) as pool:
        accepted = pool.starmap(worker, [(str(db), index) for index in range(4)])
    results["concurrent_accepted"] = accepted
    snapshot_errors: list[str] = []
    stop = threading.Event()

    def reader() -> None:
        reader_budget = PersistentBudget(db, limits=LIMITS, run_limits=RUN_LIMITS,
                                         attempt_id="reader", require_existing=True)
        while not stop.is_set():
            try:
                snap = reader_budget.snapshot()
                for stage, row in snap["stages"].items():
                    if row["reserved"] != row["verified"] + row["unknown"] + row["pending"]:
                        raise AssertionError(stage)
            except BaseException as exc:  # pragma: no cover - diagnostic path
                snapshot_errors.append(repr(exc))
                stop.set()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    stop.set()
    thread.join(timeout=2)
    results["snapshot_errors"] = snapshot_errors

    # A missing database must fail closed for the training entry point.
    try:
        PersistentBudget(out / "missing.sqlite3", limits=LIMITS, run_limits=RUN_LIMITS,
                         attempt_id="missing", require_existing=True)
        results["missing_path_fails_closed"] = False
    except FileNotFoundError:
        results["missing_path_fails_closed"] = True

    snapshot = PersistentBudget(db, limits=LIMITS, run_limits=RUN_LIMITS,
                                attempt_id="final", require_existing=True).snapshot()
    results["final_snapshot"] = snapshot
    results["run_A_environment_verified"] = sum(
        int(row["verified"]) for attempt, values in snapshot["attempts"].items()
        for stage, row in values.items() if row.get("run_id") == "run-A" and stage == "environment_steps"
    )
    results["all_pass"] = bool(
        results["attempt_switch_capped"] and sum(accepted) == 8 and
        not snapshot_errors and results["missing_path_fails_closed"] and
        results["run_A_environment_verified"] == 8
    )
    (out / "acceptance-result.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "final_snapshot"}, indent=2, sort_keys=True))
    return 0 if results["all_pass"] else 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.out.resolve()))
