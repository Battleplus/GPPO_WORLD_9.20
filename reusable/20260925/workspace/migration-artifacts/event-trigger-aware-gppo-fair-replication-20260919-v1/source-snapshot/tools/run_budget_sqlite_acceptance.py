"""No-training acceptance suite for the SQLite budget ledger."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import sqlite3
import threading

from gppo_world.budget_executor import BudgetStateConflict, PersistentBudget


LIMIT = 10_000
WORKERS = 4
OPS_PER_WORKER = LIMIT // WORKERS


def reserve_confirm_worker(db_path: str, attempt_id: str, count: int) -> int:
    budget = PersistentBudget(db_path, limits={"environment_steps": LIMIT}, attempt_id=attempt_id)
    for _ in range(count):
        token = budget.reserve("environment_steps")
        budget.complete(token)
    return count


def reserve_then_exit_worker(db_path: str, token_path: str) -> None:
    budget = PersistentBudget(db_path, limits={"environment_steps": 2}, attempt_id="interrupted")
    token = budget.reserve("environment_steps")
    Path(token_path).write_text(json.dumps(token), encoding="utf-8")
    raise SystemExit(0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"refusing to reuse acceptance output: {out}")
    out.mkdir(parents=True)
    results: dict[str, object] = {"schema": "budget-sqlite-acceptance/1.0.0", "tests": {}}

    db_path = out / "concurrent.sqlite3"
    budget = PersistentBudget(db_path, limits={"environment_steps": LIMIT}, attempt_id="reader")
    stop = threading.Event()
    read_errors: list[str] = []

    def read_snapshots() -> None:
        while not stop.is_set():
            try:
                budget.snapshot()
            except Exception as exc:
                read_errors.append(repr(exc))
                stop.set()

    reader = threading.Thread(target=read_snapshots, name="sqlite-budget-reader")
    reader.start()
    context = mp.get_context("spawn")
    processes = [context.Process(target=reserve_confirm_worker,
                                 args=(str(db_path), f"worker-{index}", OPS_PER_WORKER))
                 for index in range(WORKERS)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(30 * 60)
    stop.set()
    reader.join(timeout=5)
    state = budget.snapshot()
    results["tests"]["concurrent_10000_reserve_confirm"] = {
        "worker_exit_codes": [process.exitcode for process in processes],
        "snapshot_read_errors": read_errors,
        "stages": state["stages"],
        "history_count": len(state["history"]),
        "pass": all(process.exitcode == 0 for process in processes) and not read_errors
        and state["stages"]["environment_steps"] == {
            "reserved": LIMIT, "verified": LIMIT, "unknown": 0, "pending": 0
        } and len(state["history"]) == 2 * LIMIT,
    }

    idem_db = out / "idempotency.sqlite3"
    idem = PersistentBudget(idem_db, limits={"environment_steps": 2}, attempt_id="idempotency")
    token = idem.reserve("environment_steps")
    idem.complete(token); idem.complete(token)
    duplicate_unknown = idem.reserve("environment_steps")
    idem.unknown(duplicate_unknown, "synthetic interruption")
    idem.unknown(duplicate_unknown, "synthetic interruption repeated")
    conflict = False
    try:
        idem.complete(duplicate_unknown)
    except BudgetStateConflict:
        conflict = True
    invalid = False
    try:
        idem.complete({"reservation_id": "missing-token"})
    except ValueError:
        invalid = True
    idem_state = idem.snapshot()
    results["tests"]["idempotent_and_conflicting_completion"] = {
        "duplicate_complete_no_double_count": idem_state["stages"]["environment_steps"]["verified"] == 1,
        "duplicate_unknown_no_double_count": idem_state["stages"]["environment_steps"]["unknown"] == 1,
        "conflict_rejected": conflict, "invalid_token_rejected": invalid,
        "pass": conflict and invalid and idem_state["stages"]["environment_steps"] == {
            "reserved": 2, "verified": 1, "unknown": 1, "pending": 0
        },
    }

    interruption_db = out / "interruption.sqlite3"
    token_path = out / "interrupted-token.json"
    interrupted = context.Process(target=reserve_then_exit_worker, args=(str(interruption_db), str(token_path)))
    interrupted.start(); interrupted.join(60)
    interrupted_budget = PersistentBudget(interruption_db, limits={"environment_steps": 2}, attempt_id="recovery")
    before = interrupted_budget.snapshot()
    interrupted_token = json.loads(token_path.read_text(encoding="utf-8"))
    interrupted_budget.unknown(interrupted_token, "child exited before external operation confirmation")
    after = interrupted_budget.snapshot()
    results["tests"]["interruption_preserves_pending_then_unknown"] = {
        "child_exit_code": interrupted.exitcode,
        "pending_before": before["stages"]["environment_steps"]["pending"],
        "unknown_after": after["stages"]["environment_steps"]["unknown"],
        "pass": interrupted.exitcode == 0 and before["stages"]["environment_steps"]["pending"] == 1
        and after["stages"]["environment_steps"]["unknown"] == 1,
    }

    repeat_db = out / "repeat.sqlite3"
    repeat = PersistentBudget(repeat_db, limits={"environment_steps": 3}, attempt_id="repeat")
    rt = repeat.reserve("environment_steps"); repeat.complete(rt)
    first_snapshot = repeat.snapshot()
    repeat_again = PersistentBudget(repeat_db, limits={"environment_steps": 3}, attempt_id="repeat-again")
    second_snapshot = repeat_again.snapshot()
    results["tests"]["restart_does_not_recount"] = {
        "before": first_snapshot["stages"], "after": second_snapshot["stages"],
        "history_before": len(first_snapshot["history"]), "history_after": len(second_snapshot["history"]),
        "pass": first_snapshot["stages"] == second_snapshot["stages"]
        and len(first_snapshot["history"]) == len(second_snapshot["history"]),
    }

    locked_db = out / "lock-busy.sqlite3"
    locked = PersistentBudget(locked_db, limits={"environment_steps": 2}, attempt_id="locked")
    raw = sqlite3.connect(locked_db, isolation_level=None)
    raw.execute("BEGIN IMMEDIATE")
    lock_failed_closed = False
    try:
        short = PersistentBudget(locked_db, limits={"environment_steps": 2}, attempt_id="blocked", lock_timeout=0.1)
        try:
            short.reserve("environment_steps")
        except sqlite3.OperationalError:
            lock_failed_closed = short.snapshot()["stages"]["environment_steps"]["reserved"] == 0
    finally:
        raw.rollback(); raw.close()
    results["tests"]["busy_lock_fails_closed"] = {"pass": lock_failed_closed}

    results["all_pass"] = all(bool(test.get("pass")) for test in results["tests"].values())
    (out / "acceptance-result.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if results["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
