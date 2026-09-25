from __future__ import annotations

import json
from pathlib import Path
import threading

import numpy as np
import torch

from gppo_world.budget_executor import BudgetExhausted, PersistentBudget
from gppo_world.event_trigger_semimarkov import DecisionWindow, decision_level_gae, discounted_window
from gppo_world.joint_training import _TrainingLedger, verify_commit_pointer


def test_budget_is_shared_across_attempts_and_never_refunds(tmp_path: Path):
    path = tmp_path / "budget.json"
    first = PersistentBudget(path, limits={"environment_steps": 3}, attempt_id="a")
    token = first.reserve("environment_steps")
    first.complete(token)
    second = PersistentBudget(path, limits={"environment_steps": 3}, attempt_id="b")
    failed = second.reserve("environment_steps")
    second.unknown(failed, "injected interruption")
    third = PersistentBudget(path, limits={"environment_steps": 3}, attempt_id="c")
    third.reserve("environment_steps")
    try:
        third.reserve("environment_steps")
    except BudgetExhausted:
        pass
    else:
        raise AssertionError("reserved/unknown budget was incorrectly refunded")
    state = third.snapshot()
    assert state["stages"]["environment_steps"] == {"reserved": 3, "verified": 1, "unknown": 1, "pending": 1}


def test_budget_snapshot_is_serialized_with_atomic_replace(tmp_path: Path):
    path = tmp_path / "budget.json"
    budget = PersistentBudget(path, limits={"environment_steps": 200}, attempt_id="writer")
    stop = threading.Event()
    errors: list[BaseException] = []

    def heartbeat():
        while not stop.is_set():
            try:
                budget.snapshot()
            except BaseException as exc:
                errors.append(exc)
                stop.set()

    thread = threading.Thread(target=heartbeat)
    thread.start()
    try:
        for _ in range(100):
            token = budget.reserve("environment_steps")
            budget.complete(token)
    finally:
        stop.set()
        thread.join(timeout=2)
    assert not errors
    assert budget.snapshot()["stages"]["environment_steps"]["verified"] == 100


def test_boundary_window_bootstrap_cuts_gae_and_hand_calculation():
    reward_a, k_a, discount_a = discounted_window([[1.0, -2.0], [0.5, -1.0]], 0.9)
    reward_b, k_b, discount_b = discounted_window([[2.0, -1.0]], 0.9)
    windows = [
        DecisionWindow(reward_a, k_a, discount_a, np.zeros(2), np.ones(2), False, False, True),
        DecisionWindow(reward_b, k_b, discount_b, np.zeros(2), np.zeros(2), False),
    ]
    adv, ret = decision_level_gae(windows, 0.9, 0.95)
    expected_a = reward_a + discount_a * np.ones(2)
    expected_b = reward_b
    np.testing.assert_allclose(adv[0], expected_a)
    np.testing.assert_allclose(adv[1], expected_b)
    np.testing.assert_allclose(ret[0], expected_a)
    np.testing.assert_allclose(ret[1], expected_b)


def test_commit_pointer_does_not_promote_larger_uncommitted_checkpoint(tmp_path: Path):
    ledger = _TrainingLedger(tmp_path / "training-ledger.jsonl", run_id="fixture")
    ledger.append({"step": 0, "record_type": "decision"})
    staged = ledger.flush_rollout()
    old_txn = tmp_path / "transactions" / "txn-step-00000001-policy-000001-world-000000.pt"
    old_txn.parent.mkdir()
    old_payload = {
        "run_id": "fixture", "persistence": {"transaction_id": "old"},
        "counters": {"environment_steps": 1, "policy_optimizer_steps": 1, "world_optimizer_steps": 0},
    }
    torch.save(old_payload, old_txn)
    ledger.publish_transaction(
        checkpoint_path=old_txn,
        counters=old_payload["counters"],
        transaction_id="old",
        ledger_range=staged,
    )
    ledger.append({"step": 1, "record_type": "decision"})
    ledger.flush_rollout()
    newer = tmp_path / "transactions" / "txn-step-00000002-policy-000002-world-000000.pt"
    torch.save({
        "run_id": "fixture", "persistence": {"transaction_id": "new"},
        "counters": {"environment_steps": 2, "policy_optimizer_steps": 2, "world_optimizer_steps": 0},
    }, newer)
    checked = verify_commit_pointer(tmp_path)
    assert checked["transaction_id"] == "old"
    assert checked["last_committed_step"] == 0
    assert checked["ledger_tail_bytes_uncommitted"] > 0
    ledger.close()
