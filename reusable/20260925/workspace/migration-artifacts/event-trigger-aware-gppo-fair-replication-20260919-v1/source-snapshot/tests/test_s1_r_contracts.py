from gppo_world.s1_r_contracts import (
    EnergyLedger,
    classify_episode,
    count_reassignments,
    low_confidence_confirmed,
)


def test_normal_termination_without_task_completion_is_incomplete():
    out = classify_episode(terminated=True, truncated=False, task_completed=False)
    assert out.episode_ended and not out.task_completed and out.task_failed
    assert out.end_reason == "terminated_incomplete"


def test_deadline_timeout_is_distinct_from_episode_timeout():
    out = classify_episode(terminated=False, truncated=True, task_completed=False, deadline_exceeded=True)
    assert out.deadline_exceeded and out.end_reason == "deadline_timeout"
    assert not out.task_failed


def test_stale_retry_can_switch_uav_and_charges_only_new_endpoint():
    ledger = EnergyLedger({0: 1.0, 1: 1.0})
    first = ledger.submit(submission_id="d0", action=2, uid=0, legal=True, noop_action=16)
    stale = ledger.submit(submission_id="d0", action=2, uid=0, legal=True, noop_action=16, stale=True)
    retry = ledger.submit(submission_id="d1", action=3, uid=1, legal=True, noop_action=16)
    assert first.charged_uid == 0 and first.energy_after == 0.0
    assert stale.status == "stale_rejected" and stale.charged_uid is None
    assert retry.charged_uid == 1 and ledger.energy == {0: 0.0, 1: 0.0}


def test_low_energy_retry_rechecks_energy():
    ledger = EnergyLedger({0: 0.0})
    result = ledger.submit(submission_id="retry", action=2, uid=0, legal=True, noop_action=16)
    assert result.status == "energy_rejected" and result.energy_after == 0.0


def test_duplicate_submission_is_not_charged_twice():
    ledger = EnergyLedger({0: 2.0})
    first = ledger.submit(submission_id="same", action=2, uid=0, legal=True, noop_action=16)
    duplicate = ledger.submit(submission_id="same", action=2, uid=0, legal=True, noop_action=16)
    assert first.status == "accepted" and duplicate.status == "duplicate_rejected"
    assert ledger.energy[0] == 1.0


def test_reassignment_excludes_initial_assignment():
    initial = {0: 0, 1: 1}
    before = {0: 0, 1: 1}
    after = {0: 1, 1: 1}
    assert count_reassignments(before, after, initial) == 0
    assert count_reassignments({0: 1, 1: 1}, {0: 0, 1: 1}, initial) == 1


def test_low_confidence_payload_without_confirmation_does_not_pass():
    assert not low_confidence_confirmed(
        event_id="weak", confidence=0.55, confirmed_ids=set(),
        occurred_at=4.0, observed_at=4.5, decision_time=5.0,
    )
    assert low_confidence_confirmed(
        event_id="weak", confidence=0.55, confirmed_ids={"weak"},
        occurred_at=4.0, observed_at=4.5, decision_time=5.0,
    )

