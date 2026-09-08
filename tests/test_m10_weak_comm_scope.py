from tools.summarize_m10_weak_comm_scope import _strict_event_class, summarize


def _summary(**overrides):
    value = {
        "interrupted": True,
        "legal_event_knowledge": {"time": 3.0},
        "first_public_candidate_time_after_knowledge": 3.0,
        "recovery_completion_success": False,
        "recovery_service_success": False,
        "failure_reasons": [],
    }
    value.update(overrides)
    return value


def test_scope_class_keeps_non_interrupted_rows_out_of_recovery_denominator():
    assert _strict_event_class(_summary(interrupted=False)) == "not_an_interrupted_recovery_event"


def test_scope_class_does_not_call_missing_public_candidate_a_policy_failure():
    assert _strict_event_class(
        _summary(first_public_candidate_time_after_knowledge=None)
    ) == "information_missing_or_stale"


def test_scope_class_separates_deadline_miss_from_information_missing():
    assert _strict_event_class(
        _summary(
            recovery_service_success=True,
            failure_reasons=["not_completed_before_deadline"],
        )
    ) == "time_or_physical_budget_insufficient_after_knowledge"


def test_summary_has_event_denominator_and_preserves_four_way_counts():
    audit = {
        "format": "test",
        "scope": {},
        "full_pool": {
            "rows": [
                {"split": "train", "tape_id": "a", "category": "both_success", "reference": _summary(recovery_completion_success=True, recovery_service_success=True)},
                {"split": "train", "tape_id": "b", "category": "both_success", "reference": _summary(interrupted=False, recovery_completion_success=True, recovery_service_success=True)},
            ]
        },
        "boundary_factor_scan": {"rows": []},
    }
    result = summarize(audit)
    assert result["full_pool"]["four_way_episode_counts"] == {"both_success": 2}
    assert result["full_pool"]["interrupted_event_metrics"]["interrupted_events"] == 1
    assert result["full_pool"]["interrupted_event_metrics"]["deadline_completed"] == 1
