import pytest

from tools.diagnose_t05_frozen_results import METRICS, paired_differences, reward_components, scenario


def row(tape, value):
    return {"tape_id": tape, **{metric: value for metric in METRICS}}


def test_pairing_uses_tape_identity():
    result = paired_differences([row("a", 5), row("b", 9)], [row("b", 7), row("a", 1)])
    assert result["n"] == 2
    assert result["candidate_minus_gppo"]["episode_return"] == 3


@pytest.mark.parametrize("candidate,baseline", [
    ([row("a", 1), row("a", 2)], [row("a", 3)]),
    ([row("a", 1)], [row("b", 2)]),
    ([], []),
])
def test_rejects_invalid_pairs(candidate, baseline):
    with pytest.raises(ValueError):
        paired_differences(candidate, baseline)


def test_reward_conservation_and_failure():
    components = dict(uncovered=5, distance=-1, load_gap=1, switches=-0.25, recovery_delay=-0.5)
    trace = {"decisions": [{"active_events_before": ["event1", "event2"], "reward": 4.25,
                            "reward_trace": {"reward_components": components}}],
             "episode_return_check": 4.25, "episode": {"episode_return": 4.25}}
    assert reward_components(trace) == components
    trace["episode_return_check"] = 9
    with pytest.raises(ValueError):
        reward_components(trace)


def test_unattributed_decision_is_not_event_return():
    components = dict(uncovered=5, distance=-1, load_gap=1, switches=-0.25, recovery_delay=-0.5)
    trace = {"decisions": [{"active_events_before": [], "reward": 4.25,
                            "reward_trace": {"reward_components": components}}],
             "episode_return_check": 4.25, "episode": {"episode_return": 0}}
    assert sum(reward_components(trace).values()) == 0


def test_frozen_scenario_taxonomy():
    assert scenario("test-unseen-0000-abcd") == "unseen"
    with pytest.raises(ValueError):
        scenario("train-single-0000-abcd")
