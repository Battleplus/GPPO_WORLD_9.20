from __future__ import annotations

from gppo_world.recorder import TransitionRecorder

from test_contracts import make_transition


def test_same_transition_has_byte_identical_recording():
    left = TransitionRecorder()
    right = TransitionRecorder()
    left.append(make_transition())
    right.append(make_transition())
    assert left.canonical_bytes() == right.canonical_bytes()
    assert left.sha256() == right.sha256()


def test_all_behavior_policies_use_same_recorder_contract():
    recorder = TransitionRecorder()
    for index, policy in enumerate(("random_legal", "greedy", "gppo")):
        recorder.append(
            make_transition(
                episode_id=f"episode-{policy}",
                tape_id=f"tape-{index}",
                behavior_policy=policy,
                step=0,
            )
        )
    assert [item.behavior_policy for item in recorder.items] == ["random_legal", "greedy", "gppo"]


def test_non_contiguous_episode_steps_are_rejected():
    recorder = TransitionRecorder()
    recorder.append(make_transition(step=0))
    try:
        recorder.append(make_transition(step=2))
    except ValueError as error:
        assert "contiguous" in str(error)
    else:
        raise AssertionError("non-contiguous step accepted")
