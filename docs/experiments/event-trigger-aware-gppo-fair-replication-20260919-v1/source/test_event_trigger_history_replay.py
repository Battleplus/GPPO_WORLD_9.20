import json

import numpy as np
import pytest
import torch

from tools.run_event_trigger_aware_gppo import (
    _select_replay_hidden,
    load_saved_decision_types,
    save_runtime_checkpoint,
)


class _CheckpointStub:
    def state_dict(self):
        return {"weight": torch.tensor([1.0])}

    def public_snapshot_digest(self):
        return (1, (0.0,), (True,))


def _replay():
    return {
        "next_policy_hidden": torch.tensor([1.0]),
        "by_action": {
            7: {"hidden": torch.tensor([7.0])},
            18: {"hidden": torch.tensor([18.0])},
        },
    }


def test_continuation_uses_public_continuation_contract_when_new_mask_is_false():
    obs = {"mask": [False] * 25, "continuation_actions": [7, 18]}
    policy_hidden, world_hidden = _select_replay_hidden(_replay(), obs, 7, False)
    assert policy_hidden.item() == 1.0
    assert world_hidden.item() == 7.0


def test_new_submission_still_requires_current_command_mask():
    obs = {"mask": [False] * 25, "continuation_actions": [7]}
    with pytest.raises(RuntimeError, match="new action"):
        _select_replay_hidden(_replay(), obs, 7, True)


def test_invalid_continuation_is_rejected():
    obs = {"mask": [False] * 25, "continuation_actions": [18]}
    with pytest.raises(RuntimeError, match="continuation action"):
        _select_replay_hidden(_replay(), obs, 7, False)


def test_saved_decision_types_are_explicit(tmp_path):
    path = tmp_path / "ledger.jsonl"
    rows = [
        {"record_type": "decision_or_continuation", "group": "T_train", "seed": 1101,
         "step": 0, "actor_decision": True, "command_submitted": True},
        {"record_type": "decision_or_continuation", "group": "T_train", "seed": 1101,
         "step": 1, "actor_decision": False, "command_submitted": False},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert load_saved_decision_types(path, 2, "T_train", 1101) == [True, False]


def test_episode_boundary_checkpoint_preserves_saved_decision_types(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    rows = [
        {"record_type": "decision_or_continuation", "group": "T_train", "seed": 1101,
         "step": 0, "episode_index": 0, "scenario_id": "episode-0",
         "actor_decision": True, "command_submitted": True},
        {"record_type": "decision_or_continuation", "group": "T_train", "seed": 1101,
         "step": 1, "episode_index": 1, "scenario_id": "episode-1",
         "actor_decision": False, "command_submitted": False},
    ]
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert load_saved_decision_types(
        ledger, 1, "T_train", 1101, episode_index=1, scenario_id="episode-1"
    ) == [False]

    policy = _CheckpointStub()
    world = _CheckpointStub()
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter])
    checkpoint = tmp_path / "boundary.pt"
    save_runtime_checkpoint(
        checkpoint, run_id="test-run", group="T_train", policy=policy, world=world,
        optimizer=optimizer, env=_CheckpointStub(), obs={"mask": [True]},
        policy_hidden=None, world_hidden=None, pending=None,
        preference_rng=np.random.default_rng(7), episode_preference=torch.tensor([0.8, 0.2]),
        counters={"last_committed_step": 1}, identity={"transaction_id": "txn-1"},
        runtime_state={
            "episode_index": 1,
            "history_observations": [{"mask": [False], "continuation_actions": [7]}],
            "history_actions": [7],
            "history_submits": [False],
        },
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["runtime"]["episode_index"] == 1
    assert payload["runtime"]["history_actions"] == [7]
    assert payload["runtime"]["history_submits"] == [False]
