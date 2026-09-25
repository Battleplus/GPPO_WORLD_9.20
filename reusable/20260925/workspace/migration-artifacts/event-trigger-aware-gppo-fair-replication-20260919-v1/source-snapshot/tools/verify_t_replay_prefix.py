"""Read-only verification of the failed T_train/1101 history prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_event_trigger_aware_gppo import (  # noqa: E402
    _select_replay_hidden,
    checkpoint_path,
    load_config,
    load_saved_decision_types,
    load_wd,
    probe,
    rebuild_hidden_from_history,
    sha256,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    runtime = payload["runtime"]
    observations = runtime.get("history_observations", [])
    actions = [int(value) for value in runtime.get("history_actions", [])]
    episode_index = int(runtime.get("episode_index", 0))
    scenario_id = runtime["environment"].scenario.tape_id
    submits = load_saved_decision_types(
        args.ledger, len(actions), "T_train", 1101,
        episode_index=episode_index, scenario_id=scenario_id,
    )
    rows = []
    invalid_new_mask_valid_continuation = []
    for step, (obs, action, submit) in enumerate(zip(observations, actions, submits)):
        mask = np.asarray(obs["mask"], dtype=np.bool_)
        continuation = {int(value) for value in obs.get("continuation_actions", ())}
        row = {
            "step": step, "action": action, "actor_decision": bool(submit),
            "submit_command": bool(submit),
            "new_command_mask_legal": bool(0 <= action < len(mask) and mask[action]),
            "continuation_legal": action in continuation,
            "continuation_actions": sorted(continuation),
            "feedback": None,
        }
        if not submit and not row["new_command_mask_legal"] and row["continuation_legal"]:
            invalid_new_mask_valid_continuation.append(step)
        rows.append(row)

    config = load_config()
    policy, world, _ = load_wd(1101, config, torch.device("cpu"))
    preference = runtime["episode_preference"]
    rebuilt_policy, rebuilt_world, replay_calls = rebuild_hidden_from_history(
        policy, world, observations, actions, submits, preference, torch.device("cpu")
    )
    stored_policy = runtime.get("policy_hidden")
    stored_world = runtime.get("world_hidden")
    policy_delta = None if stored_policy is None else float((rebuilt_policy - stored_policy).abs().max())
    world_delta = None if stored_world is None else float((rebuilt_world - stored_world).abs().max())
    report = {
        "status": "passed",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "ledger": str(args.ledger),
        "ledger_sha256": sha256(args.ledger),
        "environment_steps": 0,
        "optimizer_calls": 0,
        "prefix_steps": len(actions),
        "episode_index": episode_index,
        "scenario_id": scenario_id,
        "decision_type_rows": len(submits),
        "invalid_new_mask_valid_continuation_steps": invalid_new_mask_valid_continuation,
        "replay_forward_calls": replay_calls,
        "stored_vs_rebuilt_policy_hidden_max_abs": policy_delta,
        "stored_vs_rebuilt_world_hidden_max_abs": world_delta,
        "hidden_tolerance": 1e-6,
        "discrete_contract_exact": all(
            row["continuation_legal"] if not row["submit_command"] else row["new_command_mask_legal"]
            for row in rows
        ),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "prefix_steps", "invalid_new_mask_valid_continuation_steps",
        "stored_vs_rebuilt_policy_hidden_max_abs", "stored_vs_rebuilt_world_hidden_max_abs",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
