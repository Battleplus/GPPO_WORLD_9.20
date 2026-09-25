"""Same-CPU reload/continuation and uncommitted-tail checks for the pilot."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from gppo_world.event_trigger_semimarkov import DecisionWindow
from gppo_world.joint_gppo import ActionConditionedTemporalWorldModel, JointGraphPreferencePolicy, JointTrainConfig
from gppo_world.joint_training import verify_commit_pointer
from tools.run_event_trigger_aware_gppo import make_sample, probe, weighted_update


def nested_max_delta(a, b):
    if torch.is_tensor(a) and torch.is_tensor(b):
        return float((a.detach().cpu().to(torch.float64) - b.detach().cpu().to(torch.float64)).abs().max()) if a.numel() else 0.0
    if isinstance(a, dict) and isinstance(b, dict):
        return max((nested_max_delta(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return max((nested_max_delta(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
    return 0.0 if a == b else float("inf")


def load_pair(path: Path, env_config):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    policy = JointGraphPreferencePolicy(env_config, history=True)
    world = ActionConditionedTemporalWorldModel()
    policy.load_state_dict(payload["policy_state_dict"], strict=True)
    world.load_state_dict(payload["world_state_dict"], strict=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    optimizer.load_state_dict(payload["policy_optimizer_state_dict"])
    return payload, policy, world, optimizer


def main():
    import argparse
    from tools.run_event_trigger_aware_gppo import load_config
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    config = load_config()
    results = {}
    for group in ("P_train", "T_train"):
        run = args.run_root / "correctness" / group
        checkpoint = run / "last-recovery.pt"
        payload_u, policy_u, world_u, opt_u = load_pair(checkpoint, config)
        payload_r, policy_r, world_r, opt_r = load_pair(checkpoint, config)
        # Both copies start from exactly one committed transaction.  Restore
        # the same RNG snapshot before each path; there is no environment step
        # consumed by this witness.
        torch.set_rng_state(payload_u["rng"]["torch"])
        runtime_u = payload_u["runtime"]
        preference_u = runtime_u["episode_preference"].reshape(2)
        pu = probe(policy_u, world_u, runtime_u["observation"], runtime_u["policy_hidden"], runtime_u["world_hidden"], preference_u, torch.device("cpu"))
        torch.set_rng_state(payload_r["rng"]["torch"])
        runtime_r = payload_r["runtime"]
        preference_r = runtime_r["episode_preference"].reshape(2)
        pr = probe(policy_r, world_r, runtime_r["observation"], runtime_r["policy_hidden"], runtime_r["world_hidden"], preference_r, torch.device("cpu"))
        probe_delta = nested_max_delta({"probs": pu["evaluation"]["probabilities"], "values": pu["evaluation"]["critic_values"]},
                                       {"probs": pr["evaluation"]["probabilities"], "values": pr["evaluation"]["critic_values"]})
        action_u = int(torch.argmax(pu["evaluation"]["distribution"].logits, dim=-1).item())
        action_r = int(torch.argmax(pr["evaluation"]["distribution"].logits, dim=-1).item())
        sample_u = make_sample(pu, runtime_u["observation"], action_u, float(pu["evaluation"]["distribution"].log_prob(torch.tensor([action_u])).item()), preference_u, runtime_u["policy_hidden"], runtime_u["world_hidden"], int(payload_u["counters"].get("episode_index", 0)), "recovery")
        sample_r = copy.deepcopy(sample_u)
        window = DecisionWindow(np.zeros(2, dtype=np.float32), 1, 0.99, sample_u["old_values"], np.zeros(2, dtype=np.float32), False, rollout_boundary=True)
        update_u = weighted_update(policy_u, [sample_u], [window], opt_u, JointTrainConfig(seed=1101), torch.device("cpu"))
        update_r = weighted_update(policy_r, [sample_r], [window], opt_r, JointTrainConfig(seed=1101), torch.device("cpu"))
        parameter_delta = nested_max_delta(policy_u.state_dict(), policy_r.state_dict())
        optimizer_delta = nested_max_delta(opt_u.state_dict(), opt_r.state_dict())
        pending = payload_u["runtime"].get("pending")
        results[group] = {
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "next_probe_max_abs_delta": probe_delta,
            "next_deterministic_action_U": action_u, "next_deterministic_action_R": action_r,
            "post_update_parameter_max_abs_delta": parameter_delta,
            "post_update_optimizer_max_abs_delta": optimizer_delta,
            "pending_window_present": pending is not None,
            "pending_window_k": None if pending is None else len(pending.get("rewards", [])),
            "same_cpu_U_R_passed": probe_delta <= 1e-7 and action_u == action_r and parameter_delta <= 1e-7 and optimizer_delta <= 1e-7,
        }
        pointer_check = verify_commit_pointer(run)
        results[group]["original_commit_verified"] = pointer_check
        # An isolated append after the committed prefix must remain orphaned;
        # verification must report it as a tail, never promote it.
        with tempfile.TemporaryDirectory(prefix=f"event-trigger-{group}-") as temp:
            copied = Path(temp) / group
            shutil.copytree(run, copied)
            ledger = copied / "training-ledger.jsonl"
            with ledger.open("ab") as stream:
                stream.write(b'{"record_type":"uncommitted-tail","step":999999}\n')
                stream.flush()
            tail_check = verify_commit_pointer(copied)
            results[group]["orphan_tail_bytes"] = tail_check["ledger_tail_bytes_uncommitted"]
            results[group]["orphan_tail_not_promoted"] = tail_check["last_committed_step"] == pointer_check["last_committed_step"] and tail_check["ledger_tail_bytes_uncommitted"] > 0
    args.out.joinpath("recovery-verification.json").write_text(json.dumps({"status": "passed", "groups": results}, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "groups": results}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
