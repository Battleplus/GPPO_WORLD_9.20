"""Same-CUDA, fresh-process replay of a frozen PPO recovery boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from gppo_world.joint_gppo import (
    JOINT_PROTOCOL, ActionConditionedTemporalWorldModel, JointGraphPreferencePolicy, JointTrainConfig,
)
from gppo_world.joint_training import (
    GROUPS, _decision_probe, _gae_vector, _joint_logprob_rows, _make_optimizers,
    _nested_max_abs_delta, _replay_hidden, _restore_rng_snapshot,
    finite_optimizer, finite_parameters, ppo_preference_update, world_model_updates,
)
from gppo_world.m10_environment import M10Config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--uninterrupted-result", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite recovery report: {args.out}")
    if not torch.cuda.is_available():
        raise SystemExit("same-CUDA recovery requires the already selected CUDA environment")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load a fresh serialized boundary in this process. U's expected state is
    # immutable in the reference; R never shares optimizer-state objects with U.
    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    if reference.get("protocol") != JOINT_PROTOCOL or reference.get("group") not in GROUPS:
        raise ValueError("recovery reference identity/protocol mismatch")
    group = reference["group"]
    resolved = json.loads(args.resolved_config.read_text(encoding="utf-8"))
    if resolved["group_config"]["group"] != group:
        raise ValueError("resolved config group does not match recovery boundary")
    env_config = M10Config(**resolved["environment"])
    train_config = JointTrainConfig(**resolved["training"])
    runtime = reference["runtime_state"]
    device = torch.device("cuda")
    spec = GROUPS[group]
    uninterrupted = (torch.load(args.uninterrupted_result, map_location="cpu", weights_only=False)
                     if args.uninterrupted_result is not None else None)

    # Exact serialized environment, identities, masks, preference/history and
    # data order are part of the recovery contract.
    environment = runtime["environment"]
    saved_obs = runtime["observation"]
    current_obs = environment._observation()
    # The returned observation may contain the one-shot event/trigger signal
    # for the just-finished step, while the environment snapshot has already
    # cleared that latch. Compare the stable public task/UAV state and validate
    # the checkpoint's own environment digest separately.
    stable_public_fields = ("uavs", "tasks", "mask")
    stable_public_exact = all(np.array_equal(np.asarray(current_obs[key]), np.asarray(saved_obs[key]))
                              for key in stable_public_fields)
    stable_public_exact = bool(stable_public_exact and current_obs.get("version") == saved_obs.get("version")
                               and current_obs.get("time") == saved_obs.get("time")
                               and current_obs.get("public_entity_ids") == saved_obs.get("public_entity_ids"))
    expected_public_digest = runtime.get("public_digest")
    if expected_public_digest is None:
        # Update-reference artifacts predate the transaction checkpoint field;
        # bind their digest to the exact public observation captured at the
        # same boundary instead of treating a missing field as a mismatch.
        expected_public_digest = (
            int(saved_obs["version"]),
            tuple(float(x) for x in np.asarray(saved_obs["flat"]).reshape(-1)),
            tuple(bool(x) for x in np.asarray(saved_obs["mask"]).reshape(-1)),
        )
    rerendered_event = float(current_obs.get("event_signal", 0.0))
    saved_event = float(saved_obs.get("event_signal", 0.0))
    event_latch_consistent = (saved_event == rerendered_event or
                              (saved_event == 1.0 and rerendered_event == 0.0
                               and any(bool(value) for value in saved_obs.get("trigger_flags", {}).values())
                               and not any(bool(value) for value in current_obs.get("trigger_flags", {}).values())))
    current_digest = environment.public_snapshot_digest()
    # The environment's one-shot event bit is intentionally cleared after
    # returning the observation. Compare every stable digest component exactly;
    # validate that transient bit only via the explicit latch contract above.
    env_digest_matches = bool(
        current_digest[0] == expected_public_digest[0]
        and current_digest[1][:-1] == expected_public_digest[1][:-1]
        and current_digest[2] == expected_public_digest[2]
    )
    public_env_matches = bool(stable_public_exact and env_digest_matches and event_latch_consistent)
    mask_exact = np.array_equal(np.asarray(saved_obs["mask"], dtype=np.bool_),
                                np.asarray(current_obs["mask"], dtype=np.bool_))
    pref = runtime.get("episode_preference")
    preference_exact = (pref is None or (pref.shape == (2,) and bool(torch.isfinite(pref).all())
                       and abs(float(pref.sum()) - 1.0) <= 1e-6))
    identities = [
        (row["scenario_id"], row["episode_index"], row["time"], row["action"])
        for row in reference["transitions"]
    ]
    data_order_exact = (runtime["data_order"] == list(range(len(reference["transitions"])))
                        and identities == runtime["transition_identity"])
    illegal = [index for index, row in enumerate(reference["transitions"])
               if not bool(np.asarray(row["mask"], dtype=np.bool_)[int(row["action"])])]

    policy = JointGraphPreferencePolicy(env_config, history=True).to(device)
    world = ActionConditionedTemporalWorldModel().to(device) if spec.world_model else None
    optimizer, world_optimizer = _make_optimizers(policy, world, spec, train_config)
    policy.load_state_dict(reference["policy_state_dict"], strict=True)
    if world is not None:
        if reference.get("world_state_dict") is None:
            raise ValueError("world-enabled recovery boundary lacks the pre-update world state")
        world.load_state_dict(reference["world_state_dict"], strict=True)
    optimizer.load_state_dict(reference["policy_optimizer_state_dict"])
    if world_optimizer is not None:
        if reference.get("world_optimizer_state_dict") is None:
            raise ValueError("world-enabled recovery boundary lacks the pre-update world optimizer")
        world_optimizer.load_state_dict(reference["world_optimizer_state_dict"])
    preference_rng = np.random.default_rng(train_config.seed + 0xA17E)
    _restore_rng_snapshot(reference["rng_state"], preference_rng)
    world_batch_rng = np.random.default_rng(train_config.seed + 0x51A7)
    if reference.get("world_replay_rng_state") is not None:
        world_batch_rng.bit_generator.state = reference["world_replay_rng_state"]
    hidden = None if runtime["policy_hidden"] is None else runtime["policy_hidden"].to(device)
    world_hidden = None if runtime["world_hidden"] is None else runtime["world_hidden"].to(device)
    pref_device = None if pref is None else pref.to(device)

    # U and R must start from identical history and random state. Reconstruct
    # recurrent state on the same CUDA device and compare it before updating.
    replayed_hidden, _ = _replay_hidden(policy, None, runtime["episode_history"], device)
    history_delta = _nested_max_abs_delta(replayed_hidden, hidden)
    restored_probe = _decision_probe(policy, world, saved_obs, hidden, world_hidden, pref_device,
                                     use_events=spec.event_auxiliary, device=device)
    probe_delta = _nested_max_abs_delta(restored_probe, reference["preupdate_probe"])
    replay_logprob = _joint_logprob_rows(policy, reference["transitions"], device)
    behavior_rows = torch.as_tensor([row["old_log_prob"] for row in reference["transitions"]],
                                    dtype=torch.float32, device=device)
    behavior_logprob_delta = float((replay_logprob - behavior_rows).abs().max().detach())
    advantages, returns = _gae_vector(reference["transitions"], train_config, device)
    gae_finite = bool(torch.isfinite(advantages).all() and torch.isfinite(returns).all())

    # R: execute the exact saved batch once after fresh-process loading. The
    # serialized expected post-state was captured from uninterrupted U.
    update_metrics = ppo_preference_update(
        policy, reference["transitions"], optimizer, train_config, device, event_group=group,
    )
    replay_world_updates = int(reference.get("world_replay_updates", 0))
    world_update_rows = []
    if replay_world_updates:
        if world is None or world_optimizer is None:
            raise ValueError("world replay update requested for a group without a world optimizer")
        world_update_rows = world_model_updates(
            policy, world, reference["transitions"], world_optimizer, train_config, device,
            updates=replay_world_updates, event_enabled=bool(reference.get("world_event_enabled")),
            batch_rng=world_batch_rng,
        )
    expected_result = uninterrupted
    expected_policy = (expected_result["policy_state_dict"] if expected_result is not None
                       else reference["expected_post_policy_state_dict"])
    expected_policy_optimizer = (expected_result["policy_optimizer_state_dict"] if expected_result is not None
                                 else reference["expected_post_policy_optimizer_state_dict"])
    parameter_delta = _nested_max_abs_delta(policy.state_dict(), expected_policy)
    optimizer_delta = _nested_max_abs_delta(optimizer.state_dict(), expected_policy_optimizer)
    world_parameter_delta = None
    world_optimizer_delta = None
    post_probe_delta = None
    if world is not None:
        if expected_result is None or expected_result.get("world_state_dict") is None:
            raise ValueError("world-enabled U-R verification requires an uninterrupted result artifact")
        world_parameter_delta = _nested_max_abs_delta(world.state_dict(), expected_result["world_state_dict"])
        world_optimizer_delta = _nested_max_abs_delta(world_optimizer.state_dict(), expected_result["world_optimizer_state_dict"])
    post_hidden, post_world_hidden = _replay_hidden(
        policy, world, runtime["episode_history"], device,
    )
    post_probe = _decision_probe(
        policy, world, saved_obs, post_hidden, post_world_hidden, pref_device,
        use_events=spec.event_auxiliary, device=device,
    )
    if expected_result is not None:
        post_probe_delta = _nested_max_abs_delta(post_probe, expected_result["post_update_probe"])
    fixed_tol = float(reference["fixed_tolerances"]["same_cuda_abs"])
    checks = {
        "same_cuda_device": torch.cuda.get_device_name(0),
        "serialized_public_environment_exact": bool(public_env_matches),
        "stable_public_entities_mask_and_time_exact": bool(stable_public_exact),
        "environment_digest_exact": bool(env_digest_matches),
        "one_shot_event_latch_consistent": bool(event_latch_consistent),
        "public_mask_exact": bool(mask_exact),
        "preference_valid": bool(preference_exact),
        "data_order_and_transition_identity_exact": bool(data_order_exact),
        "illegal_actions": illegal,
        "history_replay_max_abs_delta": history_delta,
        "next_action_distribution_max_abs_delta": probe_delta,
        "behavior_logprob_max_abs_delta": behavior_logprob_delta,
        "gae_bootstrap_finite": gae_finite,
        "post_update_parameter_max_abs_delta": parameter_delta,
        "post_update_optimizer_max_abs_delta": optimizer_delta,
        "post_world_parameter_max_abs_delta": world_parameter_delta,
        "post_world_optimizer_max_abs_delta": world_optimizer_delta,
        "post_update_probe_max_abs_delta": post_probe_delta,
        "policy_parameters_finite": finite_parameters(policy),
        "optimizer_state_finite": finite_optimizer(optimizer),
        "optimizer_step_calls_R": 1,
        "world_optimizer_step_calls_R": len(world_update_rows),
    }
    passed = all((
        checks["serialized_public_environment_exact"], checks["public_mask_exact"], checks["preference_valid"],
        checks["data_order_and_transition_identity_exact"], not illegal,
        history_delta <= fixed_tol, probe_delta <= fixed_tol,
        behavior_logprob_delta <= fixed_tol, gae_finite,
        parameter_delta <= fixed_tol, optimizer_delta <= fixed_tol,
        world_parameter_delta is None or world_parameter_delta <= fixed_tol,
        world_optimizer_delta is None or world_optimizer_delta <= fixed_tol,
        post_probe_delta is None or post_probe_delta <= fixed_tol,
        checks["policy_parameters_finite"], checks["optimizer_state_finite"],
    ))
    report = {
        "status": "passed" if passed else "failed", "run_id": reference["run_id"],
        "group": group, "verification_mode": "fresh-process same-CUDA U-vs-R",
        "environment_steps_replayed": 0, "policy_optimizer_calls_R": 1,
        "world_optimizer_calls_R": len(world_update_rows), "reference_sha256": sha256(args.reference),
        "fixed_same_cuda_tolerance": fixed_tol, "checks": checks,
        "replay_update_metrics": update_metrics,
        "replay_world_update_rows": world_update_rows,
        "cpu_cuda_migration_diagnostic": "not run; not a pilot gate",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
