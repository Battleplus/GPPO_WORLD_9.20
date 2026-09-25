from __future__ import annotations

from gppo_world.dataset import audit_manifest


def episode(split, policy, profile, tape, seed=1):
    return {
        "split": split,
        "behavior_policy": policy,
        "scenario_id": profile,
        "tape_id": tape,
        "seed": seed,
        "transition_count": 1,
    }


def complete_manifest():
    profiles = ("normal", "single", "sequential", "overlap", "burst", "long_gap", "weak_comm")
    policies = ("random_legal", "greedy", "gppo")
    episodes = []
    for split_index, split in enumerate(("train", "validation", "test")):
        for profile_index, profile in enumerate(profiles):
            for policy in policies:
                episodes.append(
                    episode(split, policy, profile, f"{split}-{profile}", 100 * split_index + profile_index)
                )
    return {"episodes": episodes, "files": {split: {} for split in ("train", "validation", "test")}}


def test_complete_grouped_manifest_passes():
    audit = audit_manifest(complete_manifest())
    assert audit["passed"]
    assert audit["split_overlap_count"] == 0


def test_same_tape_seed_across_splits_fails():
    manifest = complete_manifest()
    manifest["episodes"].append(episode("test", "gppo", "single", "train-single", 1))
    audit = audit_manifest(manifest)
    assert not audit["passed"]
    assert audit["split_overlap_count"] == 1


def test_missing_behavior_policy_fails():
    manifest = complete_manifest()
    manifest["episodes"] = [item for item in manifest["episodes"] if item["behavior_policy"] != "gppo"]
    assert not audit_manifest(manifest)["passed"]
